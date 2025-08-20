#!/usr/bin/env python3
import os
import warnings
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import datasets
from datasets import Dataset
from PIL import Image
import numpy as np
import pandas as pd

from transformers import FlavaModel, FlavaProcessor, FlavaConfig

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------
# Parse CLI arguments
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Flava with CRIC dataset + explanations")

    # Dataset paths
    parser.add_argument("--train_file", type=str, required=True, help="Path to train_questions.json")
    parser.add_argument("--val_file", type=str, required=True, help="Path to val_questions.json")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test_questions.json")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing CRIC images")
    parser.add_argument("--error_files", nargs="+", required=True, help="Paths to error text files (error1.txt, error2.txt, etc.)")

    # Explanation files
    parser.add_argument("--train_expl", type=str, required=True, help="Path to explanations for train set")
    parser.add_argument("--val_expl", type=str, required=True, help="Path to explanations for val set")
    parser.add_argument("--test_expl", type=str, required=True, help="Path to explanations for test set")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer")
    parser.add_argument("--steps", type=int, default=50, help="Steps to print loss")
    parser.add_argument("--grad_acc_steps", type=int, default=2, help="Gradient accumulation steps")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 / cpu)")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default="./flava-chkpt-cric.pth", help="Path to save best model")

    return parser.parse_args()


# ---------------------------
# Helper Functions
# ---------------------------
def findUnique(targetList):
    uniqueList = []
    for word in targetList:
        if word not in uniqueList:
            uniqueList.append(word)
    return uniqueList


# ---------------------------
# Loss Functions
# ---------------------------
class GeneralisedCrossEntropy(nn.Module):
    def __init__(self, q: float = 0.7):
        super().__init__()
        assert 0 < q <= 1, "q must be in (0,1]"
        self.q = q

    def forward(self, logits, targets_idx):
        probs = F.softmax(logits, dim=1).clamp(1e-7, 1.0)
        p_true = probs.gather(1, targets_idx.unsqueeze(1)).squeeze()
        loss = (1.0 - p_true.pow(self.q)) / self.q
        return loss.mean()


class CEplusGCE(nn.Module):
    def __init__(self, q: float = 0.7, lam: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.gce = GeneralisedCrossEntropy(q)
        self.lam = lam

    def forward(self, logits, targets_idx):
        return self.lam * self.ce(logits, targets_idx) + (1 - self.lam) * self.gce(logits, targets_idx)


# ---------------------------
# Model Wrapper
# ---------------------------
class FlavaWrapper(nn.Module):
    def __init__(self, model, numOfClasses):
        super(FlavaWrapper, self).__init__()
        self.flava_model = model
        self.classifier = nn.Sequential(
            nn.Linear(768, 1536, bias=True),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, numOfClasses + 1)
        )

    def forward(self, input_ids, pixel_values, attention_mask, fact_embeds=None):
        flava_output = self.flava_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        pooled_output = flava_output.multimodal_output.pooler_output
        logits = self.classifier(pooled_output)
        return (logits,)


# ---------------------------
# Custom Dataset
# ---------------------------
class CustomDataset(Dataset):
    def __init__(self, df, processor):
        self.dataset_ = df
        self.processor = processor

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, idx):
        images = self.dataset_[idx]['images']
        questions = self.dataset_[idx]['questions']
        labels = self.dataset_[idx]['labels']

        encoding = self.processor(images, questions, padding=True, truncation=True, return_tensors='pt')
        labels = torch.tensor(labels, dtype=torch.float32)
        encoding['labels'] = labels
        return encoding


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    batch = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'pixel_values': torch.stack(pixel_values),
        'token_type_ids': torch.stack(token_type_ids),
        'labels': torch.stack(labels, dim=0)
    }
    return batch


# ---------------------------
# Accuracy
# ---------------------------
def flat_accuracy(logits, indices):
    max_pos_logits = torch.argmax(logits, dim=1)
    max_pos_labels = torch.argmax(indices, dim=1)
    acc = (max_pos_logits == max_pos_labels).sum().item() / len(indices)
    return acc


# ---------------------------
# Main Training & Eval
# ---------------------------
def main():
    args = parse_args()
    device = args.device

    # ---------------------------
    # Load JSON data
    # ---------------------------
    with open(args.train_file, "r") as file:
        train_json = json.load(file)
    with open(args.val_file, "r") as file:
        val_json = json.load(file)
    with open(args.test_file, "r") as file:
        test_json = json.load(file)

    # Load error indices
    indexToExclude = []
    for path in args.error_files:
        with open(path, "r") as file:
            for line in file:
                indexToExclude.append(int(line.strip()))

    # ---------------------------
    # Build Training set
    # ---------------------------
    questionList, answerList, imgList, k_triplet = [], [], [], []
    for i in tqdm(range(len(train_json))):
        if i in indexToExclude:
            continue
        pointer = train_json[i]
        questionList.append(pointer['question'])
        answerList.append(pointer['answer'])
        imgList.append(os.path.join(args.image_dir, f"{pointer['image_id']}.jpg"))
        k_triplet.append(" ".join(pointer['sub_graph']['knowledge_items'][0]['triplet']) + ". ")

    explanationTrain = [line.strip().split('_')[1] for line in open(args.train_expl, "r")]
    questionList = [explanationTrain[i] + " [SEP] " + questionList[i] for i in range(len(questionList))]

    # ---------------------------
    # Validation
    # ---------------------------
    questionList_val, answerList_val, imgList_val, k_triplet_val = [], [], [], []
    for entry in val_json:
        questionList_val.append(entry['question'])
        answerList_val.append(entry['answer'])
        imgList_val.append(os.path.join(args.image_dir, f"{entry['image_id']}.jpg"))
        k_triplet_val.append(" ".join(entry['sub_graph']['knowledge_items'][0]['triplet']) + ". ")

    explanationVal = [line.strip().split('_')[1] for line in open(args.val_expl, "r")]
    questionList_val = [explanationVal[i] + " [SEP] " + questionList_val[i] for i in range(len(questionList_val))]

    # ---------------------------
    # Test
    # ---------------------------
    questionList_test, answerList_test, imgList_test, k_triplet_test = [], [], [], []
    for entry in test_json:
        questionList_test.append(entry['question'])
        answerList_test.append(entry['answer'])
        imgList_test.append(os.path.join(args.image_dir, f"{entry['image_id']}.jpg"))
        k_triplet_test.append(" ".join(entry['sub_graph']['knowledge_items'][0]['triplet']) + ". ")

    explanationTest = [line.strip().split('_')[1] for line in open(args.test_expl, "r")]
    questionList_test = [explanationTest[i] + " [SEP] " + questionList_test[i] for i in range(len(questionList_test))]

    # ---------------------------
    # Mapping Labels
    # ---------------------------
    answerPool = answerList + answerList_val + answerList_test
    uniqueAnsList = findUnique(answerPool)
    mapping = {word: idx for idx, word in enumerate(uniqueAnsList)}
    reverse_mapping = {idx: word for word, idx in mapping.items()}
    numOfClasses = max(mapping.values())

    # Build HF datasets
    def build_dataset(qs, ans, imgs):
        scores = []
        for a in ans:
            s = [0] * (numOfClasses + 1)
            s[mapping[a]] = 1
            scores.append(s)
        return Dataset.from_dict({'questions': qs, 'labels': scores, 'images': imgs}).cast_column("images", datasets.Image())

    train_set = build_dataset(questionList, answerList, imgList)
    val_set = build_dataset(questionList_val, answerList_val, imgList_val)
    test_set = build_dataset(questionList_test, answerList_test, imgList_test)

    # ---------------------------
    # Load Model
    # ---------------------------
    flava_config = FlavaConfig(label2id=mapping, id2label=reverse_mapping, max_position_embeddings=100,
                               attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2)
    flava_model = FlavaModel.from_pretrained("facebook/flava-full", config=flava_config, ignore_mismatched_sizes=True)
    flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full", model_max_length=100)

    model = FlavaWrapper(flava_model, numOfClasses).to(device)

    # ---------------------------
    # DataLoader
    # ---------------------------
    train_loader = DataLoader(CustomDataset(train_set, flava_processor), batch_size=args.batch_size,
                              collate_fn=collate_fn, shuffle=True, pin_memory=True)
    val_loader = DataLoader(CustomDataset(val_set, flava_processor), batch_size=args.batch_size,
                            collate_fn=collate_fn, shuffle=False, pin_memory=True)
    test_loader = DataLoader(CustomDataset(test_set, flava_processor), batch_size=args.batch_size,
                             collate_fn=collate_fn, shuffle=False, pin_memory=True)

    # ---------------------------
    # Loss & Optimizer
    # ---------------------------
    criterion = CEplusGCE(q=0.7, lam=0.4).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------------------
    # Training Loop
    # ---------------------------
    best_val_accuracy = 0
    train_loss = 0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        for idx, batch in enumerate(train_loader):
            input_ids, attention_masks = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_masks, pixel_values=pixel_values)
            logits = output[0]
            targets_idx = labels.argmax(dim=1)
            loss = criterion(logits, targets_idx)

            if idx % args.steps == 0:
                print(f"{idx} -> Loss: {round(loss.item(), 6)}")

            loss.backward()
            if idx % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        print(f"Epoch {epoch+1} finished.")

    # Save final model
    torch.save(model.state_dict(), args.checkpoint)
    print(f"\n✅ Model saved at {args.checkpoint}")


if __name__ == "__main__":
    main()
