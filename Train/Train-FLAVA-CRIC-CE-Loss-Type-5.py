import os
import json
import warnings
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import datasets
from datasets import Dataset
from transformers import FlavaModel, FlavaProcessor, FlavaConfig

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------
# CLI Arguments
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train FLAVA on CRIC dataset with explanations")

    parser.add_argument("--train_file", type=str, required=True, help="Path to train_questions.json")
    parser.add_argument("--val_file", type=str, required=True, help="Path to val_questions.json")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test_questions.json")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images directory")

    parser.add_argument("--error_files", nargs="+", required=True, help="Paths to error text files")
    parser.add_argument("--train_expl", type=str, required=True, help="Path to train explanations file")
    parser.add_argument("--val_expl", type=str, required=True, help="Path to val explanations file")
    parser.add_argument("--test_expl", type=str, required=True, help="Path to test explanations file")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--grad_acc_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--steps", type=int, default=50, help="Steps to print loss")
    parser.add_argument("--checkpoint", type=str, default="./flava-cric-checkpoint.pth", help="Path to save best model")

    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train on")

    return parser.parse_args()


# ---------------------------
# Utilities
# ---------------------------
def findUnique(targetList):
    uniqueList = []
    for word in targetList:
        if word not in uniqueList:
            uniqueList.append(word)
    return uniqueList


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

    def forward(self, input_ids, pixel_values, attention_mask):
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
# Training
# ---------------------------
def flat_accuracy(logits, indices):
    max_pos_logits = torch.argmax(logits, dim=1)
    max_pos_labels = torch.argmax(indices, dim=1)
    return (max_pos_logits == max_pos_labels).sum().item() / len(indices)


def calculateValidationAccuracy(model, val_loader, device, loss_fn):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids, attention_masks = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_masks, pixel_values=pixel_values)
            logits = output[0]
            loss = loss_fn(logits, labels)

            eval_loss += loss.item()
            eval_accuracy += flat_accuracy(logits, labels)

    return (eval_loss / len(val_loader)), (eval_accuracy / len(val_loader))


def main():
    args = parse_args()
    device = args.device

    # ---------------------------
    # Load JSON
    # ---------------------------
    with open(args.train_file, "r") as file:
        train_json = json.load(file)
    with open(args.val_file, "r") as file:
        val_json = json.load(file)
    with open(args.test_file, "r") as file:
        test_json = json.load(file)

    # Error indices
    indexToExclude = []
    for ef in args.error_files:
        with open(ef, "r") as file:
            for line in file:
                indexToExclude.append(int(line.strip()))

    # ---------------------------
    # Train Set
    # ---------------------------
    questionList, answerList, imgList = [], [], []
    for i in tqdm(range(len(train_json)), desc="Processing train set"):
        if i in indexToExclude:
            continue
        pointer = train_json[i]
        questionList.append(pointer['question'])
        answerList.append(pointer['answer'])
        imgList.append(os.path.join(args.image_dir, str(pointer['image_id']) + ".jpg"))

    explanationTrain = [line.strip().split('_')[1] for line in open(args.train_expl, "r")]
    questionList = [explanationTrain[i] + " [SEP] " + questionList[i] for i in range(len(questionList))]

    # ---------------------------
    # Val Set
    # ---------------------------
    questionList_val, answerList_val, imgList_val = [], [], []
    for i in range(len(val_json)):
        pointer = val_json[i]
        questionList_val.append(pointer['question'])
        answerList_val.append(pointer['answer'])
        imgList_val.append(os.path.join(args.image_dir, str(pointer['image_id']) + ".jpg"))

    explanationVal = [line.strip().split('_')[1] for line in open(args.val_expl, "r")]
    questionList_val = [explanationVal[i] + " [SEP] " + questionList_val[i] for i in range(len(questionList_val))]

    # ---------------------------
    # Test Set
    # ---------------------------
    questionList_test, answerList_test, imgList_test = [], [], []
    for i in range(len(test_json)):
        pointer = test_json[i]
        questionList_test.append(pointer['question'])
        answerList_test.append(pointer['answer'])
        imgList_test.append(os.path.join(args.image_dir, str(pointer['image_id']) + ".jpg"))

    explanationTest = [line.strip().split('_')[1] for line in open(args.test_expl, "r")]
    questionList_test = [explanationTest[i] + " [SEP] " + questionList_test[i] for i in range(len(questionList_test))]

    # ---------------------------
    # Label mapping
    # ---------------------------
    answerPool = answerList + answerList_val + answerList_test
    uniqueAnsList = findUnique(answerPool)
    mapping = {word: idx for idx, word in enumerate(uniqueAnsList)}
    reverse_mapping = {idx: word for word, idx in mapping.items()}
    numOfClasses = max(mapping.values())

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
    # Model
    # ---------------------------
    flava_config = FlavaConfig(label2id=mapping, id2label=reverse_mapping,
                               max_position_embeddings=100, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2)
    flava_model = FlavaModel.from_pretrained("facebook/flava-full", config=flava_config, ignore_mismatched_sizes=True)
    flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full", model_max_length=100)

    model = FlavaWrapper(flava_model, numOfClasses).to(device)

    # ---------------------------
    # Dataloaders
    # ---------------------------
    train_loader = DataLoader(CustomDataset(train_set, flava_processor), batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)
    val_loader = DataLoader(CustomDataset(val_set, flava_processor), batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True)
    test_loader = DataLoader(CustomDataset(test_set, flava_processor), batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True)

    # ---------------------------
    # Loss & Optimizer
    # ---------------------------
    loss_fn = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------------------
    # Training
    # ---------------------------
    best_val_accuracy = 0
    train_loss = 0

    for epoch in range(args.epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            input_ids, attention_masks = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_masks, pixel_values=pixel_values)
            logits = output[0]
            loss = loss_fn(logits, labels)

            if idx % args.steps == 0:
                print(f"{idx} -> Loss: {round(loss.item(), 6)}")

            loss.backward()
            if idx % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

            if idx != 0 and idx % 3000 == 0:
                val_loss, val_acc = calculateValidationAccuracy(model, val_loader, device, loss_fn)
                print(f"Epoch {epoch+1}, Train Loss: {round(train_loss/3000, 6)}, Val Acc: {round(val_acc * 100, 6)}%")
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    torch.save(model.state_dict(), args.checkpoint)
                    print("******** Model Checkpoint Saved ********")
                train_loss = 0
                model.train()

    print("\nTraining complete. Best checkpoint saved at:", args.checkpoint)


if __name__ == "__main__":
    main()
