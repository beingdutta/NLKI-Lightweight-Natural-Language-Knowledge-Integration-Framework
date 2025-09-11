import os
import json
import argparse
from tqdm.auto import tqdm

from datasets import Dataset
import datasets
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig


# ---------------------------
# Custom Dataset
# ---------------------------
class CricDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.processor = processor
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encodings = self.processor(
            images=item["images"],
            text=item["questions"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encodings = {k: v.squeeze() for k, v in encodings.items()}
        encodings["labels"] = torch.tensor(item["scores"], dtype=torch.float32)
        return encodings


# ---------------------------
# Symmetric Cross-Entropy
# ---------------------------
class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta

    def forward(self, logits, targets_idx):
        ce = F.cross_entropy(logits, targets_idx)
        pred = F.softmax(logits, dim=1).clamp(1e-7, 1.0)
        onehot = F.one_hot(targets_idx, logits.size(1)).float()
        rce = (-torch.sum(pred * onehot, dim=1)).mean()
        return self.alpha * ce + self.beta * rce


# ---------------------------
# Collate Function
# ---------------------------
def collate_fn(batch, processor):
    input_ids = [item["input_ids"] for item in batch]
    pixel_values = [item["pixel_values"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    token_type_ids = [item["token_type_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "token_type_ids": torch.stack(token_type_ids),
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": torch.stack(labels, dim=0),
    }


# ---------------------------
# Helper Functions
# ---------------------------
def findUnique(targetList):
    uniqueList = []
    for word in targetList:
        if word not in uniqueList:
            uniqueList.append(word)
    return uniqueList


def calculateAccuracyVal(model, processor, val_dataset_object, device, batch_size=32):
    val_dataloader = DataLoader(
        val_dataset_object, batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    model.eval()
    matchScore, totalLoss, loopCounter = 0, 0.0, 0

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            targets = torch.argmax(batch["labels"], dim=1)
            matchScore += (predictions == targets).sum().item()
            totalLoss += outputs.loss.item()
            loopCounter += batch["labels"].size(0)

    accuracyVal = (matchScore / loopCounter) * 100
    avgLoss = totalLoss / len(val_dataloader)
    return accuracyVal, avgLoss


def calculateAccuracyTest(model, processor, test_dataset_object, device, batch_size=32):
    test_dataloader = DataLoader(
        test_dataset_object, batch_size=batch_size,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    model.eval()
    matchScore, loopCounter = 0, 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            targets = torch.argmax(batch["labels"], dim=1)
            matchScore += (predictions == targets).sum().item()
            loopCounter += batch["labels"].size(0)

    print(f"\nTotal Questions {loopCounter}")
    print(f"Correctly classified {matchScore}")
    print(f"Accuracy is: {(matchScore / loopCounter) * 100}")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train VILT on CRIC with explanations")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--error_files", nargs="+", required=True)
    parser.add_argument("--train_expl", type=str, required=True)
    parser.add_argument("--val_expl", type=str, required=True)
    parser.add_argument("--test_expl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default="./vilt-checkpoint")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load datasets
    with open(args.train_file) as f: train_json = json.load(f)
    with open(args.val_file) as f: val_json = json.load(f)
    with open(args.test_file) as f: test_json = json.load(f)

    # Error indices
    indexToExclude = []
    for ef in args.error_files:
        with open(ef, "r") as file:
            for line in file:
                indexToExclude.append(int(line.strip()))

    # Train set
    questionList, answerList, imgList = [], [], []
    for i, pointer in enumerate(train_json):
        if i in indexToExclude: continue
        questionList.append(pointer["question"])
        answerList.append(pointer["answer"])
        imgList.append(pointer["image_id"])

    explanationTrain = []
    with open(args.train_expl, "r") as file:
        for lines in file:
            explanationTrain.append(lines.strip().split("_")[1])
    questionList = [explanationTrain[i] + " [SEP] " + questionList[i] for i in range(len(questionList))]

    # Val set
    questionList_val, answerList_val, imgList_val = [], [], []
    for pointer in val_json:
        questionList_val.append(pointer["question"])
        answerList_val.append(pointer["answer"])
        imgList_val.append(pointer["image_id"])

    explanationVal = []
    with open(args.val_expl, "r") as file:
        for lines in file:
            explanationVal.append(lines.strip().split("_")[1])
    questionList_val = [explanationVal[i] + " [SEP] " + questionList_val[i] for i in range(len(questionList_val))]

    # Test set
    questionList_test, answerList_test, imgList_test = [], [], []
    for pointer in test_json:
        questionList_test.append(pointer["question"])
        answerList_test.append(pointer["answer"])
        imgList_test.append(pointer["image_id"])

    explanationTest = []
    with open(args.test_expl, "r") as file:
        for lines in file:
            explanationTest.append(lines.strip().split("_")[1])
    questionList_test = [explanationTest[i] + " [SEP] " + questionList_test[i] for i in range(len(questionList_test))]

    # Mapping
    answerPool = answerList + answerList_val + answerList_test
    mapping, counter = {}, 0
    for word in findUnique(answerPool):
        mapping[word] = counter
        counter += 1
    reverse_mapping = {v: k for k, v in mapping.items()}
    numOfClasses = max(mapping.values())

    # Convert HF datasets
    def build_dataset(qList, aList, iList, split):
        labels, scores, imgPaths = [], [], []
        for i in range(len(aList)):
            labels.append(mapping[aList[i]])
            s = [0] * (numOfClasses + 1)
            s[mapping[aList[i]]] = 1
            scores.append(s)
            imgPaths.append(os.path.join(args.image_dir, f"{iList[i]}.jpg"))
        return Dataset.from_dict({"questions": qList, "labels": labels, "scores": scores, "images": imgPaths}).cast_column("images", datasets.Image())

    modified_train_set = build_dataset(questionList, answerList, imgList, "train")
    modified_val_set = build_dataset(questionList_val, answerList_val, imgList_val, "val")
    modified_test_set = build_dataset(questionList_test, answerList_test, imgList_test, "test")

    # Model
    config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-mlm",
        id2label=reverse_mapping,
        label2id=mapping,
    ).to(device)

    criterion = SymmetricCrossEntropy(alpha=0.1, beta=1.0).to(device)

    train_dataset_object = CricDataset(modified_train_set, processor)
    val_dataset_object = CricDataset(modified_val_set, processor)
    test_dataset_object = CricDataset(modified_test_set, processor)

    train_dataloader = DataLoader(
        train_dataset_object,
        collate_fn=lambda b: collate_fn(b, processor),
        shuffle=True,
        batch_size=args.batch_size,
    )

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    trainLoss, best_val_accuracy = 0, 0

    print("\nFinetuning Begins in Training Loop")
    model.train()
    for epoch in tqdm(range(args.epochs)):
        for idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            logits = output.logits
            targets_idx = batch["labels"].argmax(dim=1)
            loss = criterion(logits, targets_idx)

            trainLoss += loss.item()
            if idx % 50 == 0:
                print(idx, "-> Loss:", round(loss.item(), 6))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx != 0 and idx % 3000 == 0:
                model.eval()
                acc_score_val, validationLoss = calculateAccuracyVal(model, processor, val_dataset_object, device)
                print(f"\nEpoch: {epoch+1}, Train Loss: {round((trainLoss/20), 4)}, Validation Accuracy: {round(acc_score_val, 4)}")

                if acc_score_val > best_val_accuracy:
                    best_val_accuracy = acc_score_val
                    model.save_pretrained(args.checkpoint)
                    print("*****Model Chkpt Saved******\n\n")

                trainLoss = 0
                model.train()

    # Final test
    loaded_model = ViltForQuestionAnswering.from_pretrained(args.checkpoint).to(device)
    calculateAccuracyTest(loaded_model, processor, test_dataset_object, device)


if __name__ == "__main__":
    main()
