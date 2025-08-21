import argparse
import json
import os
from tqdm.auto import tqdm
from datasets import Dataset
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViLT for CRIC with explanations")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--error_files", nargs="+", required=True)
    parser.add_argument("--train_expl", type=str, required=True)
    parser.add_argument("--val_expl", type=str, required=True)
    parser.add_argument("--test_expl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save_dir", type=str, default="./vilt-checkpoints")
    return parser.parse_args()


class CricDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encodings = self.processor(images=item["images"], text=item["questions"],
                                   padding="max_length", truncation=True, return_tensors="pt")
        encodings = {k: v.squeeze() for k, v in encodings.items()}
        encodings["labels"] = torch.tensor(item["scores"], dtype=torch.float32)
        return encodings


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


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_explanations(path):
    explanations = []
    with open(path, "r") as f:
        for line in f:
            explanations.append(line.strip().split("_")[1])
    return explanations


def main():
    args = parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # Load datasets
    train_json = load_json(args.train_file)
    val_json = load_json(args.val_file)
    test_json = load_json(args.test_file)

    print(len(train_json), len(val_json), len(test_json))

    # Exclude bad indices
    indexToExclude = []
    for ef in args.error_files:
        with open(ef, "r") as f:
            for line in f:
                indexToExclude.append(int(line.strip()))

    # Process training set
    questionList, answerList, imgList = [], [], []
    for i in tqdm(range(len(train_json))):
        if i in indexToExclude:
            continue
        pointer = train_json[i]
        questionList.append(pointer["question"])
        answerList.append(pointer["answer"])
        imgList.append(pointer["image_id"])

    explanationTrain = load_explanations(args.train_expl)
    questionList = [explanationTrain[i] + " [SEP] " + questionList[i] for i in range(len(questionList))]

    # Process validation set
    questionList_val, answerList_val, imgList_val = [], [], []
    for i in tqdm(range(len(val_json))):
        pointer = val_json[i]
        questionList_val.append(pointer["question"])
        answerList_val.append(pointer["answer"])
        imgList_val.append(pointer["image_id"])
    explanationVal = load_explanations(args.val_expl)
    questionList_val = [explanationVal[i] + " [SEP] " + questionList_val[i] for i in range(len(questionList_val))]

    # Process test set
    questionList_test, answerList_test, imgList_test = [], [], []
    for i in tqdm(range(len(test_json))):
        pointer = test_json[i]
        questionList_test.append(pointer["question"])
        answerList_test.append(pointer["answer"])
        imgList_test.append(pointer["image_id"])
    explanationTest = load_explanations(args.test_expl)
    questionList_test = [explanationTest[i] + " [SEP] " + questionList_test[i] for i in range(len(questionList_test))]

    # Label mapping
    answerPool = answerList + answerList_val + answerList_test
    uniqueAnsList = list(dict.fromkeys(answerPool))
    mapping = {word: idx for idx, word in enumerate(uniqueAnsList)}
    reverse_mapping = {idx: word for word, idx in mapping.items()}
    numOfClasses = len(mapping)

    def prepare_dataset(qList, aList, iList):
        labels, scores, imgPaths = [], [], []
        for ans in aList:
            labels.append(mapping[ans])
        for ans in aList:
            s = [0] * numOfClasses
            s[mapping[ans]] = 1
            scores.append(s)
        for img in iList:
            imgPaths.append(os.path.join(args.image_dir, str(img) + ".jpg"))
        return Dataset.from_dict({"questions": qList, "labels": labels, "scores": scores, "images": imgPaths}).cast_column("images", datasets.Image())

    modified_train_set = prepare_dataset(questionList, answerList, imgList)
    modified_val_set = prepare_dataset(questionList_val, answerList_val, imgList_val)
    modified_test_set = prepare_dataset(questionList_test, answerList_test, imgList_test)

    # Model
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", id2label=reverse_mapping, label2id=mapping).to(device)

    train_dataset_object = CricDataset(modified_train_set, processor)
    val_dataset_object = CricDataset(modified_val_set, processor)
    test_dataset_object = CricDataset(modified_test_set, processor)

    train_dataloader = DataLoader(train_dataset_object, collate_fn=lambda b: collate_fn(b, processor),
                                  shuffle=True, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_val_accuracy = 0
    model.train()
    for epoch in range(args.epochs):
        trainLoss = 0
        for idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            trainLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 50 == 0:
                print(f"Epoch {epoch+1}, Step {idx}, Loss {loss.item():.6f}")

        # Save after every epoch
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        print(f"Model checkpoint saved at {args.save_dir}")


if __name__ == "__main__":
    main()
