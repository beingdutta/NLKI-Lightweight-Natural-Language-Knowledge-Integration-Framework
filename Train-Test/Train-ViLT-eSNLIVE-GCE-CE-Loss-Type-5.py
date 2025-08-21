import os, re, json, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import datasets
from datasets import Dataset

from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering

# -------------------
# Loss Functions
# -------------------
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
        self.ce  = nn.CrossEntropyLoss()
        self.gce = GeneralisedCrossEntropy(q)
        self.lam = lam

    def forward(self, logits, targets_idx):
        return self.lam * self.ce(logits, targets_idx) + (1 - self.lam) * self.gce(logits, targets_idx)


# -------------------
# Dataset Class
# -------------------
class ESNLIDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.hf_dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['images']
        text_input = item['text_input']
        numeric_label = item['labels']

        encodings = self.processor(images=image, text=text_input,
                                   padding="max_length", truncation=True,
                                   return_tensors="pt")
        encodings = {k: v.squeeze() for k,v in encodings.items()}
        encodings['labels'] = torch.tensor(numeric_label, dtype=torch.float32)
        return encodings


# -------------------
# Collate Function
# -------------------
def collate_fn(batch, processor):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'token_type_ids': torch.stack(token_type_ids),
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': torch.stack(labels, dim=0)
    }


# -------------------
# Accuracy Functions
# -------------------
def evaluate(loader, model, processor, device, criterion=None, calc_loss=True, desc="Evaluating"):
    model.eval()
    matchScore, loopCounter, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            labels_idx = torch.argmax(batch['labels'], dim=1)

            matchScore += (preds == labels_idx).sum().item()
            loopCounter += len(labels_idx)

            if calc_loss and criterion is not None:
                total_loss += criterion(logits, labels_idx).item()

    accuracy = (matchScore / loopCounter) * 100
    avg_loss = total_loss / len(loader) if calc_loss else None
    return accuracy, avg_loss


# -------------------
# Main Training Loop
# -------------------
def main(args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # Labels
    label2id = {'contradiction':0, 'entailment':1, 'neutral':2}
    id2label = {0:'contradiction', 1:'entailment', 2:'neutral'}

    # Load CSVs
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    test_df  = pd.read_csv(args.test_csv)

    # Process dataset -> huggingface Dataset
    def process_split(df, explanation_path):
        img_paths, labels, hypo = [], [], []
        for i in range(len(df)):
            img_paths.append(os.path.join(args.image_dir, df['Flickr30kID'][i]))
            labels.append(df['gold_label'][i])
            hypo.append(df['hypothesis'][i])
        # attach llama explanations
        with open(explanation_path,'r') as f:
            llama_expl = [ln.strip().split('_')[1] for ln in f]
        hypo = [llama_expl[i] + ' [SEP] ' + hypo[i] for i in range(len(hypo))]

        onehot = []
        for lb in labels:
            s = [0]*3
            s[label2id[lb]] = 1
            onehot.append(s)

        data = {'text_input': hypo, 'labels': onehot, 'images': img_paths}
        hf_set = Dataset.from_dict(data)
        return hf_set.cast_column("images", datasets.Image())

    train_set = process_split(train_df, args.train_expl)
    val_set   = process_split(val_df, args.val_expl)
    test_set  = process_split(test_df, args.test_expl)

    # Model + Processor
    viltConfig = ViltConfig(max_position_embeddings=100, label2id=label2id, id2label=id2label)
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", config=viltConfig,
                                                     ignore_mismatched_sizes=True).to(device)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", model_max_length=100)

    # Dataset objects
    train_data = ESNLIDataset(train_set, processor)
    val_data   = ESNLIDataset(val_set, processor)
    test_data  = ESNLIDataset(test_set, processor)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, processor))
    val_loader   = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, processor))
    test_loader  = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, processor))

    # Loss & Optimizer
    criterion = CEplusGCE(q=args.q, lam=args.lam).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    best_val_acc = 0
    trainLoss = 0
    step_size = args.step_size

    for epoch in range(args.epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = model(**batch).logits
            targets_idx = batch['labels'].argmax(dim=1)
            loss = criterion(logits, targets_idx)

            trainLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 50 == 0:
                print(f"Epoch {epoch+1} Step {idx} -> Loss: {loss.item():.6f}")

            if idx != 0 and idx % step_size == 0:
                val_acc, val_loss = evaluate(val_loader, model, processor, device, criterion, True, "Validation")
                print(f"\nEpoch {epoch+1} | Step {idx} | Train Loss: {trainLoss/step_size:.6f} "
                      f"| Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.6f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model.save_pretrained(args.output_dir)
                    print("***** Model checkpoint saved *****")

                trainLoss = 0

    # Final Test Eval
    loaded_model = ViltForQuestionAnswering.from_pretrained(args.output_dir).to(device)
    test_acc, _ = evaluate(test_loader, loaded_model, processor, device, None, False, "Test")
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--train_expl", type=str, required=True)
    parser.add_argument("--val_expl", type=str, required=True)
    parser.add_argument("--test_expl", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--q", type=float, default=0.8)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--step_size", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default="./vilt-chkpt-esnli")
    args = parser.parse_args()
    main(args)
