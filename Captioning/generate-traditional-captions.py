import os
import json
import torch
import argparse
from tqdm.auto import tqdm
from PIL import Image as img
from torch.utils.data import DataLoader
from datasets import Dataset, Image as HFImage
from transformers import BlipProcessor, BlipForConditionalGeneration

def pil_collate_fn(batch):
    return batch  # return list of PIL images as-is

# ---------------------------
# Dataset Class
# ---------------------------
class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]['images']


# ---------------------------
# Argument Parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate image captions using BLIP")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to CRIC JSON file (test set)')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output', type=str, default='blip_image_captions_full_test_set.txt', help='Output caption file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='Device to use (cuda or cpu)')
    parser.add_argument('--max_samples', type=int, default=100, help='Number of samples to process')

    return parser.parse_args()


# ---------------------------
# Main Function
# ---------------------------
def main():
    args = parse_args()
    device = f'cuda:{args.device}'

    # Load BLIP model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    # Load dataset
    with open(args.dataset_path, "r") as file:
        test_json = json.load(file)

    img_paths, questions = [], []
    for entry in test_json:
        questions.append(entry['question'])  # not used, but preserved
        img_paths.append(os.path.join(args.image_dir, f"{entry['image_id']}.jpg"))

    questions = questions[:args.max_samples]
    img_paths = img_paths[:args.max_samples]

    hf_dict = {'questions': questions, 'images': img_paths}
    hf_dataset = Dataset.from_dict(hf_dict).cast_column("images", HFImage())

    dataset = ImageCaptionDataset(hf_dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=pil_collate_fn)

    print(f'Total batches: {len(dataloader)}')

    # Caption generation
    captions = []
    print('Generating captions...')
    for batch_images in tqdm(dataloader):
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        output_ids = model.generate(**inputs)
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True, max_length=25)
        captions.extend(decoded)

    # Save captions
    print(f'Writing to {args.output}')
    with open(args.output, 'w') as file:
        for idx, caption in enumerate(captions):
            file.write(f"{idx}_{caption}\n")

    print(f"\nCaptions written to: {args.output}")
    print(f"Total captions: {len(captions)}")
    print("Done!")


if __name__ == "__main__":
    main()
