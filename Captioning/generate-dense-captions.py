import argparse
import os
import json
import torch
import warnings
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Dataset Class
# -----------------------------
class CRICDataset(Dataset):
    def __init__(self, image_dir, img_list, transform=None):
        self.image_dir = image_dir
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.img_list[idx]}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return idx, image

# -----------------------------
# Collate Function
# -----------------------------
def custom_collate_fn(batch):
    indices, images = zip(*batch)
    return list(indices), list(images)

# -----------------------------
# Florence Inference
# -----------------------------
def run_florence_batch(task_prompt, images, processor, model):
    repeated_prompts = [task_prompt] * len(images)
    inputs = processor(text=repeated_prompts, images=images, return_tensors="pt", padding=True).to('cuda', torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    parsed_answers = [
        processor.post_process_generation(
            text,
            task=task_prompt,
            image_size=(images[i].width, images[i].height)
        )
        for i, text in enumerate(generated_texts)
    ]
    return parsed_answers

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate dense captions using Florence-2 model")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to CRIC JSON file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output', type=str, default='denseCaptions.txt', help='Path to save output captions')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for processing images')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--max_samples', type=int, default=100, help='Number of samples to process')
    return parser.parse_args()

# -----------------------------
# Main Function
# -----------------------------
def main():
    args = parse_args()
    device = f'cuda:{args.device}'

    # Load JSON
    with open(args.dataset_path, "r") as file:
        dataset_json = json.load(file)

    questionList, answerList, imgList, gt_explanation = [], [], [], []
    for i, pointer in enumerate(dataset_json):
        questionList.append(pointer['question'])
        answerList.append(pointer['answer'])
        imgList.append(pointer['image_id'])
        gt_explanation.append(' '.join(pointer['sub_graph']['knowledge_items'][0]['triplet']) + '.')

    # Subset
    questionList = questionList[:args.max_samples]
    imgList = imgList[:args.max_samples]

    # Data preparation
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB"))
    ])
    dataset = CRICDataset(image_dir=args.image_dir, img_list=imgList, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Load model
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Run caption generation
    task_prompt = '<MORE_DETAILED_CAPTION>'
    with open(args.output, 'w') as file:
        for batch_indices, batch_images in tqdm(dataloader):
            denseCaptionsBatch = run_florence_batch(task_prompt, batch_images, processor, model)
            captions = [caption_dict.get('<MORE_DETAILED_CAPTION>', 'No caption').replace("\n", "") for caption_dict in denseCaptionsBatch]
            file.writelines(f"{caption}\n" for caption in captions)

    print(f"✅ Captions saved to {args.output}")

# -----------------------------
if __name__ == "__main__":
    main()
