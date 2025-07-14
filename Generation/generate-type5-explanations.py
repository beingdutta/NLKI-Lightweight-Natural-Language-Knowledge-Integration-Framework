import os
import json
import torch
import warnings
import argparse
from PIL import Image
from tqdm.auto import tqdm
from transformers import pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Generate explanation
# ---------------------------
def generate_llama_output(prompt, pipe):
    messages = [
        {"role": "system", "content": "Through reasoning, given some context and a conclusion, you can find the most plausible explanation"},
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        formatted_prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][len(formatted_prompt):]


# ---------------------------
# Argument Parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Type-5 LLaMA3 explanations for CRIC Test")
    parser.add_argument('--test_json', type=str, required=True)
    parser.add_argument('--dense_caption', type=str, required=True)
    parser.add_argument('--region_caption', type=str, required=True)
    parser.add_argument('--object_caption', type=str, required=True)
    parser.add_argument('--traditional_caption', type=str, required=True)
    parser.add_argument('--retrieved_facts', type=str, required=True)
    parser.add_argument('--output', type=str, default='Type-5-Explanation-CRIC-Test.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_samples', type=int, default=None)
    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        token='YOUR_TOKEN',
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device,
        pad_token_id=128009
    )

    with open(args.test_json, "r") as f:
        test_json = json.load(f)

    question_list, answer_list, img_list = [], [], []
    for entry in test_json:
        question_list.append(entry['question'])
        answer_list.append(entry['answer'])
        img_list.append(entry['image_id'])

    with open(args.traditional_caption, 'r') as f:
        conventional_captions = [line.strip().split('_', 1)[1] for line in f]

    with open(args.retrieved_facts, 'r') as f:
        retrieved_facts = [line.strip().split('_', 1)[1].split(',') for line in f]

    with open(args.dense_caption, 'r') as f:
        dense_captions = [line.strip().split('_', 1)[1] for line in f]

    with open(args.region_caption, 'r') as f:
        region_captions = [line.strip().split('_', 1)[1].split(',') for line in f]

    with open(args.object_caption, 'r') as f:
        object_captions = [line.strip().split('_', 1)[1].split(',') for line in f]

    if args.max_samples:
        question_list = question_list[:args.max_samples]
        answer_list = answer_list[:args.max_samples]
        img_list = img_list[:args.max_samples]
        gt_explanations = gt_explanations[:args.max_samples]
        conventional_captions = conventional_captions[:args.max_samples]
        retrieved_facts = retrieved_facts[:args.max_samples]
        dense_captions = dense_captions[:args.max_samples]
        region_captions = region_captions[:args.max_samples]
        object_captions = object_captions[:args.max_samples]

    print(f"\nSanity check: {len(question_list)} examples")

    print('\nGenerating Explanations...')
    with open(args.output, 'w') as f:
        for idx in tqdm(range(len(question_list))):
            prompt = f'''Ground the question with image description, region based captions carefully

            Image Description: {dense_captions[idx]}.
            Objects Present: {object_captions[idx]}
            Region Captions: {region_captions[idx]}
            Question: {question_list[idx]}.
            Retrieved Facts: {retrieved_facts[idx]}

            You need to generate a short single line explanation within 10 words, based solely only on the objects present, image description, region captions and question which can help VQA models derive the answer. 
            You might focus on the relevant retrieved Facts if enough information is not derivable from the image description and region caption provided.
            Do not introduce any new objects which is not present in the image and question back.
            The forbidden words are: "image description", "captions", "region captions". Dont start with "Here is single-line explanation", "The missing knowledge fact that is", "The missing knowledge fact that helps", "most plausible explanation is", "Missing knowledge fact"
            '''
            explanation = generate_llama_output(prompt, pipe)
            f.write(f"{idx}_{explanation.strip()}\n")

    print(f"\nExplanations written to {args.output}")


if __name__ == "__main__":
    main()
