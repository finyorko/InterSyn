import json
import torch
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
import re
import os
import argparse
import logging
from typing import List

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model & Image Preprocessing Functions ---

def split_model(model_name: str) -> dict:
    """
    Creates a device map to distribute a large model across available GPUs.
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size == 0:
        logging.warning("No CUDA devices found. Running on CPU, which will be very slow.")
        return 'cpu'
    if world_size == 1:
        return 'auto' # Let transformers handle single-GPU placement

    # A dictionary mapping model names to their number of layers.
    num_layers_dict = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80
    }
    if model_name not in num_layers_dict:
        raise ValueError(f"model_name '{model_name}' not found in the layer configuration. "
                         f"Available models: {list(num_layers_dict.keys())}")

    num_layers = num_layers_dict[model_name]
    # Since the first GPU will be used for ViT, treat it as having less capacity for language layers.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    
    # Assign layers to each GPU
    layer_assignments = [num_layers_per_gpu] * world_size
    layer_assignments[0] = math.ceil(layer_assignments[0] * 0.5)
    
    layer_cnt = 0
    for i, num_layer in enumerate(layer_assignments):
        for _ in range(num_layer):
            if layer_cnt < num_layers:
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1

    # Assign remaining components to the first GPU (device 0)
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    
    logging.info(f"Created device map for {world_size} GPUs.")
    return device_map

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    try:
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_file}")
        return None

def parse_scores(score_string: str, num_scores: int) -> List[int]:
    """
    Attempts to extract a specific number of numeric scores from the model's text response.
    """
    if not isinstance(score_string, str) or not score_string.strip():
        return [-1] * num_scores
    s = score_string.strip()
    m = re.search(r'\[([^\]]+)\]', s, flags=re.S)
    if m:
        inner = m.group(1)
        nums = re.findall(r'-?\d+', inner)
        if len(nums) >= num_scores:
            return [int(x) for x in nums[:num_scores]]
    head = s.split('###', 1)[0]
    kv = re.findall(r':\s*(-?\d+)', head)
    if len(kv) >= num_scores:
        return [int(x) for x in kv[:num_scores]]
    all_nums = re.findall(r'-?\d+', head)
    if len(all_nums) >= num_scores:
        return [int(x) for x in all_nums[:num_scores]]
    logging.warning(f"parse_scores fallback failed. Raw head: {head[:200]}")
    return [-1] * num_scores


def main(args):
    """
    Main function to run the evaluation process.
    """
    # --- 1. Load Model and Tokenizer ---
    logging.info(f"Loading model: {args.model_name} from {args.model_path}")
    device_map = split_model(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    logging.info("Model and tokenizer loaded successfully.")

    # --- 2. Read prompt file ---
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as prompt_file:
            my_prompt = prompt_file.read().strip()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {args.prompt_file}. Exiting.")
        return

    # --- 3. Process the input file ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}. Exiting.")
        return

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample)
    processed_data = []
    
    # Open the output file once to write to it periodically
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for i, line in enumerate(tqdm(lines, total=total_lines, desc="Processing jsonl lines")):
            data = json.loads(line)
            question = data.get('question', '')
            answer = data.get('answer', '')
            image_name = data.get('image', '')

            image_path = os.path.join(args.image_dir, image_name) if image_name else None
            image_desc = "<image>" if image_path else "null"
            if not answer and image_path:
                answer = "null"

            full_prompt = f"{my_prompt}\n<chatbegin> Question: {question}; \nAnswer:\n {answer}\n image: {image_desc}\n<chatend>"

            # Generate response
            with torch.no_grad():
                if image_path:
                    pixel_values = load_image(image_path, max_num=args.max_img_num, input_size=args.image_size)
                    if pixel_values is None:
                        output_text = "Error: Could not load image."
                    else:
                        pixel_values = pixel_values.to(torch.bfloat16).cuda()
                        output_text = model.chat(tokenizer, pixel_values, full_prompt, generation_config)
                else:
                    output_text = model.chat(tokenizer, None, full_prompt, generation_config)
            
            scores = parse_scores(output_text, args.num_scores)
            data["labels"] = scores
            logging.info(f"Processed item {i+1}/{total_lines}. Scores: {scores}")

            processed_data.append(data)

            # Save progress at specified intervals or on the last item
            if (i + 1) % args.save_interval == 0 or (i + 1) == total_lines:
                for item in processed_data:
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                out_f.flush() # Ensure data is written to disk
                logging.info(f"Saved {len(processed_data)} items to {args.output_file}")
                processed_data = []

    logging.info("Processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model answers using an InternVL model.")

    # --- File Path Arguments ---
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the InternVL model directory.')
    parser.add_argument('--input_file', type=str, default='model_answer.jsonl',
                        help='Path to the input JSONL file containing model answers.')
    parser.add_argument('--output_file', type=str, default='internvl_scores.jsonl',
                        help='Path to the output JSONL file where scores will be saved.')
    parser.add_argument('--prompt_file', type=str, default='score_prompt.txt',
                        help='Path to the text file containing the evaluation prompt.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Base directory where image files are stored.')

    # --- Model Configuration Arguments ---
    parser.add_argument('--model_name', type=str, default='InternVL2_5-78B',
                        choices=['InternVL2_5-1B', 'InternVL2_5-2B', 'InternVL2_5-4B', 'InternVL2_5-8B',
                                 'InternVL2_5-26B', 'InternVL2_5-38B', 'InternVL2_5-78B'],
                        help='The specific name of the InternVL model to configure device mapping.')
    parser.add_argument('--image_size', type=int, default=448,
                        help='The input size for image patches.')
    parser.add_argument('--max_img_num', type=int, default=12,
                        help='Maximum number of image patches for dynamic preprocessing.')

    # --- Generation & Processing Arguments ---
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of new tokens for the model to generate.')
    parser.add_argument('--do_sample', action='store_true', default=True,
                        help='Whether to use sampling during generation.')
    parser.add_argument('--num_scores', type=int, default=4,
                        help='The number of scores to extract from the response.')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save progress to the output file every N items.')

    args = parser.parse_args()
    main(args)
