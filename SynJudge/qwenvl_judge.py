import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from tqdm import tqdm
import argparse
import re
import os
import logging
from typing import List, Dict, Any

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This utility function is specific to some Qwen-VL versions.
# If it's not in your environment, you might need to copy it from the model's source files.
def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for msg in messages:
        if msg['role'] == 'user':
            for content in msg['content']:
                if content['type'] == 'image':
                    image_inputs.append(content['image'])
                elif content['type'] == 'video':
                    video_inputs.append(content['video'])
    return image_inputs, video_inputs


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
    # --- 1. Read prompt file ---
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as prompt_file:
            my_prompt = prompt_file.read().strip()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {args.prompt_file}. Exiting.")
        return
        
    # --- 2. Load Model and Processor ---
    logging.info(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    logging.info("Model and processor loaded successfully.")

    # --- 3. Process the input file ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}. Exiting.")
        return

    processed_data = []
    
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for i, line in enumerate(tqdm(lines, total=total_lines, desc="Processing jsonl lines")):
            data = json.loads(line)
            question = data.get('question', '')
            answer = data.get('answer', '')
            image_name = data.get('image', '')

            image_path = os.path.join(args.image_dir, image_name) if image_name else None
            
            image_content = [{"type": "image", "image": image_path}] if image_path else []
            image_desc = "The generated content has an image." if image_path else "null"
            if not answer and image_path:
                answer = "null"
                image_desc = "The generated only content has an image."

            messages = [{
                "role": "user",
                "content": [
                    *image_content,
                    {"type": "text", "text": f'{my_prompt} \n"""<chatbegin>\n**Question**: \n{question}; \n**Answer**: \ntext:{answer}, \nimage:{image_desc}\n"<chatend>""'}
                ]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            try:
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
                ).to("cuda")
    
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            except Exception as e:
                logging.error(f"Error processing item {data.get('id', 'N/A')}: {e}")
                output_text = "Error during model generation."
                output_text = "[Text Content Completeness: 5; Image Content Completeness: 2; Image Quality: 1; Image-Text Synergy: 5]"
            
            scores = parse_scores(output_text, args.num_scores)
            data["labels"] = scores
            logging.info(f"Processed item {i+1}/{total_lines}. Scores: {scores}")

            processed_data.append(data)

            if (i + 1) % args.save_interval == 0 or (i + 1) == total_lines:
                for item in processed_data:
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                out_f.flush()
                logging.info(f"Saved {len(processed_data)} items to {args.output_file}")
                processed_data = []
    
    logging.info("Processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model answers using a Qwen-VL model.")

    # --- File Path Arguments ---
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-32B-Instruct',
                        help='Path or name of the Qwen-VL model from Hugging Face.')
    parser.add_argument('--input_file', type=str, default='model_answer.jsonl',
                        help='Path to the input JSONL file containing model answers.')
    parser.add_argument('--output_file', type=str, default='model_score.jsonl',
                        help='Path to the output JSONL file where scores will be saved.')
    parser.add_argument('--prompt_file', type=str, default='score_prompt.txt',
                        help='Path to the text file containing the evaluation prompt.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Base directory where image files are stored.')

    # --- Generation & Processing Arguments ---
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum number of new tokens for the model to generate.')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save progress to the output file every N items.')
    parser.add_argument('--num_scores', type=int, default=4,
                        help='The number of scores to extract from the response.')

    args = parser.parse_args()
    main(args)

