import json
import time
import base64
import http.client
import os
import re
import argparse
import logging
from typing import List
from tqdm import tqdm

# --- Setup logging ---
# This ensures that warning messages from parse_scores are displayed.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_scores(score_string: str, num_scores: int) -> List[int]:
    """
    Attempts to extract a specific number of numeric scores from the model's text response.

    Args:
        score_string: The raw text output from the language model.
        num_scores: The expected number of scores to extract.

    Returns:
        A list of integers if successful, otherwise a list of -1s.
        e.g., [1, 2, 3, 4] or [-1, -1, -1, -1]
    """
    if not isinstance(score_string, str) or not score_string.strip():
        return [-1] * num_scores
    s = score_string.strip()

    # Strategy 1: Look for scores in a bracket, e.g., "[1, 0, 1, 1]"
    m = re.search(r'\[([^\]]+)\]', s, flags=re.S)
    if m:
        inner = m.group(1)
        nums = re.findall(r'-?\d+', inner)
        if len(nums) >= num_scores:
            return [int(x) for x in nums[:num_scores]]

    # Strategy 2: Look for numbers following a colon, e.g., "Score 1: 5"
    head = s.split('###', 1)[0]
    kv = re.findall(r':\s*(-?\d+)', head)
    if len(kv) >= num_scores:
        return [int(x) for x in kv[:num_scores]]

    # Strategy 3: Find all numbers in the first part of the response
    all_nums = re.findall(r'-?\d+', head)
    if len(all_nums) >= num_scores:
        return [int(x) for x in all_nums[:num_scores]]

    logging.warning(f"parse_scores fallback failed. Raw head: {head[:200]}")
    return [-1] * num_scores


def get_gpt4_answer(question: str, image_path: str, conn: http.client.HTTPConnection, headers: dict, args: argparse.Namespace) -> str:
    """
    Sends a request to the GPT-4o API with text and an optional image.

    Args:
        question: The prompt/question text.
        image_path: The local path to an image file.
        conn: The HTTP connection object.
        headers: The request headers.
        args: The command-line arguments object.

    Returns:
        The content of the model's response as a string, or "extract failed" on error.
    """
    def encode_image(img_path):
        try:
            with open(img_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to load or encode image {img_path}: {str(e)}")
            return None

    content = [{"type": "text", "text": question}]
    if image_path:
        base64_image = encode_image(image_path)
        if base64_image:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    payload = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens
    })

    try:
        conn.request("POST", args.api_endpoint, payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_json = json.loads(data.decode("utf-8"))
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            logging.error(f"API response did not contain valid choices: {response_json}")
            return "extract failed"
            
    except Exception as e:
        logging.error(f"API request failed: {e}")
        # Re-establish connection for the next attempt
        conn.close()
        conn.connect()
        return "extract failed"


def safe_get_gpt_scores(prompt: str, image_path: str, conn: http.client.HTTPConnection, headers: dict, args: argparse.Namespace) -> List[int]:
    """
    A wrapper for get_gpt4_answer that includes retry logic and score parsing.
    """
    for attempt in range(1, args.max_retries + 1):
        output_text = get_gpt4_answer(prompt, image_path, conn, headers, args)
        if output_text and "extract failed" not in output_text:
            scores = parse_scores(output_text, args.num_scores)
            if scores[0] != -1:  # Check if parsing was successful
                return scores
        logging.warning(f"Attempt {attempt}/{args.max_retries} failed. Retrying in {1 + attempt} seconds...")
        time.sleep(1 + attempt)

    logging.error(f"Failed after {args.max_retries} attempts. Marking as [-1, ...]")
    return [-1] * args.num_scores


def main(args):
    """
    Main function to run the scoring process.
    """
    # --- 1. Read prompt file ---
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as prompt_file:
            my_prompt = prompt_file.read().strip()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {args.prompt_file}. Exiting.")
        return

    # --- 2. Setup API connection ---
    conn = http.client.HTTPConnection(args.api_host)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {args.api_key}'
    }

    # --- 3. Process the input file ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}. Exiting.")
        return

    processed_data = []
    for i, line in enumerate(tqdm(lines, total=total_lines, desc="Processing jsonl lines")):
        data = json.loads(line)
        question = data.get('question', '')
        answer = data.get('answer', '')
        image_name = data.get('image', '')
        
        image_path = os.path.join(args.image_dir, image_name) if image_name else ""
        
        image_desc = "The generated content has an image." if image_path else "null"
        if not answer and image_path:
            answer = "null"
            image_desc = "The generated content only has an image."

        input_text = f'{my_prompt} \n"""<chatbegin>\n**Question**: \n{question}; \n**Answer**: \ntext:{answer}, \nimage:{image_desc}\n"<chatend>""'

        scores = safe_get_gpt_scores(input_text, image_path, conn, headers, args)
        data["labels"] = scores
        logging.info(f"Processed item {i+1}/{total_lines}. Scores: {scores}")

        processed_data.append(data)

        # Save progress at specified intervals or on the last item
        if (i + 1) % args.save_interval == 0 or (i + 1) == total_lines:
            with open(args.output_file, 'a', encoding='utf-8') as out_f:
                for item in processed_data:
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logging.info(f"Saved {len(processed_data)} items to {args.output_file}")
            processed_data = []
            
    conn.close()
    logging.info("Processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model answers using a remote GPT-4o API.")

    # --- File Path Arguments ---
    parser.add_argument('--input_file', type=str, default='model_answer.jsonl',
                        help='Path to the input JSONL file containing model answers.')
    parser.add_argument('--output_file', type=str, default='model_score.jsonl',
                        help='Path to the output JSONL file where scores will be saved.')
    parser.add_argument('--prompt_file', type=str, default='score_prompt.txt',
                        help='Path to the text file containing the evaluation prompt.')
    parser.add_argument('--image_dir', type=str, default='/path/to/your/images',
                        help='Base directory where image files are stored.')

    # --- API Configuration Arguments ---
    parser.add_argument('--api_key', type=str, default='sk-your-key-here',
                        help='API key for the service.')
    parser.add_argument('--api_host', type=str, default='127.0.0.1:8000',
                        help='Host and port for the API endpoint (e.g., "host:port").')
    parser.add_argument('--api_endpoint', type=str, default='/v1/chat/completions',
                        help='URL suffix for the chat completions endpoint.')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='The name of the model to use for evaluation.')

    # --- Processing Control Arguments ---
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Maximum number of retries for a failed API call.')
    parser.add_argument('--num_scores', type=int, default=4,
                        help='The number of scores to extract from the response.')
    parser.add_argument('--max_tokens', type=int, default=300,
                        help='Maximum number of tokens to generate in the response.')
    parser.add_argument('--save_interval', type=int, default=3,
                        help='Save progress to the output file every N items.')

    args = parser.parse_args()
    main(args)
