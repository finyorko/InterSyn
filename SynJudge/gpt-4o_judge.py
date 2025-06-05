import json
import requests
import time
from tqdm import tqdm
import base64
import http.client
import os

# read prompt file
prompt_file_path = 'score_prompt.txt'
try:
    with open(prompt_file_path, 'r', encoding='utf-8') as prompt_file:
        my_prompt = prompt_file.read().strip()
except FileNotFoundError:
    print(f"can not find prompt file: {prompt_file_path}")
    my_prompt = ""

# api config
key ='<YOUR_API_KEY>'
host = "<BASE_HOST>"
url_suffix = "/v1/chat/completions"

# print(host)
conn = http.client.HTTPSConnection(host)

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {key}'
}

# read jsonl file
jsonl_path = 'model_answer.jsonl'
output_jsonl_path = 'gpt_judge.jsonl'

# count jsonl line number
with open(jsonl_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

def parse_score_string(output_text):
    """
    Attempt to extract 4 numeric segments from the text returned by GPT-4o.
    if success return [int, int, int, int], else return [-1, -1, -1, -1]。
    """
    try:
        output_text = output_text.strip('[]')
        parts = output_text.split(';')
        scores = []
        for part in parts:
            score = int(part.split(':')[1].strip())
            scores.append(score)
        if len(scores) != 4:
            raise ValueError("score failed")
        return scores
    except (IndexError, ValueError):
        return None

def safe_get_gpt_scores(prompt, image_path, max_retries=5):
    for attempt in range(1, max_retries + 1):
        output_text = get_gpt4answer(prompt, image_path)
        if output_text and "extract failed" not in output_text:
            scores = parse_score_string(output_text)
            if scores:
                return scores
        time.sleep(1 + attempt)
    print("❌ After multiple attempts, still failed, marked as [-1, -1, -1, -1]")
    return [-1, -1, -1, -1]

def get_gpt4answer(question, image_path):
    # image to base64
    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Failed to load image: {str(e)}")
            return None
    content = []
    content.append({"type": "text", "text": question})
    if image_path != "" and image_path is not None:
        base64_image = encode_image(image_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}
        ],
        "max_tokens": 300
    })
    conn.request("POST", url_suffix, payload, headers)
    res = conn.getresponse()
    data = res.read()
    try:
        response_json = json.loads(data.decode("utf-8"))
        chatgpt_response = response_json["choices"][0]["message"]["content"]
    except:
        chatgpt_response = "extract failed"
    return chatgpt_response

processed_data = []
counter = 0
image_dir = "<your_image_dir>"

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_lines, desc="Processing jsonl lines"):
        data = json.loads(line)
        question = data.get('question', '')
        answer = data.get('answer', '')
        image_path = data.get('image', '')

        if image_path:
            image_path = f"{image_dir}/{image_path}"
            image = "The generated content has an image."
        else:
            image = "null"

        if answer == "" and image != "null":
            answer = "null"
            image = "The generated only content has an image."

        input_text = f'{my_prompt} \n"""<chatbegin>\n**Question**: \n{question}; \n**Answer**: \ntext:{answer}, \nimage:{image}\n"<chatend>""'

        scores = safe_get_gpt_scores(input_text, image_path)
        print(scores)
        data["labels"] = scores

        processed_data.append(data)
        counter += 1

        # save every 5 items
        if counter % 5 == 0:
            with open(output_jsonl_path, 'a', encoding='utf-8') as out_f:
                for item in processed_data:
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            processed_data = []

    if processed_data:
        with open(output_jsonl_path, 'a', encoding='utf-8') as out_f:
            for item in processed_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
