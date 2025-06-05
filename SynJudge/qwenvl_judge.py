import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
from tqdm import tqdm
import argparse
import re
import os

# read score prompt
prompt_file_path = 'score_prompt.txt'
try:
    with open(prompt_file_path, 'r', encoding='utf-8') as prompt_file:
        my_prompt = prompt_file.read().strip()
except FileNotFoundError:
    print(f"can not find prompt file: {prompt_file_path}")
    my_prompt = ""

parser = argparse.ArgumentParser(description='parser jsonl file')

parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-32B-Instruct',
                    help='input judge model path')
parser.add_argument('--jsonl_path', type=str, default='model_answer.jsonl',
                    help='input model answer jsonl, default is model_answer.jsonl')
parser.add_argument('--output_jsonl_path', type=str, default='model_score.jsonl',
                    help='output score jsonl path, defalut is model_score.jsonl')
parser.add_argument('--image_dir', type=str, default='images',
                    help='model answer images path, defalut is images')

args = parser.parse_args()


model_path = args.model_path
jsonl_path = args.jsonl_path
output_jsonl_path = args.output_jsonl_path
image_dir = args.image_dir

print(f"model_path: {os.path.basename(model_path)}")
print(f"jsonl_path: {jsonl_path}")
print(f"output_jsonl_path: {output_jsonl_path}")
print(f"image_dir: {os.path.basename(image_dir)}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

rocessor = AutoProcessor.from_pretrained(model_path)

with open(jsonl_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

processed_data = []
counter = 0

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_lines, desc="Processing jsonl lines"):
        data = json.loads(line)

        id = data.get('id', '')
        question = data.get('question', '')
        answer = data.get('answer', '')
        image_path = data.get('image', '')

        if image_path:
            image_path = f"{image_dir}/{image_path}"
            image_content = [{"type": "image", "image": image_path}]
            image="The generated content has an image."
        else:
            image_content = []
            image = "null"

        if answer == "" and image != "null":
            answer = "null"
            image="The generated only content has an image."

        messages = [
            {
                "role": "user",
                "content": [
                    *image_content,
                    {"type": "text", "text": f'{my_prompt} \n"""<chatbegin>\n**Question**: \n{question}; \n**Answer**: \ntext:{answer}, \nimage:{image}\n"<chatend>""'}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print("\n")
        print(output_text)

        def extract_scores(output_text):
            output_text = output_text.replace('null', '0')
            pattern = r'(?:Text Response Quality|Image Response Quality|Image Aesthetic Quality|Text-Image Consistency): (\d)'
            matches = re.findall(pattern, output_text)
            if len(matches) == 4:
                scores = [int(score) for score in matches]
            else:
                scores = [-1, -1, -1, -1]
            return scores

        try:
            scores = extract_scores(output_text)
        except (IndexError, ValueError):
            print(f"ID {id} score failed, write labels: [-1, -1, -1, -1]")
            scores = [-1, -1, -1, -1]

        data["labels"] = scores
        processed_data.append(data)
        counter += 1

        if counter % 10 == 0:
            with open(output_jsonl_path, 'a', encoding='utf-8') as out_f:
                for item in processed_data:
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            processed_data = []

    if processed_data:
        with open(output_jsonl_path, 'a', encoding='utf-8') as out_f:
            for item in processed_data:
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
    