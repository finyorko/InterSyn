import json
import torch
from modelscope import snapshot_download
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from modelscope import AutoModelForCausalLM, AutoTokenizer as AutoTokenizer_QW
import math
import re

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

path_intern="<internvl_model_path>"
# choose ckpt
device_map = split_model("InternVL2_5-78B")

model = AutoModel.from_pretrained(
    path_intern,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map
).eval()

tokenizer = AutoTokenizer.from_pretrained(path_intern, trust_remote_code=True, use_fast=False)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
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
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
    return pixel_values


def generate_by_internvl(prompt, img_path=None):
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    if img_path:
        pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
        
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    else:
        response = model.chat(tokenizer, None, prompt, generation_config)
    return response


prompt_file_path = 'score_prompt.txt'
try:
    with open(prompt_file_path, 'r', encoding='utf-8') as prompt_file:
        my_prompt = prompt_file.read().strip()
except FileNotFoundError:
    print(f"can not find prompt file: {prompt_file_path}")
    my_prompt = ""


jsonl_path = 'judge_data.jsonl'
output_jsonl_path = 'internvl_judge_trained_10_epoch.jsonl'

with open(jsonl_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

processed_data = []
counter = 0
image_dir = "<your_image_dir>"

with open(jsonl_path, 'r', encoding='utf-8') as f, open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
    for line in tqdm(f, total=total_lines, desc="Processing jsonl lines"):
        data = json.loads(line)
        question = data.get('question', '')
        answer = data.get('answer', '')
        image_path = data.get('image', '')
        if image_path:
            image_path = f"{image_dir}/{image_path}"
            image="<image>"
        else:
            image_path = None
            image="null"

        if answer == "":
            answer = "null"

        new_prompt = f"{my_prompt}\n<chatbegin> Question: {question}; \nAnswer:\n {answer}\n image: {image}\n<chatend>"
        output_text = generate_by_internvl(new_prompt, img_path=image_path)

        
        def extract_scores(output_text):
            output_text = output_text.replace('null', '0')
            pattern = r'(?:Text Response Quality|Image Response Quality|Image Aesthetic Quality|Text-Image Consistency): (\d)'
            matches = re.findall(pattern, output_text)
            if len(matches) == 4:
                scores = [int(score) for score in matches]
            else:
                scores = [-1, -1, -1, -1]
            return scores
            
        scores = extract_scores(output_text)
        data["labels"] = scores

        processed_data.append(data)
        counter += 1

        if counter % 10 == 0:
            with open(output_jsonl_path, 'a', encoding='utf-8') as out_f:
                for item in processed_data:
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            processed_data = []