import os
import json
import argparse
from tqdm import tqdm
from utils.flux import generate_image_by_flux
from utils.internvl import generate_by_internvl
from utils.qwen import generate_by_qw
from utils.get_prompt import *
from utils.config_loader import ques_temp_file, object_file
import random
import logging
import re

# log format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# parser 
parser = argparse.ArgumentParser(description='Process the specified JSON file to generate images, process them, and update the JSON file.')

# Filename without JSON suffix. {json_file_basename}_{dataset_id}.json as the final filename}
parser.add_argument('--json_file_basename', type=str, default="data_test", help='Specify the name of the JSON file to process.')
# The final output directory is `output_parent_dir/dataset_id`.
parser.add_argument('--output_parent_dir', type=str, default="./t2ti_data", help='jsonl and images output parent dir')
parser.add_argument('--dataset_id', type=str, default="00000", help='dataset_id: output child dir')
parser.add_argument('--conversation_turn', type=int, default=1, help='Number of conversation turns for per dataset sample.')
parser.add_argument('--mod_q_suggestion_num', type=int, default=3, help='Number of modifications for question.')
parser.add_argument('--mod_ac_suggestion_num', type=int, default=3, help='Number of modifications for answer.')
parser.add_argument('--mod_c_suggestion_num', type=int, default=3, help='Number of modifications for caption.')
parser.add_argument('--dataset_size', type=int, default=10000, help='Number of dataset sample to generate.')
parser.add_argument('--save_batch_size', type=int, default=1000, help='Number of batch size to save.')

args = parser.parse_args()

output_dir = os.path.join(args.output_parent_dir, args.dataset_id)
output_parent_dir = args.output_parent_dir
dataset_id = args.dataset_id
conversation_turn = args.conversation_turn
mod_q_suggestion_num = args.mod_q_suggestion_num
mod_ac_suggestion_num = args.mod_ac_suggestion_num
mod_c_suggestion_num = args.mod_c_suggestion_num
dataset_size = args.dataset_size

images_dir = os.path.join(output_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

def get_json_list(file_path):
    def extract_deepest(data):
        result = []
        if isinstance(data, list):
            result.extend(data)
        elif isinstance(data, dict):
            for value in data.values():
                result.extend(extract_deepest(value))
        return result

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        return extract_deepest(json_data)
    except FileNotFoundError:
        print(f"can not find '{file_path}' ")
        return []
    except json.JSONDecodeError:
        print(f"Unable to parse JSON data in '{file_path}', please check if the file format is correct.")
        return []


topic_list = get_json_list(object_file)
ques_temp_list = get_json_list(ques_temp_file)

def split_ac(ac):
    try:
        answer_pattern = re.compile(r'answer:\s*(.*?)(?=caption:|$)', re.DOTALL)
        caption_pattern = re.compile(r'caption:\s*(.*)', re.DOTALL)

        # extract answer
        answer_match = answer_pattern.search(ac)
        answer = answer_match.group(1).strip() if answer_match else None

        # extract image caption
        caption_match = caption_pattern.search(ac)
        caption = caption_match.group(1).strip() if caption_match else None
        return answer, caption
    except json.JSONDecodeError as e:
        logging.error(f"ac parse error: {e}")
        return "error_answer", "error_caption"

def generate_final_q(random_topic, history_qac_list, mod_q_suggestion_num, ques_temp):
    gen_q_prompt = get_gen_q_prompt(random_topic, ques_temp, history_qac_list)
    old_q = generate_by_qw(gen_q_prompt)
    logging.info(f"old_q:\n{old_q}\n")
    new_q = None
    for i in range(mod_q_suggestion_num):
        mod_q_suggestion_prompt = get_mod_q_suggestion_prompt(random_topic, old_q, history_qac_list)
        mod_q_suggestion = generate_by_qw(mod_q_suggestion_prompt)
        logging.info(f"{i} mod_q_suggestion:\n{mod_q_suggestion}\n")
        if mod_q_suggestion.strip().lower() == "none":
            break
        mod_q_prompt = get_mod_q_prompt(old_q, mod_q_suggestion)
        new_q = generate_by_qw(mod_q_prompt)
        logging.info(f"{i} new_q:\n{new_q}\n")
        old_q = new_q
    final_q = new_q if new_q is not None else old_q
    return final_q

def generate_final_ac(history_qac_list, final_q, mod_ac_suggestion_num):
    gen_ac_prompt = get_gen_ac_prompt(final_q, history_qac_list)
    old_ac = generate_by_qw(gen_ac_prompt)
    logging.info(f"old_ac:\n{old_ac}\n")
    new_ac = None
    for i in range(mod_ac_suggestion_num):
        mod_ac_suggestion_prompt = get_mod_ac_suggestion_prompt(final_q, old_ac, history_qac_list)
        mod_ac_suggestion = generate_by_qw(mod_ac_suggestion_prompt)
        logging.info(f"{i} mod_ac_suggestion:\n{mod_ac_suggestion}\n")
        if mod_ac_suggestion.strip().lower() == "none":
            break
        mod_ac_prompt = get_mod_ac_prompt(old_ac, mod_ac_suggestion)
        new_ac = generate_by_qw(mod_ac_prompt)
        logging.info(f"{i} new_ac:\n{new_ac}\n")
        old_ac = new_ac
    final_ac = new_ac if new_ac is not None else old_ac
    return final_ac

def generate_final_c(final_q, final_a, old_c, old_img, mod_c_suggestion_num, history_qac_list):
    image_path = old_img
    mod_c_suggestion_prompt = get_mod_c_suggestion_prompt(final_q, final_a, old_c, history_qac_list)
    new_c = None
    for i in range(mod_c_suggestion_num):
        mod_c_suggestion = generate_by_internvl(mod_c_suggestion_prompt, image_path)
        logging.info(f"{i} mod_c_suggestion:\n{mod_c_suggestion}\n")
        if mod_c_suggestion.strip().lower() == "none":
            break
        mod_c_prompt = get_mod_c_prompt(old_c, mod_c_suggestion)
        new_c = generate_by_qw(mod_c_prompt)
        logging.info(f"{i} new_c:\n{new_c}\n")
        generate_image_by_flux(new_c, image_path)
        old_c = new_c
    final_c = new_c if new_c is not None else old_c
    return final_c

def main():
    existing_data = []
    id_num_digits = 10

    # object JSON file path, example: ./t2ti_data/{t2ti_single}_{00000}.json
    output_json_path = os.path.join(output_dir, f"{args.json_file_basename}_{args.dataset_id}.json")
    output_jsonl_path = os.path.join(output_dir, f"{args.json_file_basename}_{args.dataset_id}.jsonl")

    # If the file does not exist, first create an empty array to ensure always a valid JSON array.
    if not os.path.exists(output_json_path):
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        logging.info(f"Created new empty JSON file: {output_json_path}")

    if not os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            pass
        logging.info(f"Created new empty JSONL file: {output_jsonl_path}")

    batch_data = [] 
    for id_index in tqdm(range(dataset_size), desc="Generating IDs"):
        # Generate a unique ID with a length of 10
        unique_id = f"{args.dataset_id}_{str(id_index).zfill(id_num_digits)}"
        logging.info(f"Generating data for ID: {unique_id}")

        # Choose a random topic
        random_topic = random.choice(topic_list)
        # Choose a random question template
        random_ques_temp = random.choice(ques_temp_list)
        logging.info(f"id {id_index} Selected topic: {random_topic}")

        conversations = []
        history_qac_list = []
    
        real_turn = 0
        # randon number for conversation turn
        random_number = random.randint(1, conversation_turn)

        for conversation_index in range(random_number):
            logging.info(f"Starting conversation turn {conversation_index + 1}/{random_number} for ID {unique_id}, index is {conversation_index}")

            # get final_q
            final_q = generate_final_q(random_topic, history_qac_list, mod_q_suggestion_num, random_ques_temp)
            logging.info(f"final_q:\n{final_q}\n")

            # get final_ac
            final_ac = generate_final_ac(history_qac_list, final_q, mod_ac_suggestion_num)
            logging.info(f"final_ac:\n{final_ac}\n")

            # get final_a, old_c
            final_a, old_c = split_ac(final_ac)
            logging.info(f"get final_a by split_ac: \n{final_a}\n")
            logging.info(f"get old_c by split_ac: \n{old_c}\n")

            # generate image
            image_filename = f"{unique_id}_caption_{real_turn}.png"
            image_path = os.path.join(images_dir, image_filename)
            try:
                generate_image_by_flux(old_c, image_path)
                logging.info(f"Image generated and saved to {image_path}")
            except Exception as e:
                logging.error(f"Failed to generate image for caption '{old_c}': {e}")
                image_filename = ""

            # refinement for caption
            final_c = generate_final_c(final_q, final_a, old_c, image_path, mod_c_suggestion_num, history_qac_list)
            logging.info(f"final_c: \n{final_c}")

            conversations.append({
                "from": "human",
                "value": final_q
            })
            conversations.append({
                "from": "gpt",
                "value": final_a,
                "caption": final_c,
                "image": image_filename
            })

            # update conversation history
            history_qac_item = {
                'question': final_q,
                'answer': final_a,
                'caption': final_c,
                'image': image_filename
            }
            history_qac_list.append(history_qac_item)

            real_turn += 1

        data_item = {
            "id": unique_id,
            "topic":random_topic,
            "conversations":conversations,
        }
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(data_item, ensure_ascii=False)
            f.write(json_line + '\n')
        
        batch_data.append(data_item)
        # save every 1000 items
        if len(batch_data) == args.save_batch_size:
            existing_data.extend(batch_data)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
            logging.info(f"{args.save_batch_size} data items have been appended to {output_json_path}")
            # clear batch data
            batch_data = []

    if batch_data:
        existing_data.extend(batch_data)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        logging.info(f"{len(batch_data)} remaining data items have been appended to {output_json_path}")


if __name__ == "__main__":
    main()