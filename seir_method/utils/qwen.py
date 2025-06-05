import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from modelscope import AutoModelForCausalLM, AutoTokenizer as AutoTokenizer_QW
from decord import VideoReader, cpu
from utils.config_loader import qwen_model_path, qwen_device


# ------------------------- load model -------------------------
model_qw = AutoModelForCausalLM.from_pretrained(
    qwen_model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer_qw = AutoTokenizer_QW.from_pretrained(qwen_model_path)


def generate_by_qw(prompt):
    messages = [
        {"role": "system", "content":  "You are a data generation assistant and you are only allowed to output in English! you are only allowed to output in English!"},
        {"role": "user", "content": prompt + "\nYou are only allowed to output in English!"}
    ]
    text = tokenizer_qw.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer_qw([text], return_tensors="pt").to(qwen_device)

    generated_ids = model_qw.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    respon_qw = tokenizer_qw.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return respon_qw
