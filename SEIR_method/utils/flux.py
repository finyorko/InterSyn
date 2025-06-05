import torch
from diffusers import FluxPipeline
import random
from utils.config_loader import flux_device, flux_model_path

# load flux model
pipe = FluxPipeline.from_pretrained(
    flux_model_path,
    torch_dtype=torch.bfloat16)
pipe.to(flux_device)

# Image generation function, receives prompt and image save path as parameters
def generate_image_by_flux(prompt, image_path):
    images = pipe(
        [prompt],
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=20,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(random.randint(0, 10000))
    ).images
    images[0].save(image_path)
