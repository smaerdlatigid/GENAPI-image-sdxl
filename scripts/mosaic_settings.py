#!/usr/bin/env python3

import os
import glob
import requests
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import matplotlib.pyplot as plt

samplers = [
    'euler',
    'dpm_fast',
    'dpmpp_sde',
    'dpmpp_sde_gpu',
    'ddpm',
    'ddim',
    'deis',
    'uni_pc',
    'lcm'
]

schedulers = [
    'normal',
    'karras',
    'exponential',
    'sgm_uniform',
    'simple',
    'ddim_uniform',
    'beta'
]

# Seed for the model
SEED = np.random.randint(2**20, 2**22)

# Directory to save the generated images
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

for sampler in tqdm(samplers, desc="Generating images"):
    for scheduler in schedulers:
        data = {
            "input": {
                "prompt": "Studio Ghibli-inspired scenic picture featuring gentle hills and mountains in the distance, serene forests with golden light, traditional architecture like shrines or temples, lanterns floating on ponds, magical foxes playing in fields, and sweeping landscapes. The sky should be painted with vivid colors including orange, pink, purple against clear blue skies",
                "suffix_prompt": "equirectangular, 360 panorama",
                "negative_prompt": "boring, text, signature, watermark, low quality, bad quality, grainy, blurry",
                "width": 2048,
                "height": 1024,
                "seed": SEED,
                "lora_strength": 1.0,
                "cfg": 3.20,
                "steps": 10,
                "sampler": sampler,
                "scheduler": scheduler,
                "tile_x": True,
                "tile_y": False,
                # "upscale_by": 0.0,
                # "upscale_steps": 10,
                # "upscale_sampler": "dpmpp_sde_gpu",
                # "upscale_scheduler": "karras",
                # "upscale_denoise": 0.3,
                # "upscale_seed": -1,
                "output_format": "webp"
            }
        }

        response = requests.post(
            f"http://localhost:{os.environ.get('INTERNAL_PORT',5000)}/predictions",
            headers={"Content-Type": "application/json"},
            json=data
        ).json()

        depth_url, image_url, _ = response['output']
        predict_time = response['metrics']['predict_time']
        image_response = requests.get(image_url)
        
        # Format the predict_time as a string in the filename
        output_filename = f"{sampler}_{scheduler}_{predict_time:.2f}s.webp"
        
        with open(os.path.join(output_dir, output_filename), 'wb') as f:
            f.write(image_response.content)

# Load the images into a list
images = []
for sampler in samplers:
    for scheduler in schedulers:
        img_path = os.path.join(output_dir, f"{sampler}_{scheduler}_*.webp")
        matching_files = glob.glob(img_path)
        if matching_files:
            img = Image.open(matching_files[0])
            images.append(img)


# Create a grid of images
fig, axs = plt.subplots(len(samplers), len(schedulers), figsize=(20, 20))
axs = axs.flatten()

for ax, img in zip(axs, images):
    ax.imshow(img, aspect='auto', interpolation='none', origin='upper')
    # Extract basename without extension and prediction timestamp from filename
    basename_without_extension = os.path.splitext(os.path.basename(img.filename))[0]
    ax.set_title(basename_without_extension, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mosaic.png'), dpi=420)
plt.close()


# Create a grid of images zoomed in
fig, axs = plt.subplots(len(samplers), len(schedulers), figsize=(20, 20))
axs = axs.flatten()

for ax, img in zip(axs, images):
    ax.imshow(img, aspect='auto', interpolation='none', origin='upper')
    # middle of image
    x = img.width // 2 - 300
    y = img.height // 2 
    # zoom in
    zoom = 0.25
    ax.set_xlim(x - img.width * zoom / 2, x + img.width * zoom / 2)
    ax.set_ylim(y + img.height * zoom / 2, y - img.height * zoom / 2)

    # Extract basename without extension and prediction timestamp from filename
    basename_without_extension = os.path.splitext(os.path.basename(img.filename))[0]
    ax.set_title(basename_without_extension, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mosaic_zoomed.png'), dpi=420)
plt.close()