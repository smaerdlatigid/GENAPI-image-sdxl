import os
import json
from typing import List
from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from cog import BasePredictor, Input
from comfyui import ComfyUI

from minio_manager import MinioStorageManager as CloudStorageManager

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

WORKFLOWS = {
    'base': os.environ.get('WORKFLOW_IMAGE', 'workflows/360-panorama-sdxl-depth.json'),
    'upscale': os.environ.get('WORKFLOW_IMAGE_UPSCALE', 'workflows/360-panorama-sdxl-upscale.json')
}

BUCKETS = {
    'base': os.environ.get('BUCKET_IMAGE', '360-panorama-sdxl'),
    'upscale': os.environ.get('BUCKET_IMAGE_UPSCALE', '360-panorama-sdxl-upscale')
}



# def optimise_image_files(
#     output_format: str = DEFAULT_FORMAT, output_quality: int = DEFAULT_QUALITY, files=[]
# ):
#     if should_optimise_images(output_format, output_quality):
#         optimised_files = []
#         for file in files:
#             if file.is_file() and file.suffix in IMAGE_FILE_EXTENSIONS:
#                 image = Image.open(file)
#                 optimised_file_path = file.with_suffix(f".{output_format}")
#                 image.save(
#                     optimised_file_path,
#                     quality=output_quality,
#                     optimize=True,
#                 )
#                 optimised_files.append(optimised_file_path)
#             else:
#                 optimised_files.append(file)

#         return optimised_files
#     else:
#         return files

class Predictor(BasePredictor):

    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        self.cloud = CloudStorageManager(
            endpoint=os.environ['MINIO_ENDPOINT'],
            access_key=os.environ['MINIO_ACCESS_KEY'],
            secret_key=os.environ['MINIO_SECRET_KEY'],
            external_endpoint=os.environ['MINIO_EXTERNAL_ENDPOINT']
        )

        # TODO add a way to test the connection to the cloud storage by listing buckets

    def predict(
        self,
        prompt: str = Input(
            description="Describe the image",
            default="Magical pizza kingdom"            
        ),
        suffix_prompt: str = Input(
            description="Suffix prompt for image generation (e.g., '360 panorama, equirectangular')",
            default="equirectangular, 360 panorama"
        ),
        negative_prompt: str = Input(
            description="Negative prompt for image generation (e.g., 'boring, text, signature, watermark, low quality, bad quality, grainy, blurry, long neck, closed eyes')",
            default="boring, text, signature, watermark, low quality, bad quality, grainy, blurry"
        ),
        width: int = Input(
            description="Width of the image",
            default=2048
        ),
        height: int = Input(
            description="Height of the image",
            default=1024
        ),
        seed: int = Input(
            description="Seed for the model, if negative will be random",
            default=-1
        ),
        lora_strength: float = Input(
            description="Strength of the LoRA model",
            default=1.0
        ),
        cfg: float = Input(
            description="CFG value for the model",
            default=3.50
        ),
        steps: int = Input(
            description="Number of steps for the model",
            default=12
        ),
        sampler: str = Input( 
            description="Sampler for the model",
            default="dpmpp_sde"
        ),
        scheduler: str = Input(
            description="Scheduler for the model",
            default="ddim_uniform"
        ),
        tile_x: bool = Input(
            description="Tile the image in the x direction",
            default=True
        ),
        tile_y: bool = Input(
            description="Tile the image in the y direction",
            default=False
        ),
        upscale_by: float = Input(
            description="Upscale the image by this factor",
            default=0.0
        ),
        upscale_steps: int = Input(
            description="Number of steps for upscaling",
            default=10
        ),
        upscale_sampler: str = Input(
            description="Upscale sampler",
            default="uni_pc"
        ),
        upscale_scheduler: str = Input(
            description="Upscale scheduler",
            default="beta"
        ),
        upscale_denoise: float = Input(
            description="Denoise the image",
            default=0.3
        ),
        upscale_seed: int = Input(
            description="Seed for upscaling",
            default=-1
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        )

    ) -> List[str]: # for local deployment

        # clean up directories
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # load the workflow
        if upscale_by > 1:
            self.cloud.bucket = BUCKETS['upscale']
            with open(WORKFLOWS['upscale'], "r") as file:
                EXAMPLE_WORKFLOW_JSON = json.loads(file.read())
        else:
            self.cloud.bucket = BUCKETS['base']
            with open(WORKFLOWS['base'], "r") as file:
                EXAMPLE_WORKFLOW_JSON = json.loads(file.read())

        wf = self.comfyUI.load_workflow(EXAMPLE_WORKFLOW_JSON)

        # connect to comfyUI
        self.comfyUI.connect()

        # update the workflow with the inputs
        wf['5']['inputs']['width'] = width
        wf['5']['inputs']['height'] = height

        # build the prompts
        wf['6']['inputs']['text'] = prompt
        wf['6']['inputs']['text'] += ", " + suffix_prompt
        wf['7']['inputs']['text'] = negative_prompt

        # lora config
        wf['10']['inputs']['strength_model'] = lora_strength

        # seed
        self.comfyUI.randomise_seeds(wf)

        if seed > 0:
            wf['11']['inputs']['seed'] = seed

        # sampler
        wf['11']['inputs']['cfg'] = cfg
        wf['11']['inputs']['steps'] = steps
        wf['11']['inputs']['tileX'] = int(tile_x)
        wf['11']['inputs']['tileY'] = int(tile_y)
        wf['11']['inputs']['sampler_name'] = sampler
        wf['11']['inputs']['scheduler'] = scheduler

        # upscaling
        if upscale_by > 1:
            wf['38']['inputs']['upscale_by'] = upscale_by
            wf['38']['inputs']['upscale_steps'] = upscale_steps
            wf['38']['inputs']['sampler_name'] = upscale_sampler
            wf['38']['inputs']['scheduler'] = upscale_scheduler
            wf['38']['inputs']['denoise'] = upscale_denoise
            
            if upscale_seed > 0:
                wf['38']['inputs']['seed'] = upscale_seed

        # run the workflow
        self.comfyUI.run_workflow(wf)

        output_directories = [OUTPUT_DIR]

        images = self.comfyUI.get_files(output_directories)

        # convert images to webp
        saved_images = []
        for img in images:
            image = Image.open(img)
            optimised_file_path = img.with_suffix(f".{output_format}")
            image.save(
                optimised_file_path,
                quality=100,
                optimize=True,
            )
            saved_images.append(optimised_file_path)

        # back up images on cloud storage
        image_hash = self.cloud.hash_file(saved_images[1])

        # upload image
        try:
            image_url = self.cloud.upload_file(saved_images[1], f"{image_hash}/image.webp", 'image/webp')
            print(f"Image uploaded to: {image_url}")
        except Exception as e:
            print(f"Failed to upload image: {e}")

            if 'duplicate' in str(e).lower():
                image_url = self.cloud.get_file_url(f"{image_hash}/image.webp")
            else:
                image_url = None

        #upload depth
        try:
            depth_url = self.cloud.upload_file(saved_images[0], f"{image_hash}/depth.webp", 'image/webp')
            print(f"Depth uploaded to: {depth_url}")
        except Exception as e:
            print(f"Failed to upload depth image: {e}")

            if 'duplicate' in str(e).lower():
                depth_url = self.cloud.get_file_url(f"{image_hash}/depth.webp")
            else:
                depth_url = None

        # create thumbnails
        image_thumbnail = self.cloud.resize_image(saved_images[1])
    
        # Save the resized image
        Image.fromarray(image_thumbnail).save('image_thumbnail.webp', format='WEBP')

        try:
            thumbnail_url = self.cloud.upload_file('image_thumbnail.webp', f"{image_hash}/image_thumbnail.webp", 'image/webp')
            print(f"Thumbnail uploaded to: {thumbnail_url}")
        except Exception as e:
            print(f"Failed to upload thumbnail: {e}")

            if 'duplicate' in str(e).lower():
                thumbnail_url = self.cloud.get_file_url(f"{image_hash}/image_thumbnail.webp")
            else:
                thumbnail_url = None

        # depth thumbnail
        depth_thumbnail = self.cloud.resize_image(saved_images[0])

        # Save the resized image
        Image.fromarray(depth_thumbnail).save('depth_thumbnail.webp', format='WEBP')

        try:
            depth_thumbnail_url = self.cloud.upload_file('depth_thumbnail.webp', f"{image_hash}/depth_thumbnail.webp", 'image/webp')
            print(f"Depth thumbnail uploaded to: {depth_thumbnail_url}")
        except Exception as e:
            print(f"Failed to upload depth thumbnail: {e}")

            if 'duplicate' in str(e).lower():
                depth_thumbnail_url = self.cloud.get_file_url(f"{image_hash}/depth_thumbnail.webp")
            else:
                depth_thumbnail_url = None

        # save to same directory as saved_images[0]
        with open(f"{saved_images[0].parent}/workflow.json", "w") as file:
            file.write(json.dumps(wf, indent=4))

        try:
            workflow_url = self.cloud.upload_file(f"{saved_images[0].parent}/workflow.json", f"{image_hash}/workflow.json", 'application/json')
            print(f"Workflow uploaded to: {workflow_url}")
        except Exception as e:
            print(f"Failed to upload workflow: {e}")

            if 'duplicate' in str(e).lower():
                workflow_url = self.cloud.get_file_url(f"{image_hash}/workflow.json")
            else:
                workflow_url = None

        # create a metadata json
        metadata = {
            "id": image_hash,
            "prompt": prompt,
            "suffix_prompt": suffix_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": wf['11']['inputs']['seed'],
            "lora_strength": lora_strength,
            "cfg": cfg,
            "steps": steps,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "sampler": sampler,
            "scheduler": scheduler,
            "upscale_by": upscale_by,
            "upscale_steps": upscale_steps,
            "upscale_sampler": upscale_sampler,
            "upscale_scheduler": upscale_scheduler,
            "upscale_denoise": upscale_denoise,
            "upscale_seed": wf.get('38', {}).get('inputs', {}).get('seed', -1),
            "output_format": output_format,
            "image_url": image_url,
            "depth_url": depth_url,
            "thumbnail_url": thumbnail_url,
            "depth_thumbnail_url": depth_thumbnail_url,
            "workflow_url": workflow_url
        }

        with open(f"{saved_images[0].parent}/metadata.json", "w") as file:
            file.write(json.dumps(metadata, indent=4))

        try:
            metadata_url = self.cloud.upload_file(f"{saved_images[0].parent}/metadata.json", f"{image_hash}/metadata.json", 'application/json')
            print(f"Metadata uploaded to: {metadata_url}")
        except Exception as e:
            print(f"Failed to upload metadata: {e}")

            if 'duplicate' in str(e).lower():
                metadata_url = self.cloud.get_file_url(f"{image_hash}/metadata.json")
            else:
                metadata_url = None

        return [depth_url, image_url, metadata_url]