import os
import json
import shutil
import tarfile
import zipfile
from typing import List

# uncomment to run cog predict
from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI

from minio_manager import MinioStorageManager as CloudStorageManager

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

WORKFLOWS = {
    'base': os.environ.get('WORKFLOW_IMAGE', 'workflows/360-panorama-sdxl-depth.json'),
    'upscale': os.environ.get('WORKFLOW_IMAGE_UPSCALE', 'workflows/360-panorama-sdxl-upscale-inpaint-depth.json'), # from prompt
    'upscale-input': os.environ.get('WORKFLOW_IMAGE_UPSCALE_INPUT', 'workflows/360-panorama-sdxl-input-upscale-inpaint-depth.json') # from id
}

BUCKETS = {
    'base': os.environ.get('BUCKET_IMAGE', '360-panorama-sdxl'),
    'upscale': os.environ.get('BUCKET_IMAGE_UPSCALE', '360-panorama-sdxl-upscale')
}


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

        try:
            self.cloud.list_buckets()
        except Exception as e:
            raise(f"Failed to connect to Minio: {e}\ntry adjusting environment variables")


    def handle_input_file(self, input_file: Path):
        file_extension = self.get_file_extension(input_file)

        if file_extension == ".tar":
            with tarfile.open(input_file, "r") as tar:
                tar.extractall(INPUT_DIR)
        elif file_extension == ".zip":
            with zipfile.ZipFile(input_file, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)
        elif file_extension in [".jpg", ".jpeg", ".png", ".webp"]:
            shutil.copy(input_file, os.path.join(INPUT_DIR, f"input{file_extension}"))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print("====================================")
        print(f"Inputs uploaded to {INPUT_DIR}:")
        self.comfyUI.get_files(INPUT_DIR)
        print("====================================")

    def get_file_extension(self, input_file: Path) -> str:
        file_extension = os.path.splitext(input_file)[1].lower()
        if not file_extension:
            with open(input_file, "rb") as f:
                file_signature = f.read(4)
            if file_signature.startswith(b"\x1f\x8b"):  # gzip signature
                file_extension = ".tar"
            elif file_signature.startswith(b"PK"):  # zip signature
                file_extension = ".zip"
            else:
                try:
                    with Image.open(input_file) as img:
                        file_extension = f".{img.format.lower()}"
                        print(f"Determined file type: {file_extension}")
                except Exception as e:
                    raise ValueError(
                        f"Unable to determine file type for: {input_file}, {e}"
                    )
        return file_extension

    def predict(
        self,
        input_file: Path = Input(
            description="Input image, tar or zip file. Read guidance on workflows and input files here: https://github.com/fofr/cog-comfyui. Alternatively, you can replace inputs with URLs in your JSON workflow and the model will download them.",
            default=None,
        ),
        input_file_id: str = Input(
            description="Input image to download from cloud storage",
            default=None,
        ),
        bucket: str = Input(
            description="Bucket to upscale image from on cloud storage",
            default="360-panorama-sdxl"
        ),
        prompt: str = Input(
            description="Describe the image",
            default="Glowing mushrooms around pyramids amidst a cosmic backdrop"            
        ),
        suffix_prompt: str = Input(
            description="Suffix prompt for image generation (e.g., '360 panorama, equirectangular')",
            default="equirectangular, 360 panorama"
        ),
        negative_prompt: str = Input(
            description="Negative prompt for image generation (e.g., 'boring, text, signature, watermark, low quality, bad quality, grainy, blurry, long neck, closed eyes')",
            default="boring, text, signature, watermark, low quality, bad quality, grainy, blurry"
        ),
        lora_strength: float = Input(
            description="Strength of the LoRA model",
            default=1.0
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
        
        ) -> List[str]:

        # clean up directories
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # load the workflow
        if input_file:
            EXAMPLE_WORKFLOW_JSON = WORKFLOWS['upscale-input']
            self.cloud.bucket = BUCKETS['upscale']
        elif input_file_id:
            EXAMPLE_WORKFLOW_JSON = WORKFLOWS['upscale-input']
            self.cloud.bucket = BUCKETS['upscale']
        else:
            if upscale_by > 1:
                EXAMPLE_WORKFLOW_JSON = WORKFLOWS['upscale']
                self.cloud.bucket = BUCKETS['upscale']
            else:
                EXAMPLE_WORKFLOW_JSON = WORKFLOWS['base']
                self.cloud.bucket = BUCKETS['base']
                print(f"Using bucket: {self.cloud.bucket}")
                print(f"Using workflow: {EXAMPLE_WORKFLOW_JSON}")

        # load the workflow
        with open(EXAMPLE_WORKFLOW_JSON, "r") as file:
            wf = self.comfyUI.load_workflow(json.loads(file.read()))

        # handle input file
        if input_file:
            self.handle_input_file(input_file)

            # Update the input file in the workflow JSON
            wf['130']['inputs']['image'] = os.path.join(INPUT_DIR, f"input{os.path.splitext(input_file)[1]}")

            # build the prompts
            wf['6']['inputs']['text'] = prompt
            wf['6']['inputs']['text'] += ", " + suffix_prompt
            wf['7']['inputs']['text'] = negative_prompt

        elif input_file_id:
            # build the path to image.webp
            cloud_path = f"{input_file_id.strip('/')}/image.webp"

            # download the input file from cloud storage
            input_file = self.cloud.download_file_to_disk(cloud_path, f"{INPUT_DIR}/input.webp", bucket)

            # Update the input file in the workflow JSON
            wf['130']['inputs']['image'] = os.path.join(INPUT_DIR, f"input.webp")

            # download + parse workflow
            cloud_path = f"{input_file_id.strip('/')}/workflow.json"
            self.cloud.download_file_to_disk(cloud_path, f"{INPUT_DIR}/workflow.json", bucket)

            with open(f"{INPUT_DIR}/workflow.json", "r") as file:
                wf_og = json.loads(file.read())
            
            # update the workflow
            wf['6']['inputs']['text'] = wf_og['6']['inputs']['text'] # prompt
            wf['7']['inputs']['text'] = wf_og['7']['inputs']['text'] # negative prompt
        else:
            # build the prompts
            wf['6']['inputs']['text'] = prompt
            wf['6']['inputs']['text'] += ", " + suffix_prompt
            wf['7']['inputs']['text'] = negative_prompt

    
        # connect to comfyUI
        self.comfyUI.connect()

        # lora config
        wf['10']['inputs']['strength_model'] = lora_strength

        # seed
        self.comfyUI.randomise_seeds(wf)

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
        if input_file_id:
            image_hash = input_file_id.strip('/')

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
            "lora_strength": lora_strength,
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

        # TODO create animations if upscale_by > 1

        return [depth_url, image_url, metadata_url]