import os
import json
import shutil
import tarfile
import zipfile
import threading
from typing import List
import numpy as np

# uncomment to run cog predict
#from dotenv import load_dotenv
#load_dotenv()

from PIL import Image
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI

from minio_manager import MinioStorageManager as CloudStorageManager
from scripts.crop_animation import create_animation
from scripts.embedding import ImageTextEmbedding

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
ANIMATION_DIR = "/tmp/animation"
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

def cleanup_animation_thread(thread):
    """Helper function to join animation thread"""
    thread.join()
    print("Animation processing completed")


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
        # base generation
        seed: int = Input(
            description="Seed for the model, if negative will be random",
            default=-1
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
        # upscaling
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
            bucket = BUCKETS['upscale']
        elif input_file_id:
            EXAMPLE_WORKFLOW_JSON = WORKFLOWS['upscale-input']
            bucket = BUCKETS['upscale']
        else:
            if upscale_by > 1:
                EXAMPLE_WORKFLOW_JSON = WORKFLOWS['upscale']
                bucket = BUCKETS['upscale']
            else:
                EXAMPLE_WORKFLOW_JSON = WORKFLOWS['base']
                bucket = BUCKETS['base']

        self.cloud.bucket = bucket
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
        
        # download image from cloud storage
        elif input_file_id:
            # build the path to image.webp
            cloud_path = f"{input_file_id.strip('/')}/image.webp"

            # download the input file from cloud storage
            input_file = self.cloud.download_file_to_disk(cloud_path, f"{INPUT_DIR}/input.webp", BUCKETS['base'])

            # Update the input file in the workflow JSON
            wf['130']['inputs']['image'] = os.path.join(INPUT_DIR, f"input.webp")

            # download + parse workflow
            cloud_path = f"{input_file_id.strip('/')}/workflow.json"
            self.cloud.download_file_to_disk(cloud_path, f"{INPUT_DIR}/workflow.json", BUCKETS['base'])

            with open(f"{INPUT_DIR}/workflow.json", "r") as file:
                wf_og = json.loads(file.read())
            
            # update the workflow using the original workflow
            wf['6']['inputs']['text'] = wf_og['6']['inputs']['text'] # prompt
            wf['7']['inputs']['text'] = wf_og['7']['inputs']['text'] # negative prompt

            # force upscaling
            if upscale_by <= 1:
                upscale_by = 2.0
        else:
            # build the prompts
            wf['6']['inputs']['text'] = prompt
            wf['6']['inputs']['text'] += ", " + suffix_prompt
            wf['7']['inputs']['text'] = negative_prompt

            # base generation settings
            wf['11']['inputs']['cfg'] = cfg
            wf['11']['inputs']['steps'] = steps
            wf['11']['inputs']['sampler_name'] = sampler
            wf['11']['inputs']['scheduler'] = scheduler
    
        # connect to comfyUI
        self.comfyUI.connect()

        # seed
        self.comfyUI.randomise_seeds(wf)

        if seed > 0:
            # only use for base generation side
            if EXAMPLE_WORKFLOW_JSON == WORKFLOWS['base']:
                wf['11']['inputs']['seed'] = seed
            elif EXAMPLE_WORKFLOW_JSON == WORKFLOWS['upscale']:
                wf['7']['inputs']['seed'] = seed

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
        sizes = []
        for img in images:
            image = Image.open(img)
            sizes.append(image.size)
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
            "width": sizes[1][0], # first is depth, second is rgb
            "height": sizes[1][1],
            "prompt": prompt,
            "suffix_prompt": suffix_prompt,
            "negative_prompt": negative_prompt,
            "cfg": cfg,
            "steps": steps,
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

        # grab seeds for base generation
        if EXAMPLE_WORKFLOW_JSON == WORKFLOWS['upscale']:
            metadata['seed'] = wf['97']['inputs']['seed']
        elif EXAMPLE_WORKFLOW_JSON == WORKFLOWS['base']:
            metadata['seed'] = wf['11']['inputs']['seed']
        else:
            metadata['seed'] = -1

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

        # create animations if upscale_by > 1
        if EXAMPLE_WORKFLOW_JSON == WORKFLOWS['upscale'] or EXAMPLE_WORKFLOW_JSON == WORKFLOWS['upscale-input']:

            # Create embeddings in background
            embeddings_thread = threading.Thread(
                target=self.create_embeddings_background,
                args=(str(saved_images[1]), prompt, image_hash, self.cloud)
            )
            embeddings_thread.start()

            # Start animation creation in background
            animation_thread = threading.Thread(
                target=self.create_animation_background,
                args=(str(saved_images[1]), image_hash, self.cloud)
            )
            animation_thread.start()

            # Create cleanup thread
            cleanup_thread = threading.Thread(
                target=cleanup_animation_thread, 
                args=(animation_thread,)
            )
            cleanup_thread.daemon = True  # Only the cleanup thread is daemonized
            cleanup_thread.start()

        return [depth_url, image_url, metadata_url]

    @staticmethod 
    def create_embeddings_background(image_path: str, prompt: str, image_hash: str, cloud_manager):
        """
        Background task to create and upload CLIP embeddings for the image and prompt.
        Takes multiple perspective crops (front, back, up, down) from the equirectangular image.
        
        Args:
            image_path (str): Path to the input equirectangular image
            prompt (str): Text prompt used to generate the image
            image_hash (str): Hash/ID of the image
            cloud_manager: Instance of CloudStorageManager
        """
        try:

            # Initialize CLIP model
            embedding_model = ImageTextEmbedding()

            # Get text embedding
            text_embedding = embedding_model.encode_text([prompt])
            text_embedding_np = text_embedding.cpu().numpy()

            # Get embedding for the full equirectangular image
            full_image_embedding = embedding_model.encode_image(image_path)
            full_image_embedding_np = full_image_embedding.cpu().numpy()
            
            # Combine all embeddings into a single numpy array
            # First row: prompt embedding
            # Second row: full equirectangular image embedding
            # Following rows: perspective view embeddings
            combined_embeddings = np.vstack([
                text_embedding_np,
                full_image_embedding_np
            ])

            # Save embeddings to temporary file
            temp_path = f"/tmp/embeddings_{image_hash}.npy"
            np.save(temp_path, combined_embeddings)

            try:
                # Upload embeddings file
                embeddings_url = cloud_manager.upload_file(
                    temp_path,
                    f"{image_hash}/embeddings.npy",
                    'application/octet-stream'
                )
                print(f"Embeddings uploaded to: {embeddings_url}")
            except Exception as e:
                print(f"Failed to upload embeddings: {e}")

            # Cleanup
            os.remove(temp_path)

        except Exception as e:
            print(f"Failed to create embeddings: {str(e)}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def create_animation_background(image_path, image_hash, cloud_manager):
        """
        Background task to create and upload animation files.
        
        Args:
            image_path (str): Path to the input image
            image_hash (str): Hash/ID of the image
            cloud_manager: Instance of CloudStorageManager
        """
        try:
            # Create temporary directory for animation
            animation_dir = os.path.join(ANIMATION_DIR, f"animation_{image_hash}")
            os.makedirs(animation_dir, exist_ok=True)

            # Create animation
            create_animation(
                input_path=image_path,
                output_dir=animation_dir,
                fps=30,
                width=768,
                aspect_ratio=9./16,  # For equirectangular images
                fov=70.0,
                num_frames=300,
                cleanup=True
            )

            # Upload animation files
            try:
                # Upload MP4
                mp4_path = os.path.join(animation_dir, "animation.mp4")
                if os.path.exists(mp4_path):
                    mp4_url = cloud_manager.upload_file(
                        mp4_path, 
                        f"{image_hash}/animation.mp4",
                        'video/mp4'
                    )
                    print(f"Animation MP4 uploaded to: {mp4_url}")

            except Exception as e:
                print(f"Failed to upload animation files: {e}")

            # Cleanup
            shutil.rmtree(animation_dir, ignore_errors=True)

        except Exception as e:
            print(f"Failed to create animation: {e}")
