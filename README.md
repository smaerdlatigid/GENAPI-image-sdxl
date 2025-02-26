# GENAPI-image-sdxl
ML Microserver for panoramic image generation with sdxl and comfyui

![](images/test_1.webp)

![](images/animation.gif)

## Setup

Ensure you have Linux/WSL2, [cog](https://github.com/replicate/cog/blob/main/docs/wsl2/wsl2.md), and the latest version of Docker installed.

Clone the repo with all submodules:

```sh
git clone --recurse-submodules git@github.com:smaerdlatigid/GENAPI-image-sdxl.git
```

Build the docker image and go into the container so you can install the custom nodes:

```sh
cog run -p 8888 bash
```

Once in the container, run the following:

```sh
./scripts/install_custom_nodes.py
```

Then, download all of the models + checkpoints:

```sh
./scripts/download_models.py
```

## Local Development

In the container, start the ComfyUI server:

```bash
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8888
```

If you want to customize your comfyui, you can add:
- new nodes in `custom_nodes.json`
- new models in `custom_models.json`

Then, run their respective scripts to install/download them and restart the server. If the nodes require new dependencies, you can add them to `cog.yaml`.

## Deployment

Prior to creating a deployment, ensure the predict function works as expected. Modify the `predict.py` file to use a dot env file by uncommenting the import. Then, set the `BUCKET_IMAGE` to `test` in `.env`. Run the following command to test the predict function:

```sh
cog predict -i prompt="surface of mars, glass geodesic domes"
```

Once you get a file_id back, you can use it to test the upscaling function:

```sh
cog predict -i input_file_id="49082b8a207de22ac76e83b2f1812f964d5bfd656b695cdaa1b45706f276459a" -i upscale_by=2.0
```

After successfully testing the predict function, you can proceed with deployment. Comment the import for dotenv and revert the bucket names to their original values. Build the docker image for deployment:

```sh
cog build -t 360-panorama-sdxl
```

Start the microservice:

```sh
docker run -d -p 7777:5000 --env-file ./.env --gpus all --name 360-panorama-sdxl 360-panorama-sdxl
```

Inspect the API docs in your browser at [http://localhost:7777/docs](). 

### Testing

Make sure the microservice is running and then test the endpoint:

```sh
curl -X 'POST' \
  'http://localhost:7777/predictions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": {
    "prompt": "Glowing mushrooms around pyramids amidst a cosmic backdrop",
    "suffix_prompt": "equirectangular, 360 panorama",
    "negative_prompt": "boring, text, signature, watermark, low quality, bad quality, grainy, blurry",
    "seed": -1,
    "cfg": 3.5,
    "steps": 12,
    "sampler": "dpmpp_sde",
    "scheduler": "ddim_uniform",
    "upscale_by": 0,
    "output_format": "webp"
  }
}'
```

You can test the upscaling function by adding the `input_file_id` from the previous request:

```sh
curl -X 'POST' \
  'http://localhost:7777/predictions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": {
    "input_file_id":"b1b9e6db0a9eac4758f367063bb4016fd2fb49f818e5fdd86fd52c6324b499dd",
    "upscale_by": 2,
    "upscale_steps": 10,
    "upscale_sampler": "uni_pc",
    "upscale_scheduler": "beta",
    "upscale_denoise": 0.3,
    "upscale_seed": -1,
    "output_format": "webp"
  }
}'
```

Inspect the container logs in docker to see if the request in action.

## Helpful Docker commands

`docker ps` - List all running containers

`docker logs <container_id>` - View the logs of a container

`docker rm $(docker ps -a -q)` - Remove all exited containers

`docker rmi $(docker images -q)` - Remove all images

`docker rmi $(docker images -f "dangling=true" -q)` - Remove all dangling images

### Mosaic Settings

Create a mosaic to assess the best performing parameters (sampler, scheduler, steps, cfg, etc.)

1. Change `BUCKET_IMAGE` in [.env]() to `test`
2. Start the microservice using the command above but add: `-v ./images:/src/images`
3. Go into the container: `docker exec -it 360-panorama-sdxl bash`
4. Run the script: `./scripts/mosaic_settings.py`

![](images/mosaic_zoomed.png)

### Cropped Animation 

To make a cropped animation from a 360 panorama:

1. Start the docker container in interactive mode:

```sh
docker run -it --rm -v ./images:/src/images --env-file ./.env --gpus 0 --name 360-panorama-sdxl-worker 360-panorama-sdxl bash
```

2. Run the script:

```sh
python scripts/crop_animation.py --input_path images/test_1.webp`
```