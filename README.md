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

Prior to creating a deployment, ensure the predict function works as expected. You can test this by running the following:

```sh
cog predict -i prompt="surface of mars, glass geodesic dome buildings with plants inside" -i bucket=test
```f

Once you get a file_id back, you can use it to test the upscaling function:

```sh
cog predict -i input_file_id="4bb8848392a5e6f6190d6d6b75587182ea85132dc10764e3dacca077a5082dc3" -i upscale_by=2.0 -i bucket=test
```

Build the docker image for deployment:

```sh
cog build -t 360-panorama-sdxl
```

Start the microservice:

```sh
docker run -d -p 7777:5000 --env-file ./.env --gpus all --name 360-panorama-sdxl 360-panorama-sdxl
```

Make sure all of the environment variables are set correctly. Inspect the API docs in your browser at [http://localhost:7777/docs](). 

### Testing

Make sure the microservice is running and then test using:

```sh
curl -X 'POST' \
  'http://localhost:7777/predictions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": {
    "bucket": "360-panorama-sdxl",
    "prompt": "Glowing mushrooms around pyramids amidst a cosmic backdrop",
    "suffix_prompt": "equirectangular, 360 panorama",
    "negative_prompt": "boring, text, signature, watermark, low quality, bad quality, grainy, blurry",
    "seed": -1,
    "cfg": 3.5,
    "steps": 12,
    "sampler": "dpmpp_sde",
    "scheduler": "ddim_uniform",
    "upscale_by": 0,
    "upscale_steps": 10,
    "upscale_sampler": "uni_pc",
    "upscale_scheduler": "beta",
    "upscale_denoise": 0.3,
    "upscale_seed": -1,
    "output_format": "webp"
  }
}'
```

You can test the upscaling function by adding the `input_file_id` from the previous request:

```
"input_file_id":"88837afa2c321b0e81e67c77f515284d037b8fa5b5834acc4f8cb35944681185",
```

Inspect the container logs in docker to see if the request in action.

## Helpful Docker commands

`docker ps` - List all running containers

`docker logs <container_id>` - View the logs of a container

`docker rm $(docker ps -a -q)` - Remove all exited containers

`docker rmi $(docker images -q)` - Remove all images

`docker rmi $(docker images -f "dangling=true" -q)` - Remove all dangling images

## Extra

Create a mosaic to assess the best performing parameters

1. Change `BUCKET_IMAGE` in [.env]() to `test`
2. Start the microservice and mount the images dir: `-v ./images:/src/images`
3. Go into the container: `docker exec -it 360-panorama-sdxl bash`
4. Run the script: `./scripts/mosaic_settings.py`

![](images/mosaic_zoomed.png)