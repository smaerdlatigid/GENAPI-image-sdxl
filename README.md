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
cog predict -i prompt="magical glowing mushrooms in space with pyramids"
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
    "prompt": "Magical mushroom forest in space",
    "suffix_prompt": "equirectangular, 360 panorama",
    "negative_prompt": "boring, text, signature, watermark, low quality, bad quality, grainy, blurry",
    "width": 2048,
    "height": 1024,
    "seed": -1,
    "lora_strength": 1.0,
    "cfg": 4.20,
    "steps": 10,
    "sampler": "dpmpp_sde_gpu",
    "scheduler": "karras",
    "tile_x": true,
    "tile_y": false,
    "output_format": "webp"
  }
}'
```

Inspect the container logs in docker to see if the request in action.

## Helpful Docker commands

`docker ps` - List all running containers

`docker logs <container_id>` - View the logs of a container

`docker rmi $(docker images -q)` - Remove all images

`docker rm $(docker ps -a -q)` - Remove all exited containers

## Extra

Create a mosaic to assess the best performing parameters

1. Change `BUCKET_IMAGE` in [.env]() to `test`
2. Start the microservice and mount the images dir: `-v ./images:/src/images`
3. Go into the container: `docker exec -it 360-panorama-sdxl bash`
4. Run the script: `./scripts/mosaic_settings.py`

![](images/mosaic_zoomed.png)