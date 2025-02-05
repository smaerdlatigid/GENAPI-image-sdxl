# GENAPI-image-sdxl
ML Microserver for panoramic image generation with sdxl and comfyui

![](animation_frames/test_1.webp)

![](animation_frames/animation.gif)

## Setup

Ensure you have Linux/WSL2, [cog](https://github.com/replicate/cog/blob/main/docs/wsl2/wsl2.md), and the latest version of Docker installed.

Clone the repo with all submodules:

```bash
git clone --recurse-submodules git@github.com:smaerdlatigid/GENAPI-image-sdxl.git
```

Build the docker image and go into the container so you can install the custom nodes:

```bash
cog run -p 8888 bash
```

Once in the container, run the following to install all the custom nodes:

```sh
./scripts/install_custom_nodes.py
```

## Local Development

To run the server locally, you can use the following command:

```bash
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8888
```
