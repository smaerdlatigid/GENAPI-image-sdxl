# GENAPI-image-sdxl
ML Microserver for image generation with sdxl and comfyui

## Setup

Ensure you have Linux/WSL2, [cog](https://github.com/replicate/cog/blob/main/docs/wsl2/wsl2.md), and the latest version of Docker installed.

Clone the repo with all submodules:

```bash
git clone --recurse-submodules git@github.com:smaerdlatigid/GENAPI-image-sdxl.git
```

Once you see the `ComfyUI` directory in the repo, you can install the custom nodes:

```sh
./scripts/install_custom_nodes.py
```

## Local Development

Build the Docker image and run ComfyUI to start the server:
```bash
cog run -p 8888 bash
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8888
```



