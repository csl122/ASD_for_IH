#!/bin/bash

cd /media/Adapter-Diffusion-IH/



apt install python3.8-venv -y
python -m venv torch
source torch/bin/activate

pip install --upgrade pip
pip install xformers==0.0.20 pytorch_lightning transformers diffusers invisible_watermark==0.1.5 
pip install basicsr==1.4.2 einops==0.6.0 omegaconf==2.3.0 gradio opencv-python
pip install pudb imageio imageio-ffmpeg k-diffusion webdataset open-clip-torch kornia safetensors timm

apt-get update && apt-get install libgl1 -y

mkdir datasets

cd datasets
curl -O -L 'http://storage.live.com/items/791C94EE7340D2CD!1043923:/datasetss.zip'
curl -O -L 'http://storage.live.com/items/791C94EE7340D2CD!1058506:/val2017_256_KDD.zip'
curl -O -L 'http://storage.live.com/items/791C94EE7340D2CD!1058486:/val2017_256_UDD.zip'
unzip datasets.zip
unzip val2017_256_KDD.zip
unzip val2017_256_UDD.zip

cd ../models

curl -O -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt

# curl -O -L https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_color_sd14v1.pth

curl -O -L 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt'
