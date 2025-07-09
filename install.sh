#!/bin/bash

# Install BasicSR from Git (bypassing wheel build issues)
pip install --no-build-isolation git+https://github.com/XPixelGroup/BasicSR.git#egg=basicsr

# Install rest of the Python packages
pip install -r requirements.txt

# Create weights folder
mkdir -p weights

# Download pretrained model files from Google Drive
echo "Downloading RealESRGAN_x4plus.pth..."
gdown https://drive.google.com/uc?export=download&id=1ANr9r3nvbPFm1WeSqU8m5uKo02EQsill -O weights/RealESRGAN_x4plus.pth 

echo "Downloading NAFNet-GoPro-width64.pth..."
gdown https://drive.google.com/uc?export=download&id=1ktLPKu3wwu_3ZkIHslL5OdG7pPLPC1Vc -O weights/NAFNet-GoPro-width64.pth 
