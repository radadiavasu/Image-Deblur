#!/bin/bash

# Install BasicSR first
pip install --no-build-isolation git+https://github.com/XPixelGroup/BasicSR.git#egg=basicsr

# Install Python dependencies
pip install -r requirements.txt

# Install gdown
pip install gdown

# Create weights directory
mkdir -p weights

# Download weights from Google Drive using gdown
echo "Downloading RealESRGAN_x4plus.pth..."
gdown --id 1ANr9r3nvbPFm1WeSqU8m5uKo02EQsill -O weights/RealESRGAN_x4plus.pth

echo "Downloading NAFNet-GoPro-width64.pth..."
gdown --id 1ktLPKu3wwu_3ZkIHslL5OdG7pPLPC1Vc -O weights/NAFNet-GoPro-width64.pth
