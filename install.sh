#!/bin/bash

# Install BasicSR from Git (bypassing wheel build issues)
pip install --no-build-isolation git+https://github.com/XPixelGroup/BasicSR.git#egg=basicsr

# Install rest of the Python packages
pip install -r requirements.txt
