#!/bin/bash

# Install BasicSR first (outside requirements.txt) to avoid wheel build errors
pip install --no-build-isolation git+https://github.com/XPixelGroup/BasicSR.git#egg=basicsr

# Install the rest of the dependencies
pip install -r requirements.txt
