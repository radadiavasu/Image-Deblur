#!/bin/bash

# Install BasicSR first without isolation to avoid __version__ KeyError
pip install --no-build-isolation git+https://github.com/XPixelGroup/BasicSR.git#egg=basicsr

# Then install the rest of the project dependencies
pip install -r requirements.txt
