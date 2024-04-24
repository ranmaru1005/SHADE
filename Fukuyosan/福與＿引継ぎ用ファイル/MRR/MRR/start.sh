#!/bin/bash
# Install or update needed software
sudo apt-get update
sudo apt-get install -yq git python3.8-venv
# Python environment setup
python3.8 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

ulimit -n 8096
cd Simulator
python load_NN.py
# gsutil cp -r result gs://mrr-result
sudo poweroff