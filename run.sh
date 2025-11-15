#!/bin/bash

# Get the absolute path of the script itself, resolving symlinks
script_full_path=$(readlink -f "$0")
script_directory=$(dirname "$script_full_path")

cd $script_directory

source python_env/bin/activate

echo "Running..."
python3 mediapipe_vr_sender.py
