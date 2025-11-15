#!/bin/bash

# Get the absolute path of the script itself, resolving symlinks
script_full_path=$(readlink -f "$0")
script_directory=$(dirname "$script_full_path")

vr_steam_directory=$(find ~ -name "SteamVR" -type d -path "*/steamapps/common/SteamVR")

cd $script_directory

echo "Steam directory:"
echo $vr_steam_directory

echo "Installing requirements"
sudo apt update
sudo apt install python3.12 python3.12-venv

echo "Creating virtual environment"
python3.12 -m venv python_env
source python_env/bin/activate

pip install opencv-python mediapipe Pillow pygame argparse scipy numpy-quaternion

echo "Copying driver into SteamVR"
cp -r mediapipe_driver $vr_steam_directory/drivers

echo "Done!"
