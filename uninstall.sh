#!/bin/bash

# Get the absolute path of the script itself, resolving symlinks
script_full_path=$(readlink -f "$0")
script_directory=$(dirname "$script_full_path")

vr_steam_directory=$(find ~ -name "SteamVR" -type d -path "*/steamapps/common/SteamVR")

cd $script_directory

echo "Removing driver"
rm -r $vr_steam_directory/drivers/mediapipe_driver

echo "Deleting environment"
rm -r python_env

echo "Deleting config"
rm config.json

echo "Done!"
