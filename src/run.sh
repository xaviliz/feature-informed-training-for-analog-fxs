#!/bin/bash

set -e

Help() {
  echo "Usage: $0 [-gpu gpu] [-h]"
  echo "  -n Set the gpu device to run"
  echo "  -h Show help message"
}

while getopts "n:h" option; do
  case $option in
  n) GPU="$OPTARG" ;;
  h)
    Help
    exit
    ;;
  *)
    echo "Invalid option"
    exit 1
    ;;
  esac
done

export CUDA_VISIBLE_DEVICES=${GPU}
export TF_CPP_MIN_LOG_LEVEL="3"

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

cd "$( # cd to the directory of the script
  dirname "$0"
)"

# Start training
# python starter.py --config config/la2a.json
python starter.py --config config/od300.json
