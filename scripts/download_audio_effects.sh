#!/bin/bash

echo "Setting up Kaggle API credentials..."

# Create .kaggle directory
mkdir -p ~/.kaggle

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "⚠️  kaggle.json not found!"
  echo ""
  echo "Please follow these steps:"
  echo "1. Go to https://www.kaggle.com/settings"
  echo "2. Click 'Create New API Token'"
  echo "3. Move the downloaded file to ~/.kaggle/kaggle.json"
  echo ""
  echo "Example:"
  echo "mv ~/Downloads/kaggle.json ~/.kaggle/"
  echo "chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
echo "✅ Permissions set correctly"

# Show config
echo ""
echo "Current Kaggle configuration:"
kaggle config view

# Test connection
echo ""
echo "Testing connection..."
kaggle competitions list -v

# Create download directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
DOWNLOAD_DIR="${SCRIPT_DIR}/../data/dataset/audio-effects-datasets-vol-1"
mkdir -p "$DOWNLOAD_DIR"
echo "✅ Download directory ready: $DOWNLOAD_DIR"

# Download the dataset
echo ""
echo "Downloading dataset..."
kaggle datasets download -d 'riccardosimionato/audio-effects-datasets-vol-1' \
  -p "$DOWNLOAD_DIR" --unzip

echo "✅ Download complete!"
ls -lah "$DOWNLOAD_DIR"
