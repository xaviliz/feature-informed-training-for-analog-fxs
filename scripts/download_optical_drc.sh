#! bin/bash

# Downloads optical-dynamic-range-compressors-cl1b-la2a dataset from Kaggle and unzip all pickle files.

# Define script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
echo $SCRIPT_DIR

# Define dataset_dir as data/ConditioningMethods/datasets/ and dataset_name
DATASET_NAME="optical-dynamic-range-compressors-la-2a-cl-1b"
DATASET_DIR="${SCRIPT_DIR}/../data/dataset/${DATASET_NAME}"
echo "DATASET_DIR: ${DATASET_DIR}"
mkdir -p $DATASET_DIR

# Define the ZIP file path
ZIP_FILE_PATH=$DATASET_DIR/$DATASET_NAME.zip

# Download zip file in dataset_dir
echo "Downloading ZIP file in ${ZIP_FILE_PATH}"
curl -L -o $ZIP_FILE_PATH https://www.kaggle.com/api/v1/datasets/download/riccardosimionato/optical-dynamic-range-compressors-la-2a-cl-1b

# Unzip the file
unzip $ZIP_FILE_PATH -d $DATASET_DIR

# Clean ZIP file
rm $ZIP_FILE_PATH
