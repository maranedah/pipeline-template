#!/bin/bash
SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Building and pushing ingestor"
cd $SCRIPT_DIR/src/components/ingestor
bash build-img.sh

echo "Building and pushing preprocess"
cd $Script_DIR/src/components/preprocess
bash build-img.sh

echo "Building and pushing model"
cd $Script_DIR/src/components/model
bash build-img.sh

echo "Building and pushing postprocess"
cd $Script_DIR/src/components/postprocess
bash build-img.sh

echo "All images built and pushed"
