#!/bin/bash

echo "Running post-generation script..."

# Add your post-generation actions here
# ...

cd terraform && terraform init && terraform apply -auto-approve

echo "Post-generation script completed."
