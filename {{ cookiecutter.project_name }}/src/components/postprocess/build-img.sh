#!/bin/bash

_CONTAINER_REGISTRY=us-central1-docker.pkg.dev
_PROJECT_ID=ml-projects-399119
_PROJECT_NAME=template
_IMG_NAME=postprocess
_RELEASE=latest

# docker build --tag $img_name:$version .
docker build --tag $_CONTAINER_REGISTRY/$_PROJECT_ID/$_PROJECT_NAME/$_IMG_NAME:$_RELEASE .
docker push $_CONTAINER_REGISTRY/$_PROJECT_ID/$_PROJECT_NAME/$_IMG_NAME:$_RELEASE
