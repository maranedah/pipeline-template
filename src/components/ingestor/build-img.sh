#!/bin/bash

project_name=example-project
base_tag=us.gcr.io/$project_id/$project_name/pipeline
img_name=ingestor
version=latest

docker build -t $base_tag/$img_name:$version
docker push $base_tag/$img_name:$version