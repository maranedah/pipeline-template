import os

from google.auth import default

_, PROJECT_ID = default()

BASE_IMAGE_PATH = os.path.join("us.gcr.io", PROJECT_ID, "imager", "stable-diffusion")
INGESTOR_DOCKER_IMAGE = os.path.join(BASE_IMAGE_PATH, "ingestor") + ":latest"
PREPROCESSING_DOCKER_IMAGE = os.path.join(BASE_IMAGE_PATH, "preprocess") + ":latest"
MODEL_DOCKER_IMAGE = os.path.join(BASE_IMAGE_PATH, "model") + ":latest"

PIPELINE_NAME = "stable-diffusion-pipeline"
COMPILED_PIPELINE_PATH = "pipeline.yaml"

SCHEDULE_DISPLAY_NAME = "stable-diffusion-pipeline-schedule"
