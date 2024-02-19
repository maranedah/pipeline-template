import os
from pathlib import Path

from google.auth import default

_, PROJECT_ID = default()
_CONTAINER_REGISTRY = "us-central1-docker.pkg.dev"
_PROJECT_NAME = "{{ cookiecutter.project_name }}"
_RELEASE = "latest"

BASE_IMAGE_PATH = os.path.join(_CONTAINER_REGISTRY, PROJECT_ID, _PROJECT_NAME)
INGESTOR_DOCKER_IMAGE = os.path.join(BASE_IMAGE_PATH, "ingestor") + f":{_RELEASE}"
PREPROCESSING_DOCKER_IMAGE = (
    os.path.join(BASE_IMAGE_PATH, "preprocess") + f":{_RELEASE}"
)
MODEL_DOCKER_IMAGE = os.path.join(BASE_IMAGE_PATH, "model") + f":{_RELEASE}"

PIPELINE_NAME = f"{_PROJECT_NAME}-pipeline"
COMPILED_PIPELINE_PATH = Path(__file__).parent / "pipeline.yaml"

SCHEDULE_DISPLAY_NAME = f"{_PROJECT_NAME}-pipeline-schedule"
