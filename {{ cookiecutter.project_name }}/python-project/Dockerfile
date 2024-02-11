FROM python:3.10-slim-buster
ARG PROJECT_DIRECTORY="/usr/src/app"

WORKDIR ${PROJECT_DIRECTORY}

RUN apt-get update && apt-get install g++ -yy && apt-get install git -yy

COPY src/components/ingestor/requirements.txt src/components/ingestor/requirements.txt 
COPY src/components/preprocess/requirements.txt src/components/preprocess/requirements.txt 
COPY src/components/model/requirements.txt src/components/model/requirements.txt 
COPY src/components/postprocess/requirements.txt src/components/postprocess/requirements.txt 
COPY src/pipelines/pipeline/requirements.txt src/components/pipeline/requirements.txt 

RUN pip install -r src/components/ingestor/requirements.txt && \
    pip install -r src/components/preprocess/requirements.txt && \
    pip install -r src/components/model/requirements.txt && \
    pip install -r src/components/postprocess/requirements.txt && \
    pip install -r src/pipelines/pipeline/requirements.txt && \
    pip install -r requirements-dev.txt 

COPY src/ src/
COPY requirements.txt requirements.txt 
RUN pip install --quiet -r requirements.txt 
RUN git init . && pre-commit install 

COPY . .