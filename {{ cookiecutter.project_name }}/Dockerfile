FROM python:3.10-slim-buster
ARG PROJECT_DIRECTORY="/usr/src/app"

WORKDIR ${PROJECT_DIRECTORY}

RUN apt-get update && apt-get install g++ -yy && apt-get install git -yy
RUN pip install --upgrade pip

COPY . .

RUN pip install --quiet -r requirements.txt
RUN pip install --quiet -r requirements-dev.txt

RUN git init . && pre-commit install
RUN pre-commit run --all

RUN pytest
