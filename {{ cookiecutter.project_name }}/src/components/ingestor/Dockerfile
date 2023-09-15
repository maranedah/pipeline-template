FROM python:3.10-slim-buster
ARG PROJECT_DIRECTORY="/usr/src/app"

WORKDIR ${PROJECT_DIRECTORY}

RUN apt-get update && apt-get install g++ -yy && apt-get install git -yy
RUN pip install --upgrade pip

RUN pip install .