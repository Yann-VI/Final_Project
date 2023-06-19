#FROM jjanzic/docker-python3-opencv
#FROM python:3.10.12
FROM hdgigante/python-opencv:4.7.0-ubuntu

WORKDIR /home/app

COPY requirements.txt /dependencies/requirements.txt
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r /dependencies/requirements.txt

COPY . /home/app

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV ARTIFACT_STORE_URI=$ARTIFACT_STORE_URI

CMD gunicorn app:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker