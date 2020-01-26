FROM python:3.6-slim-stretch


# install stuff
RUN apt update && apt install -y python3-dev gcc
RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y git
#RUN apt-get install -y bash
RUN apt-get install -y curl

# Set the working directory fo /app 
WORKDIR /app

# add dir
COPY requirements.txt /app
COPY coco.txt /app

# install
RUN pip install -r requirements.txt
RUN pip install -r coco.txt
ENV PILLOW_VERSION=6.2.0


# install retinanet
COPY aerial_pedestrian_detection-master /app/aerial_pedestrian_detection-master
RUN cd aerial_pedestrian_detection-master && python setup.py build_ext --inplace && pip install .

# add dir
COPY . /app

# gcloud
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
RUN pip install --upgrade google-cloud-storage
#RUN gcloud init

# Install gcsfuse.
# RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" | tee /etc/apt/sources.list.d/gcsfuse.list
# RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
# RUN apt-get update
# RUN apt-get install -y gcsfuse


# Make port 80 available to the world outside the container
EXPOSE 80

# Define environment variable
ENV NAME World
ENV GOOGLE_APPLICATION_CREDENTIALS /app/key.json


# EXPOSE 5000
# CMD python app.py
CMD ["python", "app.py"] 

