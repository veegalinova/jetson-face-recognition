FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y git cmake libsm6 libxext6 libxrender-dev libx11-dev && \
    apt-get install -y python3 python3-pip


RUN pip3 install scikit-build dlib

RUN pip3 install -U pip

COPY requirements.txt requirements.txt
# workaround for correct hnswlib installation
RUN pip3 install pybind11 numpy

RUN pip3 install -r requirements.txt
