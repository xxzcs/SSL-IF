ARG BASE_IMAGE==nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install --no-install-recommends -y software-properties-common python3-distutils && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install --no-install-recommends -y python3.8 python3-pip python3.8-dev python3.8-distutils && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* 

    RUN ln -sf /usr/bin/python3.8 /usr/bin/python3

RUN pip3 install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117