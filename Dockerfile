FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt /app

RUN apt-get update

RUN apt install python3-pip -y

RUN apt-get install python3.10 -y

RUN python3 --version

RUN python3 -m pip install --upgrade pip

RUN apt install git -y

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install git+https://github.com/open-mmlab/mmocr.git@v0.6.0

RUN pip3 install mmdet==2.25.0

RUN pip install openmim

RUN mim install mmcv-full==1.5.3

RUN pip3 install -r requirements.txt