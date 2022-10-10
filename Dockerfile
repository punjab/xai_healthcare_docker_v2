FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
ENV TORCH_CUDA_ARCH_LIST=Turing
ARG TORCH_CUDA_ARCH_LIST=Turing
ENV DEBIAN_FRONTEND="noninteractive" TZ="America/Vancouver"
ADD https://xai-healthcare.s3.amazonaws.com/vgg16_ft.pth .
ADD https://xai-healthcare.s3.amazonaws.com/trns_model.pt .
ADD https://xai-healthcare.s3.amazonaws.com/model_rn50_v2.pth .
COPY . /usr/app
EXPOSE 80
WORKDIR /usr/app
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip install --upgrade pip
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install -r ./requirements.txt
CMD streamlit run vgg16_streamlit.py --server.port 80 --server.maxUploadSize 1024 --deprecation.showPyplotGlobalUse false
