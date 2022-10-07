FROM continuumio/anaconda3
COPY . /usr/app
EXPOSE 80
WORKDIR /usr/app
USER root
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
USER ec2-user
RUN pip install -r requirements.txt
CMD streamlit run vgg16_streamlit.py --server.port 80 --server.maxUploadSize 1024 --deprecation.showPyplotGlobalUse false