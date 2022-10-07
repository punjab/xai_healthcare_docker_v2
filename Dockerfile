FROM bitnami/pytorch
COPY . /usr/app
EXPOSE 80
WORKDIR /usr/app
USER root
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
CMD streamlit run vgg16_streamlit.py --server.port 80 --server.maxUploadSize 1024 --deprecation.showPyplotGlobalUse false