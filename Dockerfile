FROM continuumio/miniconda3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update
RUN apt-get install -y --no-install-recommends --allow-unauthenticated \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip \
    libffi-dev \
    software-properties-common

RUN mkdir /fever/
RUN mkdir /work

VOLUME /work
WORKDIR /fever

RUN git clone https://github.com/necla-ml/fever2018-model.git
WORKDIR /fever/fever2018-model
RUN cat best_params.all-title-one-r55.jl.a? >best_params.all-title-one-r55.jl && rm -v best_params.all-title-one-r55.jl.a?
WORKDIR /fever/

ADD requirements.txt /fever
RUN pip install -r requirements.txt
RUN python -m spacy download en

RUN mkdir /fever/fever2018-retrieval
RUN mkdir /fever/finetune-transformer-lm
RUN mkdir /fever/configs

ADD fever2018-retrieval /fever/fever2018-retrieval
ADD finetune-transformer-lm /fever/finetune-transformer-lm
ADD *.py /fever/
ADD configs /fever/configs
ADD predict.sh /fever/

ENV PYTHONPATH fever2018-retrieval/src:finetune-transformer-lm:.
CMD ["waitress-serve", "--host=0.0.0.0","--port=5000", "--call", "system:web"]
