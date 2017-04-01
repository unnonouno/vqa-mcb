FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-pip \
        python-setuptools \
    && rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
ENV CLONE_TAG=feature/20160617_cb_softattention

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/akirafukui/caffe.git .
COPY caffe/Makefile.config /opt/caffe/
RUN pip install --upgrade pip setuptools && \
    apt-get remove -y python-pip python-setuptools
RUN cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
#    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
#    mkdir build && cd build && \
#    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)"

RUN CPATH=`python -c 'import numpy; print(numpy.get_include())'` make pycaffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace

RUN git clone --depth 1 -b fix-path https://github.com/unnonouno/vqa-mcb.git .
RUN pip install \
    flask \
    spacy \
    opencv-python

RUN apt-get update && apt-get install -y unzip
RUN wget -q -O multi_att_2_glove_pretrained.zip https://www.dropbox.com/s/o19k39lvt5cm0bc/multi_att_2_glove_pretrained.zip?dl=0 && unzip multi_att_2_glove_pretrained.zip
COPY resnet_model/ResNet_mean.binaryproto /workspace/resnet_model/
COPY resnet_model/ResNet-152-deploy.prototxt /workspace/resnet_model/
COPY resnet_model/ResNet-152-model.caffemodel /workspace/resnet_model/

CMD python server/server.py
