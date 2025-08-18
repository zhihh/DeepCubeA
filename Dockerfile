# FROM nvidia/cuda:10.2-base-ubuntu18.04
FROM perfectweb/base:cuda-10.2-devel-ubuntu18.04
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y python3.7 curl python3-distutils
RUN ln -s /usr/bin/python3.7 /usr/bin/python
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# python3.7已经被归档了，如下
RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN mkdir /deepcube
COPY ./requirements.txt /deepcube
WORKDIR /deepcube
RUN pip install -r requirements.txt
