FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
        cmake \
        libboost-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-math-dev \
        libboost-test-dev \
        zlib1g-dev \
        python3 \
        python3-pip \
        libboost-python-dev \
        git && \
        apt-get clean && apt-get autoclean

RUN git clone --recursive https://github.com/VowpalWabbit/vowpal_wabbit.git

WORKDIR /vowpal_wabbit

RUN make

WORKDIR /

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 7777

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--port=7777","--allow-root"]

