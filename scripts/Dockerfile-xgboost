FROM ubuntu:18.04

# Install essential dependencies

RUN apt-get update && apt-get install -y \
      build-essential \
      curl \
      libcurl3-dev \
      git \
      libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /src && \
    cd /src && \
    git clone --depth 1 --recursive https://github.com/dmlc/xgboost.git && \
    cd /src/xgboost && make && \
    cp /src/xgboost/lib/libxgboost.so /usr/local/lib && ldconfig -n -v /usr/local/lib && \
    cp -r /src/xgboost/include/xgboost /usr/local/include/xgboost && \
    cp -r /src/xgboost/rabit/include/rabit /usr/local/include/rabit && \
    cp -r /src/xgboost/dmlc-core/include/dmlc /usr/local/include/dmlc && \
    rm -rf /src/xgboost
