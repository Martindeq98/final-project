# hash:sha256:9e5095d266e8d84c8579b6790a579ba84493f8f54835b9eca9d626eb43e4ef28
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy==1.20.2 \
        scipy==1.6.2 \
    && conda clean -ya
