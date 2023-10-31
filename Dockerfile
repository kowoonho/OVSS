# Ubuntu를 기반으로 한 Docker 이미지 사용
FROM dzw001/cuda11.1-cudnn8-python3.6-pytorch1.8.1-ubuntu18.04:latest

# 기본적인 툴 및 종속성 설치
RUN apt-get update
RUN apt-get install -y gnupg2
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y sudo
RUN apt-get install -y nano

ENV TERM="xterm-256color"


# Anaconda 설치 스크립트 다운로드
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O anaconda.sh && \
    chmod +x anaconda.sh && \
    ./anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh

# 환경 변수 설정
ENV PATH /opt/conda/bin:$PATH

# 초기화 스크립트 실행 (bash 셸을 사용할 때 필요)
RUN conda init bash