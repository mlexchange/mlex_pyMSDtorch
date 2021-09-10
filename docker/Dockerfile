FROM python:3.9

COPY docker/requirements.txt requirements.txt
# COPY setup setup
# COPY core core

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential\
         libsm6 \
         libxext6 \
         libxrender-dev \
         ffmpeg  \
    python3-pip &&\
    pip install --upgrade pip 
# RUN python setup/setup.py install 

COPY pymsdtorch pymsdtorch
RUN pip install -r requirements.txt
RUN pip install -e pymsdtorch/

WORKDIR /app/work/
# COPY scr scr

CMD ["bash"]