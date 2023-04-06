FROM python:3.9

COPY docker/requirements.txt requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential\
         libsm6 \
         libxext6 \
         libxrender-dev \
         ffmpeg  \
	 tree \
    python3-pip &&\
    pip install --upgrade pip  

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/phzwart/dlsia.git

WORKDIR /app/work/
COPY src src

CMD ["bash"]