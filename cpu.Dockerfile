FROM tensorflow/tensorflow:1.8.0-py3

RUN mkdir /root/mimic2
WORKDIR /root/mimic2

COPY requirements.txt /root/mimic2/requirements.txt
RUN pip install --upgrade pip && pip install  --no-cache-dir -r requirements.txt
RUN apt update && apt install -y ffmpeg

COPY . /root/mimic2

ENTRYPOINT [ "/bin/bash" ]
