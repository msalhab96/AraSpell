FROM python:latest

EXPOSE 6006

EXPOSE 8888

RUN apt-get update -y && apt-get upgrade -y

COPY . .

RUN pip install --upgrade

RUN pip install -r requirements.txt

RUN pip install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install jupyter
