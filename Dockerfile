FROM python:3.7

ADD sashimi.py .

RUN apt-get update && apt-get install -y \
    python3-pip

RUN \ 
    pip3 install --no-cache-dir Cython

RUN pip3 install numpy==1.16.4

RUN \
    pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . /

CMD [ "python", "./sashimi.py" ]