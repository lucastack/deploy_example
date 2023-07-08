FROM python:3.10

WORKDIR /root

RUN pip install --upgrade pip

COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

COPY . /root/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
