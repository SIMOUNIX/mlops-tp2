FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-pip

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY model_utils.py .
COPY my_apy.py .

CMD ["uvicorn", "my_apy:app", "--host", "0.0.0.0", "--port", "8000"]