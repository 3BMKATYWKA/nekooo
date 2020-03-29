FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

ENV PYTHONPATH=".:$PYTHONPATH"

COPY entrypoint.sh ./
COPY nekooo ./nekooo/

CMD ["bash", "entrypoint.sh"]
