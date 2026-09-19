FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir httpx==0.28.1

COPY loadgen.py .

CMD ["python", "loadgen.py"]
