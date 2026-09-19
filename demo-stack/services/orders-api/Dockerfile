FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir \
      fastapi==0.115.6 \
      "uvicorn[standard]"==0.34.0 \
      httpx==0.28.1 \
      psycopg2-binary==2.9.10 \
      pydantic==2.10.4

COPY app.py .

EXPOSE 8080

# Single worker on purpose: an OOM kill must take the whole container down so
# `docker inspect` reports State.OOMKilled, which is the signal the agent reads.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
