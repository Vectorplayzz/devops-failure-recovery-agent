FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir \
      fastapi==0.115.6 \
      "uvicorn[standard]"==0.34.0 \
      pydantic==2.10.4

COPY app.py .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
