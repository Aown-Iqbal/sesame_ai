FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cloud/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sesame_ai/ /app/sesame_ai/
COPY cloud/app.py /app/app.py

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
