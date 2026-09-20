FROM python:3.11-slim

# HF Spaces run the container and expect the app on $app_port (7860).
WORKDIR /app

# Keep huggingface_hub's cache in a writable location.
ENV HF_HOME=/tmp/hf_home \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
