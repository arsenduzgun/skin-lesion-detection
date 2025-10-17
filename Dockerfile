# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps for Pillow + healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
      libjpeg62-turbo zlib1g curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Bake everything into the image (fallback if no mounts are provided) ---
COPY app.py /app/app.py
COPY templates/ /app/templates/
COPY static/ /app/static/
COPY model-dev/models/ /app/model-dev/models/
COPY model-dev/dataset/classes.json /app/model-dev/dataset/classes.json

# Defaults (compose can override); matches repo layout
ENV MODEL_PATH=/app/model-dev/models/model.keras \
    CLASSES_PATH=/app/model-dev/dataset/classes.json \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# Optional healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:5000/ || exit 1

# Gunicorn (no reload by default; compose will add --reload in dev)
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-t", "120", "-b", "0.0.0.0:5000", "app:app"]
