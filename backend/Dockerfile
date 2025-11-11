FROM python:3.11-slim

# System deps for Pillow + healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
      libjpeg62-turbo zlib1g curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /backend

# Install runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Bake everything into the image (fallback if no mounts are provided) ---
COPY app.py /backend/app.py
COPY templates/ /backend/templates/
COPY static/ /backend/static/
COPY models/ /backend/models/

# Defaults (compose can override); matches repo layout
ENV MODEL_PATH=/backend/models/HAM10000_efficientnetB0.keras \
    META_PATH=/backend/models/HAM10000_meta.json \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# Optional healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:5000/ || exit 1

# Gunicorn (no reload by default; compose will add --reload in dev)
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-t", "120", "-b", "0.0.0.0:5000", "app:app"]