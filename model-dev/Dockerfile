# Includes Python, CUDA, cuDNN, AND Jupyter Lab
FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=1

# Minimal OS deps (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (optional; you can keep root if you prefer)
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /model-dev

# ---- Python deps ----
# IMPORTANT: do NOT include 'tensorflow' in requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Optional: a named kernel in Jupyter
RUN python -m ipykernel install --name docker-jupyter --display-name "Python (docker)"

# Switch to non-root (optional)
USER ${USERNAME}

# Jupyter is already in the base image; expose same port
EXPOSE 8888

# Use ServerApp.* flags in JupyterLab 4
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=devtoken"]
