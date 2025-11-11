# Skin Lesion Detection (HAM10000)

This project provides a web-based interface and development environment for classifying skin lesions from dermoscopic images using a pretrained EfficientNet-based model trained on the HAM10000 dataset.

## Requirements

- **Docker**
- **Git LFS**

## Run the Web Application

### Build & Start

docker compose up --build -d

### Access

http://localhost:5000

### Stop

docker compose down

## Run the Jupyter Notebook (Model Development)

### Build the Development Image

docker build -t model-dev-image -f model-dev/Dockerfile model-dev

### Run the Container

docker run -d --rm --name model-dev-container --gpus all --shm-size=8g -p 8888:8888 -v "${PWD}:/workspace" -w /workspace/model-dev model-dev-image

### Access

http://127.0.0.1:8888/?token=devtoken

### Stop Container

docker stop model-dev-container

### Remove Image

docker rmi model-dev-image

## Dataset

This project uses the publicly available HAM10000 dermoscopic image dataset.

You can access and download it here: https://api.isic-archive.com/collections/212/

This project focuses on developing a deep learning-based system to classify skin lesions using dermatoscopic images from the HAM10000 dataset. The model is designed to predict the type of skin lesion based on image input, offering a potential tool for assisting dermatologists in diagnosing skin conditions, including various types of skin cancer.