# Dockerfile for DGX Spark (ARM64 / aarch64 + NVIDIA GPU)
#
# Uses PyTorch official ARM64 CUDA image as base.
# EasyOCR (PyTorch-based) supports ARM64 + NVIDIA GPU natively.
#
# Check CUDA version: nvidia-smi | grep "CUDA Version"
# DGX Spark CUDA 13.0 → use PyTorch with CUDA 12.6 (forward compatible)
#
ARG PYTORCH_TAG=2.7.0-cuda12.6-cudnn9-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

LABEL maintainer="video-sub-extrator"
LABEL description="Japanese video hard subtitle extractor (GPU, ARM64)"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# pytorch/pytorch base image already includes torch + CUDA
# easyocr uses torch for GPU inference on ARM64
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# Copy source code
COPY src/ ./src/
COPY main.py config.yaml ./

# Create I/O directories
RUN mkdir -p /app/output /app/videos

# Pre-download EasyOCR Japanese model at build time
RUN python -c "\
import easyocr; \
print('Pre-loading EasyOCR Japanese model...'); \
reader = easyocr.Reader(['ja'], gpu=False, verbose=False); \
print('EasyOCR model cached.')" || echo "Model pre-download skipped"

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
