# Dockerfile for DGX Spark (ARM64 + NVIDIA GPU)
#
# Base: NVIDIA NGC PyTorch container, optimized for DGX hardware (Grace Blackwell / ARM64)
# Includes: CUDA 13.0, cuDNN 9, TensorRT, PyTorch - pre-tuned for DGX Spark
# NGC catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
FROM nvcr.io/nvidia/pytorch:26.04-py3

LABEL maintainer="video-sub-extrator"
LABEL description="Japanese video hard subtitle extractor - DGX Spark GPU"

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

# Python dependencies
# torch/torchvision already provided by NGC base image
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# Copy source code
COPY src/ ./src/
COPY main.py config.yaml ./

RUN mkdir -p /app/output /app/videos

# Pre-download EasyOCR Japanese model at build time
RUN python -c "\
import easyocr; \
print('Pre-loading EasyOCR Japanese model...'); \
reader = easyocr.Reader(['ja'], gpu=False, verbose=False); \
print('EasyOCR model cached.')" || echo "Model pre-download skipped (will download at first run)"

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
