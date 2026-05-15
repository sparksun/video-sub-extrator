# Dockerfile
# PaddlePaddle official GPU image from Docker Hub
# Use CUDA 12.6 + cuDNN 9.5 (compatible with DGX Spark / modern NVIDIA GPUs)
# Check your CUDA version with: nvidia-smi
# Available tags: https://hub.docker.com/r/paddlepaddle/paddle/tags
ARG PADDLE_TAG=3.3.0-gpu-cuda13.0-cudnn9.13
FROM paddlepaddle/paddle:${PADDLE_TAG}

LABEL maintainer="video-sub-extrator"
LABEL description="Japanese video hard subtitle extractor (GPU)"

WORKDIR /app

# System dependencies: FFmpeg, OpenCV runtime libs, CJK fonts
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
# paddlepaddle-gpu is already provided by the base image — skip it here
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# Copy source code
COPY src/ ./src/
COPY main.py config.yaml ./

# Create I/O directories
RUN mkdir -p /app/output /app/videos

# Pre-download PaddleOCR Japanese models at build time
# (cached in image layer → no download delay at runtime)
RUN PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -c "\
from paddleocr import PaddleOCR; \
print('Pre-loading Japanese OCR models...'); \
ocr = PaddleOCR(lang='japan', device='cpu', use_textline_orientation=True, text_rec_score_thresh=0.3); \
print('Models cached successfully.')" || echo "Model pre-download skipped (will download at first run)"

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
