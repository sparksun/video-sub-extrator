# Dockerfile
# 基于 PaddlePaddle 官方 GPU 镜像（CUDA 12.3 + cuDNN 9）
# 适配 DGX Spark (NVIDIA GPU)
FROM registry.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.3-cudnn9.0-trt8.6

LABEL maintainer="video-sub-extrator"
LABEL description="日文视频硬字幕提取工具 (GPU 版)"

# 设置工作目录
WORKDIR /app

# 安装系统依赖（FFmpeg、字体等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-gpu.txt .

# 安装 Python 依赖（跳过 paddlepaddle，镜像已内置 GPU 版）
RUN pip install --no-cache-dir \
    "opencv-python-headless>=4.8.0" \
    "numpy>=1.24.0" \
    "Pillow>=10.0.0" \
    "click>=8.1.0" \
    "PyYAML>=6.0" \
    "tqdm>=4.65.0" \
    "ffmpeg-python>=0.2.0" \
    "paddleocr>=2.7.0" \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# 复制项目代码
COPY src/ ./src/
COPY main.py config.yaml ./

# 创建输出目录
RUN mkdir -p /app/output /app/videos

# 预下载 PaddleOCR 日文模型（构建时缓存，避免运行时下载）
RUN PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -c "\
from paddleocr import PaddleOCR; \
print('Pre-downloading Japanese OCR models...'); \
ocr = PaddleOCR(lang='japan', device='cpu', use_textline_orientation=True, text_rec_score_thresh=0.3); \
print('Models ready.')" || true

# 设置环境变量
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV PYTHONUNBUFFERED=1

# 默认入口
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
