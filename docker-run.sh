#!/bin/bash
# docker-run.sh - DGX Spark GPU inference helper
# OCR engine: EasyOCR (PyTorch-based, ARM64 + NVIDIA GPU native support)
#
# Usage:
#   bash docker-run.sh build                          # Build image
#   bash docker-run.sh run <video> [options]           # Run (local file, uses baked-in code)
#   bash docker-run.sh run-nas <video> [options]       # Run (NAS file, uses baked-in code)
#   bash docker-run.sh dev <video> [options]           # Run (local file, mounts local code — no rebuild needed)
#   bash docker-run.sh dev-nas <video> [options]       # Run (NAS file, mounts local code — no rebuild needed)
#   bash docker-run.sh shell                           # Interactive shell
#   bash docker-run.sh gpu-check                       # Verify GPU

IMAGE_NAME="video-sub-extrator:gpu"
OUTPUT_DIR="$(pwd)/output"
MODEL_CACHE="easyocr_model_cache"
# 本地代码挂载（dev 模式用，修改代码后无需重新 build）
DEV_MOUNTS="\
  -v $(pwd)/src:/app/src:ro \
  -v $(pwd)/main.py:/app/main.py:ro \
  -v $(pwd)/config.yaml:/app/config.yaml:ro"

mkdir -p "${OUTPUT_DIR}"

# ── 通用 docker run 函数 ────────────────────────────────────────────────────
_run_docker() {
  local extra_mounts="$1"; shift
  local input_arg="$1";    shift
  local extra_args="$@"

  docker run --rm --gpus all \
    ${extra_mounts} \
    -v "${OUTPUT_DIR}:/app/output" \
    -v "${MODEL_CACHE}:/root/.EasyOCR" \
    "${IMAGE_NAME}" \
    --input "${input_arg}" \
    --output /app/output \
    --backend easyocr \
    --use-gpu \
    ${extra_args}

  echo ""
  echo "📂 输出文件保存到宿主机: ${OUTPUT_DIR}/"
}

case "$1" in
  build)
    echo "Building Docker image: ${IMAGE_NAME}"
    echo "  Base: nvcr.io/nvidia/pytorch:26.04-py3 (NVIDIA NGC, DGX Spark optimized)"
    echo "  OCR engine: EasyOCR (ARM64 + NVIDIA GPU)"
    echo ""
    docker build -t "${IMAGE_NAME}" .
    echo ""
    echo "Build complete. Next steps:"
    echo "  bash docker-run.sh gpu-check"
    echo "  bash docker-run.sh run-nas /mnt/nas/Videos/xxx.mp4"
    ;;

  run)
    VIDEO_PATH="$2"
    if [ -z "${VIDEO_PATH}" ]; then
      echo "Usage: bash docker-run.sh run <video_path> [extra_args...]"
      exit 1
    fi
    shift 2
    VIDEO_DIR="$(dirname "${VIDEO_PATH}")"
    VIDEO_FILE="$(basename "${VIDEO_PATH}")"
    echo "Running OCR (GPU) on: ${VIDEO_PATH}"
    _run_docker \
      "-v ${VIDEO_DIR}:/input:ro" \
      "/input/${VIDEO_FILE}" \
      "$@"
    ;;

  run-nas)
    VIDEO_PATH="$2"
    if [ -z "${VIDEO_PATH}" ]; then
      echo "Usage: bash docker-run.sh run-nas /mnt/nas/Videos/xxx.mp4 [extra_args...]"
      exit 1
    fi
    shift 2
    echo "Running OCR (GPU) on NAS: ${VIDEO_PATH}"
    _run_docker \
      "-v /mnt/nas:/mnt/nas:ro" \
      "${VIDEO_PATH}" \
      "$@"
    ;;

  dev)
    # 开发模式：挂载本地 src/ main.py config.yaml，修改代码后无需 rebuild
    VIDEO_PATH="$2"
    if [ -z "${VIDEO_PATH}" ]; then
      echo "Usage: bash docker-run.sh dev <video_path> [extra_args...]"
      exit 1
    fi
    shift 2
    VIDEO_DIR="$(dirname "${VIDEO_PATH}")"
    VIDEO_FILE="$(basename "${VIDEO_PATH}")"
    echo "[DEV] Running OCR with local code (no rebuild needed): ${VIDEO_PATH}"
    _run_docker \
      "-v ${VIDEO_DIR}:/input:ro ${DEV_MOUNTS}" \
      "/input/${VIDEO_FILE}" \
      "$@"
    ;;

  dev-nas)
    # 开发模式 + NAS 路径
    VIDEO_PATH="$2"
    if [ -z "${VIDEO_PATH}" ]; then
      echo "Usage: bash docker-run.sh dev-nas /mnt/nas/Videos/xxx.mp4 [extra_args...]"
      exit 1
    fi
    shift 2
    echo "[DEV] Running OCR with local code on NAS: ${VIDEO_PATH}"
    _run_docker \
      "-v /mnt/nas:/mnt/nas:ro ${DEV_MOUNTS}" \
      "${VIDEO_PATH}" \
      "$@"
    ;;

  shell)
    echo "Opening interactive shell in container..."
    docker run --rm -it --gpus all \
      -v /mnt/nas:/mnt/nas:ro \
      -v "${OUTPUT_DIR}:/app/output" \
      -v "${MODEL_CACHE}:/root/.EasyOCR" \
      --entrypoint /bin/bash \
      "${IMAGE_NAME}"
    ;;

  gpu-check)
    echo "Checking GPU availability inside Docker..."
    docker run --rm --gpus all \
      --entrypoint python \
      "${IMAGE_NAME}" -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}:', torch.cuda.get_device_name(i))
else:
    print('WARNING: CUDA not available. Check nvidia-container-toolkit.')
"
    ;;

  *)
    echo "Usage:"
    echo "  bash docker-run.sh build                          # Build image"
    echo "  bash docker-run.sh run <video> [options]          # Run OCR (local file)"
    echo "  bash docker-run.sh run-nas <video> [options]      # Run OCR (NAS file)"
    echo "  bash docker-run.sh dev <video> [options]          # Run OCR, mount local code (no rebuild)"
    echo "  bash docker-run.sh dev-nas <video> [options]      # Run OCR on NAS, mount local code"
    echo "  bash docker-run.sh shell                          # Open container shell"
    echo "  bash docker-run.sh gpu-check                      # Verify GPU access"
    echo ""
    echo "Dev mode mounts: src/, main.py, config.yaml from current directory"
    ;;
esac
