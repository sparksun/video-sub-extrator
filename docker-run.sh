#!/bin/bash
# docker-run.sh - DGX Spark GPU inference helper
# OCR engine: EasyOCR (PyTorch-based, ARM64 + NVIDIA GPU native support)
#
# Usage:
#   bash docker-run.sh build                     # Build image
#   bash docker-run.sh run <video_path>           # Run on local file
#   bash docker-run.sh run-nas <video_path>       # Run on NAS file
#   bash docker-run.sh shell                      # Interactive shell
#   bash docker-run.sh gpu-check                  # Verify GPU

IMAGE_NAME="video-sub-extrator:gpu"
OUTPUT_DIR="$(pwd)/output"
MODEL_CACHE="easyocr_model_cache"

mkdir -p "${OUTPUT_DIR}"

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
    EXTRA_ARGS="$@"
    VIDEO_DIR="$(dirname "${VIDEO_PATH}")"
    VIDEO_FILE="$(basename "${VIDEO_PATH}")"
    echo "Running OCR (GPU) on: ${VIDEO_PATH}"
    docker run --rm --gpus all \
      -v "${VIDEO_DIR}:/input:ro" \
      -v "${OUTPUT_DIR}:/app/output" \
      -v "${MODEL_CACHE}:/root/.EasyOCR" \
      "${IMAGE_NAME}" \
      --input "/input/${VIDEO_FILE}" \
      --output /app/output \
      --backend easyocr \
      --use-gpu \
      ${EXTRA_ARGS}
    ;;

  run-nas)
    VIDEO_PATH="$2"
    if [ -z "${VIDEO_PATH}" ]; then
      echo "Usage: bash docker-run.sh run-nas /mnt/nas/Videos/xxx.mp4 [extra_args...]"
      exit 1
    fi
    shift 2
    EXTRA_ARGS="$@"
    echo "Running OCR (GPU) on NAS: ${VIDEO_PATH}"
    docker run --rm --gpus all \
      -v /mnt/nas:/mnt/nas:ro \
      -v "${OUTPUT_DIR}:/app/output" \
      -v "${MODEL_CACHE}:/root/.EasyOCR" \
      "${IMAGE_NAME}" \
      --input "${VIDEO_PATH}" \
      --output /app/output \
      --backend easyocr \
      --use-gpu \
      ${EXTRA_ARGS}
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
    echo "  bash docker-run.sh build                        # Build image"
    echo "  bash docker-run.sh run <video> [options]        # Run OCR (local file)"
    echo "  bash docker-run.sh run-nas <video> [options]    # Run OCR (NAS file)"
    echo "  bash docker-run.sh shell                        # Open container shell"
    echo "  bash docker-run.sh gpu-check                    # Verify GPU access"
    ;;
esac
