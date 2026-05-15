#!/bin/bash
# docker-run.sh - 快速启动 Docker GPU 推理的辅助脚本
# 用法:
#   bash docker-run.sh build                     # 构建镜像
#   bash docker-run.sh run <video_path>           # 运行提取
#   bash docker-run.sh run <video_path> [options] # 带额外参数
#   bash docker-run.sh shell                      # 进入容器 shell
#   bash docker-run.sh gpu-check                  # 检查 GPU 是否可用

IMAGE_NAME="video-sub-extrator:gpu"
OUTPUT_DIR="$(pwd)/output"
MODEL_CACHE="paddle_model_cache"

mkdir -p "${OUTPUT_DIR}"

case "$1" in
  build)
    # Usage: bash docker-run.sh build [cuda_tag]
    # Examples:
    #   bash docker-run.sh build                                  # default: cuda12.3
    #   bash docker-run.sh build 3.0.0-gpu-cuda12.6-cudnn9.5     # specify tag
    PADDLE_TAG="${2:-3.3.0-gpu-cuda13.0-cudnn9.13}"
    echo "Building Docker image: ${IMAGE_NAME}"
    echo "  Base image: paddlepaddle/paddle:${PADDLE_TAG}"
    echo ""
    echo "Tip: Run 'nvidia-smi' on DGX Spark to check your CUDA version."
    echo "     Available tags: https://hub.docker.com/r/paddlepaddle/paddle/tags"
    echo ""
    docker build \
      --build-arg PADDLE_TAG="${PADDLE_TAG}" \
      -t "${IMAGE_NAME}" .
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

    # 判断是本地路径还是 NAS 路径
    VIDEO_DIR="$(dirname "${VIDEO_PATH}")"
    VIDEO_FILE="$(basename "${VIDEO_PATH}")"

    echo "Running OCR on: ${VIDEO_PATH}"
    echo "Extra args: ${EXTRA_ARGS}"

    docker run --rm --gpus all \
      -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
      -v "${VIDEO_DIR}:/input:ro" \
      -v "${OUTPUT_DIR}:/app/output" \
      -v "${MODEL_CACHE}:/root/.paddlex" \
      "${IMAGE_NAME}" \
      --input "/input/${VIDEO_FILE}" \
      --output /app/output \
      --use-gpu \
      ${EXTRA_ARGS}
    ;;

  run-nas)
    # 专门用于 NAS 路径，直接挂载整个 /mnt/nas
    VIDEO_PATH="$2"
    if [ -z "${VIDEO_PATH}" ]; then
      echo "Usage: bash docker-run.sh run-nas /mnt/nas/Videos/xxx.mp4 [extra_args...]"
      exit 1
    fi
    shift 2
    EXTRA_ARGS="$@"

    echo "Running OCR on NAS file: ${VIDEO_PATH}"

    docker run --rm --gpus all \
      -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
      -v /mnt/nas:/mnt/nas:ro \
      -v "${OUTPUT_DIR}:/app/output" \
      -v "${MODEL_CACHE}:/root/.paddlex" \
      "${IMAGE_NAME}" \
      --input "${VIDEO_PATH}" \
      --output /app/output \
      --use-gpu \
      ${EXTRA_ARGS}
    ;;

  shell)
    echo "Opening interactive shell in container..."
    docker run --rm -it --gpus all \
      -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
      -v /mnt/nas:/mnt/nas:ro \
      -v "${OUTPUT_DIR}:/app/output" \
      -v "${MODEL_CACHE}:/root/.paddlex" \
      --entrypoint /bin/bash \
      "${IMAGE_NAME}"
    ;;

  gpu-check)
    echo "Checking GPU availability inside Docker..."
    docker run --rm --gpus all \
      "${IMAGE_NAME}" python -c "
import paddle
print('PaddlePaddle version:', paddle.__version__)
print('CUDA available:', paddle.is_compiled_with_cuda())
if paddle.is_compiled_with_cuda():
    print('GPU count:', paddle.device.cuda.device_count())
    print('GPU name:', paddle.device.cuda.get_device_name(0))
" 2>/dev/null || docker run --rm "${IMAGE_NAME}" python -c "
import paddle
print('PaddlePaddle version:', paddle.__version__)
print('CUDA available:', paddle.is_compiled_with_cuda())
print('WARNING: No GPU detected, --gpus flag may not be working')
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
