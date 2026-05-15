#!/bin/bash
# setup.sh - Create project venv and install all dependencies
# Usage: bash setup.sh

VENV_DIR="venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/${VENV_DIR}"
PIP="${VENV_PATH}/bin/pip"

echo "Starting environment setup..."
echo "Project: ${SCRIPT_DIR}"
echo "Venv:    ${VENV_PATH}"
echo ""

# Helper: try multiple pip mirrors in order
pip_install() {
    local mirrors=(
        "https://mirrors.aliyun.com/pypi/simple/"
        "https://pypi.org/simple/"
        "https://pypi.tuna.tsinghua.edu.cn/simple/"
    )
    local host
    local mirror
    for mirror in "${mirrors[@]}"; do
        echo "  Trying mirror: ${mirror}"
        host=$(echo "${mirror}" | awk -F/ '{print $3}')
        if "${PIP}" install -i "${mirror}" --trusted-host "${host}" "$@"; then
            return 0
        fi
        echo "  Mirror failed, trying next..."
    done
    echo "  ERROR: All mirrors failed. Check network and re-run setup.sh"
    return 1
}

# Step 1: Create venv
if [ -d "${VENV_PATH}" ]; then
    echo "[1/5] venv already exists, skipping."
else
    echo "[1/5] Creating venv..."
    python3 -m venv "${VENV_PATH}"
    echo "      Done: ${VENV_PATH}"
fi
echo ""

# Step 2: Upgrade pip
echo "[2/5] Upgrading pip..."
pip_install --upgrade pip -q
echo ""

# Step 3: Install base dependencies
echo "[3/5] Installing base dependencies (opencv, click, tqdm, etc.)..."
pip_install \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "Pillow>=10.0.0" \
    "click>=8.1.0" \
    "PyYAML>=6.0" \
    "tqdm>=4.65.0" \
    "ffmpeg-python>=0.2.0"
echo ""

# Step 4: Install PaddlePaddle CPU (macOS / Apple Silicon)
echo "[4/5] Installing PaddlePaddle CPU (v3.x required by PaddleOCR 3.x)..."
pip_install "paddlepaddle>=3.0.0"
echo ""

# Step 5: Install PaddleOCR
echo "[5/5] Installing PaddleOCR..."
pip_install "paddleocr>=2.7.0"

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "Activate venv and run:"
echo "  source venv/bin/activate"
echo "  python main.py --input videos/test_japanese.mp4"
echo ""
echo "Or run directly without activating:"
echo "  ./venv/bin/python main.py --input videos/test_japanese.mp4"
echo ""
