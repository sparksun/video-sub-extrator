#!/bin/bash
# setup.sh — 使用项目本地 venv 创建虚拟环境并安装所有依赖
# 用法: bash setup.sh

VENV_DIR="venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/${VENV_DIR}"
PIP="${VENV_PATH}/bin/pip"

echo "🚀 日文视频硬字幕提取工具 — 环境初始化"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  项目目录: ${SCRIPT_DIR}"
echo "  Venv 路径: ${VENV_PATH}"
echo ""

# ── 辅助函数：尝试多个 pip 镜像源 ────────────────────────────────
pip_install() {
    local MIRRORS=(
        "https://pypi.org/simple/"
        "https://mirrors.aliyun.com/pypi/simple/"
        "https://pypi.tuna.tsinghua.edu.cn/simple/"
    )
    for mirror in "${MIRRORS[@]}"; do
        echo "   尝试镜像: ${mirror}"
        local host
        host=$(echo "${mirror}" | awk -F/ '{print $3}')
        if "${PIP}" install -i "${mirror}" --trusted-host "${host}" "$@"; then
            return 0
        fi
        echo "   ⚠️  镜像失败，尝试下一个..."
    done
    echo "   ❌ 所有镜像均失败，请检查网络后重新运行 setup.sh"
    return 1
}

# ── Step 1: 创建 venv ─────────────────────────────────────────────
if [ -d "${VENV_PATH}" ]; then
    echo "✅ 检测到已有 venv，跳过创建"
else
    echo "📁 创建项目 venv..."
    python3 -m venv "${VENV_PATH}"
    echo "   完成: ${VENV_PATH}"
fi
echo ""

# ── Step 2: 升级 pip ─────────────────────────────────────────────
echo "⬆️  升级 pip..."
pip_install --upgrade pip -q
echo ""

# ── Step 3: 安装基础依赖 ──────────────────────────────────────────
echo "📦 安装基础依赖 (opencv, click, tqdm 等)..."
pip_install \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "Pillow>=10.0.0" \
    "click>=8.1.0" \
    "PyYAML>=6.0" \
    "tqdm>=4.65.0" \
    "ffmpeg-python>=0.2.0"
echo ""

# ── Step 4: 安装 PaddlePaddle (CPU, macOS/Apple Silicon) ─────────
echo "🧠 安装 PaddlePaddle (CPU 版, macOS/Apple Silicon)..."
pip_install paddlepaddle==2.6.2 \
    -f https://www.paddlepaddle.org.cn/whl/mac/cpu/stable.html
echo ""

# ── Step 5: 安装 PaddleOCR ───────────────────────────────────────
echo "🔍 安装 PaddleOCR..."
pip_install "paddleocr>=2.7.0"

# ── 完成 ─────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 环境安装完成！"
echo ""
echo "激活 venv 并运行："
echo "  source venv/bin/activate"
echo "  python main.py --input videos/test_japanese.mp4"
echo ""
echo "或直接用 venv Python（无需 activate）："
echo "  ./venv/bin/python main.py --input videos/test_japanese.mp4"
echo ""
