# 日文视频硬字幕提取工具

从 MP4 等视频文件中自动提取硬字幕（burned-in subtitle），输出为 Markdown 表格或 TXT 文件，方便日语学习与字幕整理。

## 功能特性

- 自动采样视频帧，提取字幕区域并进行 OCR 识别
- 支持双 OCR 引擎：**PaddleOCR**（macOS/CPU）和 **EasyOCR**（DGX Spark/GPU）
- 相邻重复字幕自动去重合并
- 输出带时间戳的 Markdown 表格和 TXT 文件
- 支持竖排日文字幕识别

## 环境要求

| 环境 | 系统 | OCR 引擎 |
|------|------|----------|
| 本地开发 | macOS (Apple Silicon / x86) | PaddleOCR (CPU) |
| 生产推理 | DGX Spark (ARM64 + NVIDIA GPU) | EasyOCR (GPU, via Docker) |

---

## 本地安装（macOS）

### 1. 初始化虚拟环境并安装依赖

```bash
bash setup.sh
```

脚本会自动：
- 在项目目录创建 `venv/`
- 安装 PaddlePaddle 3.x (CPU)、PaddleOCR、OpenCV 等全部依赖
- 支持阿里云 / PyPI 多镜像源自动切换

### 2. 激活 venv

```bash
source venv/bin/activate
```

### 3. 运行（首次运行会自动下载 OCR 模型，约 300MB）

```bash
python main.py --input videos/test_japanese.mp4
```

---

## DGX Spark 部署（Docker + GPU）

### 前提

确认 NVIDIA Container Toolkit 已安装，且 `docker run --gpus all` 可用。

### 1. 构建镜像

基础镜像：`nvcr.io/nvidia/pytorch:26.04-py3`（NVIDIA NGC 官方，DGX Spark 专属优化）

```bash
bash docker-run.sh build
```

### 2. 验证 GPU

```bash
bash docker-run.sh gpu-check
# 预期输出: CUDA available: True  GPU 0: NVIDIA ...
```

### 3. 处理视频

```bash
# NAS 视频
bash docker-run.sh run-nas /mnt/nas/Videos/xxx.mp4

# 本地视频
bash docker-run.sh run /path/to/video.mp4

# 进入容器调试
bash docker-run.sh shell
```

---

## 使用说明

### 基本命令

```bash
python main.py --input <视频路径> [选项]
```

### 所有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 输入视频路径 |
| `--output` | `output/` | 输出目录 |
| `--format` | `md,txt` | 输出格式，逗号分隔 |
| `--fps` | `1.0` | 采样帧率（帧/秒），越高越准但更慢 |
| `--subtitle-region` | `bottom20` | 字幕垂直区域预设（见下表） |
| `--subtitle-bbox` | — | 精确字幕矩形 `x1,y1,x2,y2`（0.0–1.0），优先于 `--subtitle-region` |
| `--confidence` | `0.7` | OCR 置信度阈值（0.0–1.0） |
| `--merge-threshold` | `0.85` | 去重合并相似度阈值 |
| `--backend` | `paddleocr` | OCR 引擎：`paddleocr` 或 `easyocr` |
| `--use-gpu` | 否 | 启用 GPU 加速 |
| `--no-timestamp` | — | 输出文件中不含时间戳 |

### 字幕区域预设

| 预设 | 说明 |
|------|------|
| `bottom10` | 视频底部 10% |
| `bottom15` | 视频底部 15% |
| `bottom20` | 视频底部 20%（**默认**） |
| `bottom25` | 视频底部 25% |
| `bottom30` | 视频底部 30% |
| `bottom50` | 视频底部 50% |
| `full` | 整帧（适合字幕位置不固定的视频） |

### 使用示例

```bash
# 快速测试（低帧率 + 宽字幕区域）
python main.py --input video.mp4 --fps 0.5 --subtitle-region bottom30

# 高精度（高帧率 + 低置信度阈值）
python main.py --input video.mp4 --fps 2.0 --confidence 0.5

# 避开角落台标/水印：精确指定字幕矩形（水平10%-90%，垂直75%-100%）
python main.py --input video.mp4 --subtitle-bbox "0.1,0.75,0.9,1.0"

# 字幕在视频中央（如竖屏视频）
python main.py --input video.mp4 --subtitle-bbox "0.05,0.85,0.95,1.0"

# 指定 PaddleOCR 引擎（macOS）
python main.py --input video.mp4 --backend paddleocr

# 指定 EasyOCR + GPU（DGX Spark 直接运行）
python main.py --input video.mp4 --backend easyocr --use-gpu
```

---

## 项目结构

```
video-sub-extrator/
├── main.py                  # CLI 入口
├── config.yaml              # 默认配置（可在此修改 backend、fps 等）
├── setup.sh                 # macOS 本地环境初始化脚本
├── Dockerfile               # DGX Spark GPU 镜像（NGC PyTorch 26.04）
├── docker-run.sh            # Docker 操作快捷脚本
├── docker-compose.yml       # Compose 配置（含 GPU 直通 + NAS 挂载）
├── requirements.txt         # macOS 依赖（PaddleOCR）
├── requirements-docker.txt  # Docker 依赖（EasyOCR）
├── src/
│   ├── frame_extractor.py   # 视频帧采样（OpenCV）
│   ├── image_preprocessor.py # 字幕区域裁剪 + 图像增强
│   ├── ocr_engine.py        # OCR 引擎封装（PaddleOCR / EasyOCR）
│   ├── text_processor.py    # 字幕去重合并
│   └── output_formatter.py  # Markdown / TXT 输出
├── videos/                  # 测试视频（不纳入版本控制）
└── output/                  # 识别结果输出目录
```

---

## OCR 引擎对比

| 特性 | PaddleOCR | EasyOCR |
|------|-----------|---------|
| 框架 | PaddlePaddle | PyTorch |
| macOS CPU | ✅ 支持 | ✅ 支持（需安装） |
| ARM64 GPU | ❌ 无官方支持 | ✅ 原生支持 |
| Docker 镜像 | ❌ 无 ARM64 镜像 | ✅ NGC 镜像 |
| 日文识别 | PP-OCRv5 日文模型 | EasyOCR 日文模型 |
| 配置参数 | `--backend paddleocr` | `--backend easyocr` |

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 未识别到字幕 | 增大 `--subtitle-region`（如 `bottom30`），降低 `--confidence`（如 `0.5`） |
| 字幕重复太多 | 降低 `--merge-threshold`（如 `0.7`） |
| 处理速度慢 | 降低 `--fps`（如 `0.5`），或在 DGX Spark 上用 Docker GPU 版 |
| macOS 运行报 easyocr 未找到 | 确认使用 `--backend paddleocr`，或运行 `pip install easyocr` |
| Docker GPU 不识别 | 运行 `bash docker-run.sh gpu-check` 诊断 |
| 首次运行很慢 | 正常，OCR 模型首次使用会自动下载并缓存 |
