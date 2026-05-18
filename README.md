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
# NAS 视频（生产模式，使用镜像内代码）
bash docker-run.sh run-nas /mnt/nas/Videos/xxx.mp4

# 本地视频（生产模式）
bash docker-run.sh run /path/to/video.mp4

# 避开角落台标/水印（指定精确字幕矩形）
bash docker-run.sh run-nas /mnt/nas/Videos/xxx.mp4 --subtitle-bbox "0.1,0.80,0.9,1.0"

# 进入容器调试
bash docker-run.sh shell
```

> **📂 输出目录说明**：`docker-run.sh` 默认将宿主机的 `$(pwd)/output/` 挂载为容器内的 `/app/output`，
> 识别结果会自动保存到该目录。若要指定其他宿主机路径，可直接修改脚本顶部的 `OUTPUT_DIR` 变量，
> 或临时覆盖（见下方「指定输出字幕目录」示例）。

#### 指定输出字幕目录示例

```bash
# 方式一：脚本内临时覆盖 OUTPUT_DIR（推荐，不修改脚本）
OUTPUT_DIR="/Volumes/NAS/subtitles/anime_A" bash docker-run.sh run-nas /mnt/nas/Videos/anime_A.mp4

# 方式二：NAS 视频 → 将字幕写到 NAS 的指定子目录
OUTPUT_DIR="/mnt/nas/subtitles/2025" bash docker-run.sh run-nas /mnt/nas/Videos/episode01.mp4

# 方式三：本地视频 → 字幕输出到桌面指定文件夹
OUTPUT_DIR="/Users/$(whoami)/Desktop/subs" bash docker-run.sh run /Users/$(whoami)/Movies/ep01.mp4

# 方式四：开发模式 + 自定义输出目录
OUTPUT_DIR="/tmp/test_output" bash docker-run.sh dev /path/to/video.mp4 --fps 0.5
```

> ⚠️ 请确保目标目录已存在，或脚本可写权限。`docker-run.sh` 会自动执行 `mkdir -p "${OUTPUT_DIR}"`。

### 4. 开发模式（修改代码后无需重新 build）

`dev` / `dev-nas` 命令会将宿主机的 `src/`、`main.py`、`config.yaml` 实时挂载进容器，代码改动即时生效：

```bash
# 本地视频（开发模式）
bash docker-run.sh dev test_japanese.mp4 --subtitle-bbox "0.1,0.80,0.9,1.0"

# NAS 视频（开发模式）
bash docker-run.sh dev-nas /mnt/nas/Videos/xxx.mp4 --subtitle-region bottom30
```

| 命令 | 代码来源 | 适用场景 |
|------|----------|----------|
| `run` / `run-nas` | 镜像内（baked-in） | 生产/稳定部署 |
| `dev` / `dev-nas` | 宿主机实时挂载 | 开发调试，改完代码直接跑 |

### 5. `run` vs `run-nas`：挂载机制详解

两个命令在功能上等价（均使用镜像内代码、GPU 推理），**核心区别在于容器如何访问视频文件**：

| 对比项 | `run <video_path>` | `run-nas <video_path>` |
|--------|-------------------|------------------------|
| **挂载对象** | 视频文件所在目录 → `/input/` | `/mnt/nas` 整目录 → `/mnt/nas/` |
| **容器内路径** | `/input/<文件名>` | 与宿主机路径完全一致 |
| **适用场景** | 视频在宿主机本地磁盘 | 视频在 NAS 网络存储，路径以 `/mnt/nas` 开头 |
| **典型调用** | `bash docker-run.sh run /data/video.mp4` | `bash docker-run.sh run-nas /mnt/nas/Videos/ep01.mp4` |

**底层 docker 命令等价展开：**

```bash
# run /data/video.mp4  →  等价于：
docker run --rm --gpus all \
  -v /data:/input:ro \
  -v $(pwd)/output:/app/output \
  video-sub-extrator:gpu \
  --input /input/video.mp4 --output /app/output --backend easyocr --use-gpu

# run-nas /mnt/nas/Videos/ep01.mp4  →  等价于：
docker run --rm --gpus all \
  -v /mnt/nas:/mnt/nas:ro \
  -v $(pwd)/output:/app/output \
  video-sub-extrator:gpu \
  --input /mnt/nas/Videos/ep01.mp4 --output /app/output --backend easyocr --use-gpu
```

> **💡 选用建议**：
> - 视频在 **本地 SSD / USB 外接盘** → 用 `run`
> - 视频在 **NAS（Synology、群晖等，已挂载到 `/mnt/nas`）** → 用 `run-nas`，容器内路径与宿主机完全一致，无需关心文件名拼接

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
| `--lang` | 自动 | OCR 语言代码（见下方「语言选项」） |
| `--backend` | `paddleocr` | OCR 引擎：`paddleocr` 或 `easyocr` |
| `--use-gpu` | 否 | 启用 GPU 加速 |
| `--no-timestamp` | — | 输出文件中不含时间戳 |

### 语言选项

| 语言 | EasyOCR `--lang` | PaddleOCR `--lang` |
|------|-----------------|--------------------|
| 日文（默认） | `ja` | `japan` |
| 简体中文 | `ch_sim` | `ch` |
| 繁体中文 | `ch_tra` | `chinese_cht` |
| 英文 | `en` | `en` |
| 中日混合 | `ch_sim,ja,en` | 不支持（仅取第一个） |

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

# 简体中文字幕（EasyOCR）
python main.py --input video.mp4 --lang ch_sim --backend easyocr

# 简体中文字幕（PaddleOCR / macOS）
python main.py --input video.mp4 --lang ch --backend paddleocr

# 中日混合字幕（EasyOCR，同时加载中文+日文+英文模型）
python main.py --input video.mp4 --lang ch_sim,ja,en --backend easyocr --use-gpu

# 指定输出字幕目录（将结果保存到指定路径而非默认的 output/）
python main.py --input video.mp4 --output /path/to/my_subtitles/

# 指定输出目录 + 格式（只输出 TXT，不生成 Markdown）
python main.py --input video.mp4 --output ~/Desktop/subs --format txt

# 完整示例：批量参数组合
python main.py --input /Volumes/NAS/Videos/ep01.mp4 \
               --output /Volumes/NAS/Subtitles/ep01 \
               --fps 1.5 \
               --subtitle-bbox "0.05,0.80,0.95,1.0" \
               --backend paddleocr \
               --format md,txt
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
