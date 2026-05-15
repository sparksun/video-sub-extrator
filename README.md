# 日文视频硬字幕提取工具

从日文 MP4 视频文件中自动提取硬字幕（烧制字幕），使用 PaddleOCR 进行日文 OCR 识别，输出为 Markdown 或 TXT 格式文件。

## 功能特点

- ✅ 支持日文硬字幕（烧制字幕）提取
- ✅ 基于 PaddleOCR，日文识别精度高
- ✅ 支持水平和竖排文字
- ✅ 智能去重合并连续相同字幕
- ✅ 输出 Markdown 表格 + TXT 两种格式
- ✅ 带时间戳标注
- ✅ 可配置字幕区域、采样率、置信度

## 安装

### 1. 创建项目本地 venv 并安装所有依赖

```bash
bash setup.sh
```

该脚本会自动完成：
- 在项目目录下创建 `venv/`
- 升级 pip
- 安装 opencv、click、tqdm 等基础依赖
- 安装 PaddlePaddle (CPU 版)
- 安装 PaddleOCR

> 如有 NVIDIA GPU，可手动将 `paddlepaddle` 替换为 `paddlepaddle-gpu`

### 2. 激活 venv

```bash
source venv/bin/activate
```

### 3. 验证安装

```bash
python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"  
```

## 快速开始

```bash
# 基本用法（输出 md + txt 到 output/ 目录）
python main.py --input videos/test_japanese.mp4

# 指定输出格式
python main.py --input videos/sample.mp4 --format md

# 提高采样率（更精细但更慢）
python main.py --input videos/sample.mp4 --fps 2

# 扩大字幕识别区域
python main.py --input videos/sample.mp4 --subtitle-region bottom30
```

## 完整参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input, -i` | 必填 | 输入视频文件路径 |
| `--output, -o` | `output/` | 输出目录 |
| `--format, -f` | `md,txt` | 输出格式（逗号分隔） |
| `--fps` | `1.0` | 视频采样帧率（帧/秒） |
| `--subtitle-region` | `bottom20` | 字幕区域：bottom10/15/20/25/30/50/full |
| `--confidence` | `0.7` | OCR 置信度阈值（0.0-1.0） |
| `--merge-threshold` | `0.85` | 相邻字幕合并相似度阈值 |
| `--lang` | `japan` | OCR 语言代码 |
| `--use-gpu / --no-gpu` | `--no-gpu` | 是否使用 GPU 加速 |
| `--include-timestamp / --no-timestamp` | `--include-timestamp` | 输出是否包含时间戳 |

## 输出示例

### Markdown 格式（output/sample.md）

```markdown
# 字幕提取结果 — sample

| 开始时间 | 结束时间 | 字幕文本 |
|----------|----------|----------|
| 00:00:03 | 00:00:07 | こんにちは、世界！ |
| 00:00:10 | 00:00:15 | 日本語字幕テスト |
```

### TXT 格式（output/sample.txt）

```
[00:00:03 --> 00:00:07]
こんにちは、世界！

[00:00:10 --> 00:00:15]
日本語字幕テスト
```

## 项目结构

```
video-sub-extrator/
├── videos/                    # 输入视频目录
├── output/                    # 输出文件目录
├── src/
│   ├── frame_extractor.py     # 模块1: 帧提取（OpenCV）
│   ├── image_preprocessor.py  # 模块2: 图像预处理（裁剪+增强）
│   ├── ocr_engine.py          # 模块3: OCR识别（PaddleOCR）
│   ├── text_processor.py      # 模块4: 去重后处理
│   └── output_formatter.py    # 模块5: 格式化输出
├── main.py                    # CLI 入口
├── config.yaml                # 默认配置
└── requirements.txt           # 依赖清单
```

## 调优建议

| 问题 | 解决方案 |
|------|----------|
| 未识别到字幕 | 增大 `--subtitle-region`（如 `bottom30`），降低 `--confidence`（如 `0.5`） |
| 字幕重复太多 | 降低 `--merge-threshold`（如 `0.7`） |
| 处理速度太慢 | 降低 `--fps`（如 `0.5`），或启用 `--use-gpu` |
| 识别精度不够 | 增大 `--fps` 提高采样率，或手动指定精确的字幕区域 |
