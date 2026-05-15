#!/usr/bin/env python3
"""
main.py — 日文视频硬字幕提取工具 CLI 入口

用法示例：
    python main.py --input videos/test_japanese.mp4
    python main.py --input videos/test.mp4 --format md,txt --fps 2
    python main.py --input videos/test.mp4 --subtitle-region bottom30 --no-timestamp
"""

import sys
import time
from pathlib import Path

import click
import yaml
from tqdm import tqdm


# ─── 配置加载 ─────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """从 YAML 文件加载默认配置。"""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ─── CLI 定义 ─────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="输入视频文件路径（必填）",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(),
    help="输出目录路径（默认: output/）",
)
@click.option(
    "--format", "-f",
    "output_format",
    default=None,
    help="输出格式，逗号分隔（默认: md,txt）。可选: md / txt",
)
@click.option(
    "--fps",
    default=None,
    type=float,
    help="视频采样帧率，帧/秒（默认: 1.0）",
)
@click.option(
    "--subtitle-region",
    default=None,
    type=click.Choice(
        ["bottom10", "bottom15", "bottom20", "bottom25", "bottom30", "bottom50", "full"],
        case_sensitive=False,
    ),
    help="字幕区域预设（默认: bottom20）",
)
@click.option(
    "--confidence",
    default=None,
    type=float,
    help="OCR 置信度阈值，0.0-1.0（默认: 0.7）",
)
@click.option(
    "--merge-threshold",
    default=None,
    type=float,
    help="相邻字幕合并相似度阈值，0.0-1.0（默认: 0.85）",
)
@click.option(
    "--lang",
    default=None,
    type=str,
    help="OCR 语言代码（默认: japan）",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=None,
    help="是否使用 GPU 加速（默认: 否）",
)
@click.option(
    "--backend",
    default=None,
    type=click.Choice(["easyocr", "paddleocr"], case_sensitive=False),
    help="OCR 引擎后端（默认: easyocr）。DGX Spark 用 easyocr，macOS 用 paddleocr",
)
@click.option(
    "--include-timestamp/--no-timestamp",
    default=True,
    help="输出文件中是否包含时间戳（默认: 是）",
)
@click.option(
    "--config",
    default="config.yaml",
    type=click.Path(),
    help="配置文件路径（默认: config.yaml）",
)
def main(
    input,
    output,
    output_format,
    fps,
    subtitle_region,
    confidence,
    merge_threshold,
    lang,
    use_gpu,
    backend,
    include_timestamp,
    config,
):
    """
    🎬 日文视频硬字幕提取工具

    从日文 MP4 视频中自动识别硬字幕（烧制字幕），输出为 Markdown 或 TXT 文件。

    \b
    技术栈: PaddleOCR (日文模型) + OpenCV + FFmpeg
    支持:   水平字幕 / 竖排字幕 / 复杂背景
    """
    # 1. 加载配置文件
    cfg = load_config(config)
    ocr_cfg = cfg.get("ocr", {})
    ext_cfg = cfg.get("extraction", {})
    post_cfg = cfg.get("postprocess", {})
    out_cfg = cfg.get("output", {})

    # 2. CLI 参数覆盖配置文件（CLI 优先级更高）
    _fps          = fps             or ext_cfg.get("fps", 1.0)
    _region       = subtitle_region or ext_cfg.get("subtitle_region", "bottom20")
    _scale        = ext_cfg.get("scale_factor", 2.0)
    _lang         = lang            or None  # None 让 OCREngine 按 backend 自动选择
    _use_gpu      = use_gpu         if use_gpu is not None else ocr_cfg.get("use_gpu", False)
    _backend      = backend         or ocr_cfg.get("backend", "easyocr")
    _confidence   = confidence      or ocr_cfg.get("confidence_threshold", 0.7)
    _merge_thr    = merge_threshold or post_cfg.get("merge_threshold", 0.85)
    _min_len      = post_cfg.get("min_text_length", 1)
    _output_dir   = output          or out_cfg.get("output_dir", "output")
    _formats_raw  = output_format   or ",".join(out_cfg.get("format", ["md", "txt"]))
    _formats      = [f.strip() for f in _formats_raw.split(",")]

    # 3. 打印运行参数
    click.echo("")
    click.echo("🎬 日文视频硬字幕提取工具")
    click.echo("─" * 50)
    click.echo(f"  输入视频: {input}")
    click.echo(f"  输出目录: {_output_dir}")
    click.echo(f"  输出格式: {', '.join(_formats)}")
    click.echo(f"  采样帧率: {_fps} 帧/秒")
    click.echo(f"  字幕区域: {_region}")
    click.echo(f"  OCR 引擎: {_backend}")
    click.echo(f"  GPU 加速: {'是' if _use_gpu else '否'}")
    click.echo(f"  置信度阈値: {_confidence}")
    click.echo("─" * 50)
    click.echo("")

    # ─── 导入各模块 ────────────────────────────────────────────────────────────
    from src.frame_extractor import FrameExtractor
    from src.image_preprocessor import ImagePreprocessor
    from src.ocr_engine import OCREngine
    from src.text_processor import TextPostProcessor
    from src.output_formatter import OutputFormatter

    # ─── 初始化各模块 ──────────────────────────────────────────────────────────
    extractor    = FrameExtractor(video_path=input, fps=_fps)
    preprocessor = ImagePreprocessor(subtitle_region=_region, scale_factor=_scale)
    ocr_engine   = OCREngine(
        backend=_backend,
        lang=_lang,
        use_gpu=_use_gpu,
        confidence_threshold=_confidence,
    )
    processor   = TextPostProcessor(merge_threshold=_merge_thr, min_text_length=_min_len)
    formatter   = OutputFormatter(output_dir=_output_dir, include_timestamp=include_timestamp)

    # ─── 获取视频信息 ──────────────────────────────────────────────────────────
    video_info = extractor.video_info
    total_samples = extractor.get_total_sample_count()
    click.echo(f"📹 视频时长: {video_info['duration_str']}")
    click.echo(f"📊 预计采样帧数: ~{total_samples} 帧")
    click.echo("")

    # ─── Pipeline 执行 ────────────────────────────────────────────────────────
    click.echo("⚙️  Step 1/3: 帧提取 + 图像预处理 + OCR 识别中...")
    start_time = time.time()
    ocr_results = []

    with tqdm(total=total_samples, unit="帧", desc="识别进度") as pbar:
        for timestamp_ms, frame in extractor.extract():
            # 预处理
            processed = preprocessor.process_for_ocr(frame)
            # OCR 识别
            result = ocr_engine.recognize(processed, timestamp_ms)
            if result is not None:
                ocr_results.append(result)
            pbar.update(1)

    elapsed = time.time() - start_time
    click.echo(f"   ✅ 完成！识别到 {len(ocr_results)} 帧有效文本，耗时 {elapsed:.1f}s")
    click.echo("")

    # ─── 后处理 ───────────────────────────────────────────────────────────────
    click.echo("⚙️  Step 2/3: 去重合并字幕...")
    subtitles = processor.process(ocr_results)
    click.echo(f"   ✅ 合并后共 {len(subtitles)} 条字幕")
    click.echo("")

    # ─── 预览 ─────────────────────────────────────────────────────────────────
    if subtitles:
        click.echo("📋 字幕预览（前5条）:")
        click.echo(formatter.preview(subtitles, max_items=5))
        click.echo("")

    # ─── 输出保存 ─────────────────────────────────────────────────────────────
    click.echo("⚙️  Step 3/3: 保存输出文件...")
    saved_files = formatter.save(
        subtitles=subtitles,
        source_filename=Path(input).name,
        formats=_formats,
        video_info=video_info,
    )

    click.echo("")
    click.echo("✅ 全部完成！输出文件：")
    for f in saved_files:
        click.echo(f"   📄 {f}")

    if not subtitles:
        click.echo("")
        click.echo("⚠️  未识别到任何字幕，请尝试以下调整：")
        click.echo("   - 增大 --subtitle-region（如 bottom30）")
        click.echo("   - 降低 --confidence（如 0.5）")
        click.echo("   - 增大 --fps（如 2.0）")

    click.echo("")


if __name__ == "__main__":
    main()
