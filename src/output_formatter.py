"""
模块 5: 输出格式化器 (OutputFormatter)

将处理后的字幕列表输出为 Markdown 或 TXT 格式文件。
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List

from .text_processor import Subtitle


class OutputFormatter:
    """
    字幕输出格式化器。

    Args:
        output_dir: 输出目录路径
        include_timestamp: 是否在输出中包含时间戳
    """

    def __init__(
        self,
        output_dir: str = "output",
        include_timestamp: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_timestamp = include_timestamp

    def save(
        self,
        subtitles: List[Subtitle],
        source_filename: str,
        formats: List[str],
        video_info: dict = None,
    ) -> List[Path]:
        """
        将字幕列表保存为指定格式的文件。

        Args:
            subtitles: 处理后的字幕列表
            source_filename: 源视频文件名（不含路径）
            formats: 输出格式列表，支持 'md' 和 'txt'
            video_info: 视频元数据字典（可选，用于 Markdown 头部）

        Returns:
            保存的文件路径列表
        """
        stem = Path(source_filename).stem
        saved_files = []

        for fmt in formats:
            fmt = fmt.lower().strip()
            if fmt == "md" or fmt == "markdown":
                path = self._save_markdown(subtitles, stem, video_info)
                saved_files.append(path)
            elif fmt == "txt":
                path = self._save_txt(subtitles, stem, video_info)
                saved_files.append(path)
            else:
                print(f"⚠️  不支持的输出格式: {fmt}，跳过。")

        return saved_files

    def _save_markdown(
        self,
        subtitles: List[Subtitle],
        stem: str,
        video_info: dict = None,
    ) -> Path:
        """生成 Markdown 格式文件。"""
        output_path = self.output_dir / f"{stem}.md"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append(f"# 字幕提取结果 — {stem}")
        lines.append("")
        lines.append(f"> **提取时间**: {now}  ")

        if video_info:
            lines.append(f"> **源文件**: `{video_info.get('filename', stem)}`  ")
            if "duration_str" in video_info:
                lines.append(f"> **视频时长**: {video_info['duration_str']}  ")
            lines.append(f"> **采样帧率**: {video_info.get('sample_fps', 'N/A')} 帧/秒  ")

        lines.append(f"> **识别语言**: 日語  ")
        lines.append(f"> **字幕条数**: {len(subtitles)}  ")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 字幕内容")
        lines.append("")

        if not subtitles:
            lines.append("*未识别到任何字幕。*")
        elif self.include_timestamp:
            lines.append("| 开始时间 | 结束时间 | 字幕文本 |")
            lines.append("|----------|----------|----------|")
            for sub in subtitles:
                text = sub.text.replace("|", "｜")  # 转义 Markdown 表格中的竖线
                lines.append(f"| {sub.start_str} | {sub.end_str} | {text} |")
        else:
            for sub in subtitles:
                lines.append(f"- {sub.text}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 纯文本字幕")
        lines.append("")
        lines.append("```")
        for sub in subtitles:
            if self.include_timestamp:
                lines.append(f"[{sub.start_str}] {sub.text}")
            else:
                lines.append(sub.text)
        lines.append("```")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    def _save_txt(
        self,
        subtitles: List[Subtitle],
        stem: str,
        video_info: dict = None,
    ) -> Path:
        """生成 TXT 格式文件。"""
        output_path = self.output_dir / f"{stem}.txt"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []

        # 文件头信息
        lines.append(f"字幕提取结果 — {stem}")
        lines.append(f"提取时间: {now}")
        if video_info:
            lines.append(f"源文件: {video_info.get('filename', stem)}")
        lines.append(f"字幕条数: {len(subtitles)}")
        lines.append("=" * 50)
        lines.append("")

        if not subtitles:
            lines.append("未识别到任何字幕。")
        else:
            for sub in subtitles:
                if self.include_timestamp:
                    lines.append(f"[{sub.start_str} --> {sub.end_str}]")
                    lines.append(sub.text)
                    lines.append("")
                else:
                    lines.append(sub.text)

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    @staticmethod
    def preview(subtitles: List[Subtitle], max_items: int = 5) -> str:
        """
        生成字幕预览字符串（用于控制台输出）。

        Args:
            subtitles: 字幕列表
            max_items: 最多显示条数

        Returns:
            预览字符串
        """
        if not subtitles:
            return "（未识别到字幕）"

        preview_lines = []
        for i, sub in enumerate(subtitles[:max_items]):
            preview_lines.append(f"  [{sub.start_str}] {sub.text}")

        if len(subtitles) > max_items:
            preview_lines.append(f"  ... 共 {len(subtitles)} 条字幕")

        return "\n".join(preview_lines)
