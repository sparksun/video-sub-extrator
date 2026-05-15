"""
模块 4: 文本后处理器 (TextPostProcessor)

对 OCR 识别结果进行去重、合并和规范化处理：
  1. 合并连续相同/相似的字幕（同一条字幕在多帧中重复出现）
  2. 为每条字幕计算开始和结束时间戳
  3. 过滤过短或无效的文本
"""

import difflib
from dataclasses import dataclass
from typing import List

from .ocr_engine import OCRResult


@dataclass
class Subtitle:
    """处理后的字幕条目。"""
    start_ms: int     # 字幕开始时间（毫秒）
    end_ms: int       # 字幕结束时间（毫秒）
    text: str         # 字幕文本

    @property
    def start_str(self) -> str:
        """开始时间戳字符串 HH:MM:SS。"""
        return _ms_to_timestamp(self.start_ms)

    @property
    def end_str(self) -> str:
        """结束时间戳字符串 HH:MM:SS。"""
        return _ms_to_timestamp(self.end_ms)

    @property
    def duration_s(self) -> float:
        """字幕持续时长（秒）。"""
        return (self.end_ms - self.start_ms) / 1000


def _ms_to_timestamp(ms: int) -> str:
    """将毫秒转换为 HH:MM:SS 格式。"""
    s = int(ms / 1000)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _text_similarity(a: str, b: str) -> float:
    """
    计算两段文本的相似度（0.0 ~ 1.0）。
    使用 SequenceMatcher，对中日文效果良好。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


class TextPostProcessor:
    """
    OCR 结果后处理器。

    Args:
        merge_threshold: 相邻帧文本相似度阈值，超过此值则合并为同一字幕。
        min_text_length: 最短字幕长度，过短的字幕将被丢弃。
    """

    def __init__(
        self,
        merge_threshold: float = 0.85,
        min_text_length: int = 1,
    ):
        self.merge_threshold = merge_threshold
        self.min_text_length = min_text_length

    def process(self, ocr_results: List[OCRResult]) -> List[Subtitle]:
        """
        将 OCR 识别结果列表处理为去重后的字幕列表。

        Args:
            ocr_results: 按时间戳排序的 OCR 识别结果列表

        Returns:
            处理后的 Subtitle 列表
        """
        if not ocr_results:
            return []

        # 过滤过短文本
        filtered = [r for r in ocr_results if len(r.text.strip()) >= self.min_text_length]
        if not filtered:
            return []

        subtitles = []
        current = filtered[0]
        current_end_ms = current.timestamp_ms

        for i in range(1, len(filtered)):
            next_result = filtered[i]
            similarity = _text_similarity(current.text, next_result.text)

            if similarity >= self.merge_threshold:
                # 相似度高 → 合并，延长当前字幕的结束时间
                current_end_ms = next_result.timestamp_ms
            else:
                # 相似度低 → 保存当前字幕，开始新字幕
                subtitle = Subtitle(
                    start_ms=current.timestamp_ms,
                    end_ms=current_end_ms,
                    text=self._normalize_text(current.text),
                )
                subtitles.append(subtitle)
                current = next_result
                current_end_ms = next_result.timestamp_ms

        # 处理最后一条
        subtitle = Subtitle(
            start_ms=current.timestamp_ms,
            end_ms=current_end_ms,
            text=self._normalize_text(current.text),
        )
        subtitles.append(subtitle)

        return subtitles

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        规范化文本：去除多余空格，统一标点等。
        """
        # 去除首尾空白
        text = text.strip()
        # 合并多个空格为单个
        import re
        text = re.sub(r"\s+", " ", text)
        return text
