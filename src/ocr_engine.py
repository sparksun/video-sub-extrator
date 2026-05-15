"""
模块 3: OCR 识别引擎 (OCREngine)

封装 PaddleOCR，对预处理后的字幕图像进行日文文字识别。
返回带时间戳、文本和置信度的识别结果。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OCRResult:
    """单帧 OCR 识别结果。"""
    timestamp_ms: int          # 帧时间戳（毫秒）
    text: str                  # 识别出的文本（多行合并）
    confidence: float          # 平均置信度
    raw_lines: List[str] = field(default_factory=list)  # 原始识别行列表


class OCREngine:
    """
    PaddleOCR 日文识别引擎封装。

    Args:
        lang: OCR 语言代码。日文使用 'japan'。
        use_gpu: 是否使用 GPU 加速。
        use_angle_cls: 是否启用方向分类（支持竖排文字）。
        confidence_threshold: 置信度阈值，低于此值的结果将被丢弃。
    """

    def __init__(
        self,
        lang: str = "japan",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        confidence_threshold: float = 0.7,
    ):
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.confidence_threshold = confidence_threshold
        self._ocr = None  # 延迟初始化（避免导入时加载模型）

    def _get_ocr(self):
        """延迟初始化 PaddleOCR（首次调用时加载模型）。"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                raise ImportError(
                    "PaddleOCR 未安装。请运行: pip install paddleocr paddlepaddle"
                )
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
        return self._ocr

    def recognize(
        self,
        image: np.ndarray,
        timestamp_ms: int,
    ) -> Optional[OCRResult]:
        """
        对单帧图像执行 OCR 识别。

        Args:
            image: BGR 格式的预处理图像
            timestamp_ms: 帧时间戳（毫秒）

        Returns:
            OCRResult 或 None（如果未识别出任何文本）
        """
        ocr = self._get_ocr()

        # 执行识别
        results = ocr.ocr(image, cls=self.use_angle_cls)

        # 解析结果：results 格式为 [[[bbox, (text, confidence)], ...]]
        if not results or results[0] is None:
            return None

        lines = []
        confidences = []

        for line in results[0]:
            if line is None:
                continue
            # line 格式：[bbox_points, (text, confidence)]
            text, confidence = line[1]
            text = text.strip()

            # 过滤低置信度和空文本
            if confidence >= self.confidence_threshold and text:
                lines.append(text)
                confidences.append(confidence)

        if not lines:
            return None

        # 多行文本合并（按位置从上到下已排序）
        combined_text = " ".join(lines)
        avg_confidence = sum(confidences) / len(confidences)

        return OCRResult(
            timestamp_ms=timestamp_ms,
            text=combined_text,
            confidence=avg_confidence,
            raw_lines=lines,
        )

    def recognize_batch(
        self,
        frames: List[tuple],  # List[(timestamp_ms, np.ndarray)]
    ) -> List[OCRResult]:
        """
        批量识别多帧。

        Args:
            frames: [(timestamp_ms, image), ...] 列表

        Returns:
            识别出文本的 OCRResult 列表（空帧不包含）
        """
        results = []
        for timestamp_ms, image in frames:
            result = self.recognize(image, timestamp_ms)
            if result is not None:
                results.append(result)
        return results
