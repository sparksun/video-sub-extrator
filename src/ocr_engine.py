"""
模块 3: OCR 识别引擎 (OCREngine)

封装 PaddleOCR v3，对预处理后的字幕图像进行日文文字识别。
返回带时间戳、文本和置信度的识别结果。

PaddleOCR v3 API 变化（与 v2 对比）：
  - use_gpu=True  →  device='gpu' / device='cpu'
  - use_angle_cls →  use_textline_orientation
  - show_log      →  已移除
  - ocr() 返回值  →  生成器，每个元素是 OCRResult 对象，需用 .boxes/.rec_texts/.rec_scores 访问
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
    PaddleOCR v3 日文识别引擎封装。

    Args:
        lang: OCR 语言代码。日文使用 'japan'。
        use_gpu: 是否使用 GPU 加速。v3 中对应 device='gpu'。
        use_textline_orientation: 是否启用方向分类（支持竖排文字）。
        confidence_threshold: 置信度阈值，低于此值的结果将被丢弃。
    """

    def __init__(
        self,
        lang: str = "japan",
        use_gpu: bool = False,
        use_textline_orientation: bool = True,
        confidence_threshold: float = 0.7,
    ):
        self.lang = lang
        self.device = "gpu" if use_gpu else "cpu"
        self.use_textline_orientation = use_textline_orientation
        self.confidence_threshold = confidence_threshold
        self._ocr = None  # 延迟初始化（首次调用时加载模型）

    def _get_ocr(self):
        """延迟初始化 PaddleOCR（首次调用时下载并加载模型）。"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                raise ImportError(
                    "PaddleOCR 未安装。请运行: pip install paddleocr paddlepaddle"
                )
            # PaddleOCR v3 新 API
            self._ocr = PaddleOCR(
                lang=self.lang,
                device=self.device,
                use_textline_orientation=self.use_textline_orientation,
                text_rec_score_thresh=self.confidence_threshold,
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

        # PaddleOCR v3: ocr() 返回生成器，每次 predict 一张图
        # 结果是 OCRResult 对象，包含 .rec_texts, .rec_scores, .boxes 属性
        results = list(ocr.ocr(image))

        if not results:
            return None

        lines = []
        confidences = []

        for result in results:
            # result 是单张图的 OCRResult 对象
            if result is None:
                continue

            # v3 API: rec_texts / rec_scores
            rec_texts = getattr(result, "rec_texts", None)
            rec_scores = getattr(result, "rec_scores", None)

            # 兼容旧式列表格式 [[bbox, (text, score)], ...]
            if rec_texts is None:
                # 尝试旧格式（如果有混合安装）
                if isinstance(result, list):
                    for line in result:
                        if line is None:
                            continue
                        text, score = line[1]
                        text = text.strip()
                        if score >= self.confidence_threshold and text:
                            lines.append(text)
                            confidences.append(score)
                continue

            # 正常 v3 格式
            if rec_texts is None or rec_scores is None:
                continue

            for text, score in zip(rec_texts, rec_scores):
                text = str(text).strip() if text else ""
                score = float(score) if score is not None else 0.0
                if score >= self.confidence_threshold and text:
                    lines.append(text)
                    confidences.append(score)

        if not lines:
            return None

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
