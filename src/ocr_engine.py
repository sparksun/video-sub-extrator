"""
模块 3: OCR 识别引擎 (OCREngine)

支持两种 OCR 后端，根据运行环境自动选择或手动指定：
  - easyocr  (默认): 基于 PyTorch，支持 ARM64 + NVIDIA GPU，推荐用于 DGX Spark
  - paddleocr:        基于 PaddlePaddle，仅 x86_64 有 GPU 支持，适合 macOS CPU 推理

用法:
  OCREngine(backend='easyocr', use_gpu=True)   # DGX Spark GPU
  OCREngine(backend='paddleocr', use_gpu=False) # macOS CPU
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OCRResult:
    """单帧 OCR 识别结果。"""
    timestamp_ms: int
    text: str
    confidence: float
    raw_lines: List[str] = field(default_factory=list)


class OCREngine:
    """
    多后端 OCR 引擎封装，统一接口。

    Args:
        backend:    'easyocr' (推荐 DGX Spark) 或 'paddleocr' (推荐 macOS)
        lang:       语言代码。easyocr 用 'ja'，paddleocr 用 'japan'
        use_gpu:    是否使用 GPU 加速
        confidence_threshold: 置信度过滤阈值
    """

    BACKEND_EASYOCR = "easyocr"
    BACKEND_PADDLE = "paddleocr"

    def __init__(
        self,
        backend: str = "easyocr",
        lang: str = None,
        use_gpu: bool = False,
        confidence_threshold: float = 0.7,
    ):
        self.backend = backend.lower()
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self._engine = None

        # 语言代码按后端规范自动映射
        if lang is None:
            self.lang = "ja" if self.backend == self.BACKEND_EASYOCR else "japan"
        else:
            self.lang = lang

    def _get_engine(self):
        """延迟初始化 OCR 引擎（首次调用时加载模型）。"""
        if self._engine is not None:
            return self._engine

        if self.backend == self.BACKEND_EASYOCR:
            self._engine = self._init_easyocr()
        elif self.backend == self.BACKEND_PADDLE:
            self._engine = self._init_paddleocr()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'easyocr' or 'paddleocr'.")

        return self._engine

    def _init_easyocr(self):
        """初始化 EasyOCR（支持 ARM64 + NVIDIA GPU）。"""
        try:
            import easyocr
        except ImportError:
            raise ImportError("EasyOCR 未安装。请运行: pip install easyocr")

        print(f"[OCR] Loading EasyOCR (lang={self.lang}, gpu={self.use_gpu})...")
        reader = easyocr.Reader(
            [self.lang],
            gpu=self.use_gpu,
            verbose=False,
        )
        return ("easyocr", reader)

    def _init_paddleocr(self):
        """初始化 PaddleOCR v3（支持 x86_64，macOS CPU 推理）。"""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("PaddleOCR 未安装。请运行: pip install paddleocr paddlepaddle")

        print(f"[OCR] Loading PaddleOCR (lang={self.lang}, device={'gpu' if self.use_gpu else 'cpu'})...")
        ocr = PaddleOCR(
            lang=self.lang,
            device="gpu" if self.use_gpu else "cpu",
            use_textline_orientation=True,
            text_rec_score_thresh=self.confidence_threshold,
        )
        return ("paddleocr", ocr)

    def recognize(self, image: np.ndarray, timestamp_ms: int) -> Optional[OCRResult]:
        """
        对单帧图像执行 OCR 识别。

        Args:
            image: BGR 格式的预处理图像
            timestamp_ms: 帧时间戳（毫秒）

        Returns:
            OCRResult 或 None
        """
        backend_name, engine = self._get_engine()

        if backend_name == "easyocr":
            return self._recognize_easyocr(engine, image, timestamp_ms)
        else:
            return self._recognize_paddleocr(engine, image, timestamp_ms)

    def _recognize_easyocr(self, reader, image: np.ndarray, timestamp_ms: int) -> Optional[OCRResult]:
        """EasyOCR 识别逻辑。"""
        # EasyOCR 接受 BGR (OpenCV) 或 RGB，这里直接传入
        results = reader.readtext(image)
        # results: [(bbox, text, confidence), ...]

        lines = []
        confidences = []
        for (_, text, conf) in results:
            text = text.strip()
            if conf >= self.confidence_threshold and text:
                lines.append(text)
                confidences.append(conf)

        if not lines:
            return None

        return OCRResult(
            timestamp_ms=timestamp_ms,
            text=" ".join(lines),
            confidence=sum(confidences) / len(confidences),
            raw_lines=lines,
        )

    def _recognize_paddleocr(self, ocr, image: np.ndarray, timestamp_ms: int) -> Optional[OCRResult]:
        """PaddleOCR v3 识别逻辑。"""
        results = list(ocr.ocr(image))
        if not results:
            return None

        lines = []
        confidences = []

        for result in results:
            if result is None:
                continue
            rec_texts = getattr(result, "rec_texts", None)
            rec_scores = getattr(result, "rec_scores", None)

            if rec_texts is None:
                # 兼容旧格式
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

            for text, score in zip(rec_texts, rec_scores):
                text = str(text).strip() if text else ""
                score = float(score) if score is not None else 0.0
                if score >= self.confidence_threshold and text:
                    lines.append(text)
                    confidences.append(score)

        if not lines:
            return None

        return OCRResult(
            timestamp_ms=timestamp_ms,
            text=" ".join(lines),
            confidence=sum(confidences) / len(confidences),
            raw_lines=lines,
        )

    def recognize_batch(self, frames: List[tuple]) -> List[OCRResult]:
        results = []
        for timestamp_ms, image in frames:
            result = self.recognize(image, timestamp_ms)
            if result is not None:
                results.append(result)
        return results
