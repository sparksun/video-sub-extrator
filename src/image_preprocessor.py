"""
模块 2: 图像预处理器 (ImagePreprocessor)

对原始视频帧进行裁剪和缩放，送入 OCR 引擎。

设计原则：
  - PaddleOCR v3 / EasyOCR 内部均有完整的图像增强流程，
    外部只需提供裁剪后的彩色图像即可，过度预处理反而降低识别率。
  - 仅执行：区域裁剪 → 等比缩放（可选）
"""

import cv2
import numpy as np
from typing import Tuple


# 字幕区域预设（画面纵向起止比例）
SUBTITLE_REGIONS = {
    "bottom10": (0.90, 1.0),
    "bottom15": (0.85, 1.0),
    "bottom20": (0.80, 1.0),   # 默认
    "bottom25": (0.75, 1.0),
    "bottom30": (0.70, 1.0),
    "bottom50": (0.50, 1.0),
    "full":     (0.0,  1.0),
}


class ImagePreprocessor:
    """
    视频帧预处理：裁剪字幕区域 + 可选缩放。

    Args:
        subtitle_region: 字幕区域预设，见 SUBTITLE_REGIONS
        scale_factor: 图像放大倍数（1.0 = 不缩放，建议 1.5-2.0）
    """

    def __init__(
        self,
        subtitle_region: str = "bottom20",
        scale_factor: float = 1.5,
    ):
        if subtitle_region not in SUBTITLE_REGIONS:
            raise ValueError(
                f"无效的字幕区域: {subtitle_region}。"
                f"可用选项: {list(SUBTITLE_REGIONS.keys())}"
            )
        self.subtitle_region = subtitle_region
        self.scale_factor = scale_factor
        self._region_ratio = SUBTITLE_REGIONS[subtitle_region]

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        裁剪字幕区域并缩放。

        Args:
            frame: BGR 格式的原始视频帧

        Returns:
            (cropped_original, processed_for_ocr)
        """
        h, w = frame.shape[:2]

        # 1. 裁剪字幕区域
        y_start = int(h * self._region_ratio[0])
        y_end   = int(h * self._region_ratio[1])
        cropped = frame[y_start:y_end, 0:w]

        # 防止裁剪出空图（如 full 区域时 y_start=0）
        if cropped.shape[0] == 0:
            cropped = frame.copy()

        # 2. 缩放（提升小字体识别率）
        if self.scale_factor != 1.0:
            new_w = int(cropped.shape[1] * self.scale_factor)
            new_h = int(cropped.shape[0] * self.scale_factor)
            processed = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            processed = cropped.copy()

        return cropped, processed

    def process_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        """
        便捷方法：直接返回适合 OCR 输入的图像（BGR 彩色）。
        """
        _, processed = self.process(frame)
        return processed
