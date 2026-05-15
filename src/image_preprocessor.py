"""
模块 2: 图像预处理器 (ImagePreprocessor)

对原始视频帧进行裁剪和增强，以提升 OCR 识别准确率。
核心策略：
  1. 裁剪字幕区域（通常在画面底部）
  2. 放大图像（小字识别率提升）
  3. 灰度化 + 自适应二值化
  4. 去噪 + 膨胀（加粗文字）
"""

import cv2
import numpy as np
from typing import Tuple


# 字幕区域预设（画面底部百分比）
SUBTITLE_REGIONS = {
    "bottom10": (0.90, 1.0),   # 底部10%
    "bottom15": (0.85, 1.0),   # 底部15%
    "bottom20": (0.80, 1.0),   # 底部20%（默认）
    "bottom25": (0.75, 1.0),   # 底部25%
    "bottom30": (0.70, 1.0),   # 底部30%
    "bottom50": (0.50, 1.0),   # 底部50%
    "full": (0.0, 1.0),        # 全帧
}


class ImagePreprocessor:
    """
    对视频帧进行预处理，提取字幕区域并增强图像质量。

    Args:
        subtitle_region: 字幕区域预设名称，见 SUBTITLE_REGIONS
        scale_factor: 图像放大倍数（建议2.0-3.0）
    """

    def __init__(
        self,
        subtitle_region: str = "bottom20",
        scale_factor: float = 2.0,
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
        对单帧图像进行预处理。

        Args:
            frame: BGR 格式的原始视频帧

        Returns:
            Tuple[cropped_original, processed]:
                - cropped_original: 裁剪后的原始帧（供调试）
                - processed: 预处理后的图像（供 OCR）
        """
        h, w = frame.shape[:2]

        # 1. 裁剪字幕区域
        y_start = int(h * self._region_ratio[0])
        y_end = int(h * self._region_ratio[1])
        cropped = frame[y_start:y_end, 0:w]

        # 2. 放大图像（提升小字识别率）
        if self.scale_factor != 1.0:
            new_w = int(cropped.shape[1] * self.scale_factor)
            new_h = int(cropped.shape[0] * self.scale_factor)
            scaled = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            scaled = cropped.copy()

        # 3. 转灰度
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # 4. 去噪（保边缘滤波）
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 5. 自适应二值化（处理复杂背景）
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        # 6. 反转（确保文字为黑色，背景为白色，有利于OCR）
        # 检测文字颜色：如果白色像素更多，文字可能是黑色，无需反转
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        if white_pixels < black_pixels:
            binary = cv2.bitwise_not(binary)

        # 7. 膨胀操作（加粗文字笔划，提升识别率）
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # 返回 BGR 格式（PaddleOCR 接受 BGR 或 RGB）
        processed = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        return cropped, processed

    def process_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        """
        便捷方法：直接返回适合 OCR 输入的预处理图像。

        Args:
            frame: BGR 格式的原始视频帧

        Returns:
            预处理后的 BGR 图像
        """
        _, processed = self.process(frame)
        return processed
