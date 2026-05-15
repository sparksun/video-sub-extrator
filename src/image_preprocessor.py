"""
模块 2: 图像预处理器 (ImagePreprocessor)

对原始视频帧进行裁剪和缩放，送入 OCR 引擎。

字幕区域指定方式（二选一）：
  1. subtitle_region 预设名称：bottom20 / bottom30 / full 等（垂直范围）
  2. subtitle_bbox 自定义矩形：(x1, y1, x2, y2) 均为 0.0–1.0 的相对坐标
     例如 (0.1, 0.75, 0.9, 1.0) = 水平10%-90%、垂直75%-100%（避开角落台标）

设计原则：PaddleOCR v3 / EasyOCR 内部有完整图像增强流程，
          外部只需提供裁剪后的彩色原图，过度预处理反而降低识别率。
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# 垂直区域预设（y_start, y_end 相对比例）
SUBTITLE_REGIONS = {
    "bottom10": (0.90, 1.0),
    "bottom15": (0.85, 1.0),
    "bottom20": (0.80, 1.0),   # 默认
    "bottom25": (0.75, 1.0),
    "bottom30": (0.70, 1.0),
    "bottom50": (0.50, 1.0),
    "full":     (0.0,  1.0),
}

# 使用预设时默认的水平边距（裁掉左右各 5%，避开角落台标/水印）
DEFAULT_HORIZONTAL_MARGIN = 0.05


class ImagePreprocessor:
    """
    视频帧预处理：裁剪字幕区域 + 可选缩放。

    Args:
        subtitle_region: 预设名称（见 SUBTITLE_REGIONS），与 subtitle_bbox 互斥
        subtitle_bbox:   自定义矩形 (x1, y1, x2, y2)，0.0–1.0 相对坐标
                         例如 (0.1, 0.75, 0.9, 1.0)。若提供则忽略 subtitle_region
        scale_factor:    图像放大倍数（1.0 = 不缩放）
    """

    def __init__(
        self,
        subtitle_region: str = "bottom20",
        subtitle_bbox: Optional[Tuple[float, float, float, float]] = None,
        scale_factor: float = 1.5,
    ):
        self.scale_factor = scale_factor

        if subtitle_bbox is not None:
            # 自定义矩形优先
            x1, y1, x2, y2 = subtitle_bbox
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                raise ValueError(
                    f"subtitle_bbox {subtitle_bbox} 无效，各值需满足 0.0≤x1<x2≤1.0, 0.0≤y1<y2≤1.0"
                )
            self._bbox = subtitle_bbox
            self.subtitle_region = f"custom({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})"
        else:
            if subtitle_region not in SUBTITLE_REGIONS:
                raise ValueError(
                    f"无效的字幕区域: {subtitle_region}。"
                    f"可用选项: {list(SUBTITLE_REGIONS.keys())}"
                )
            y_start, y_end = SUBTITLE_REGIONS[subtitle_region]
            # 预设自动加水平边距（full 模式除外）
            if subtitle_region == "full":
                x1, x2 = 0.0, 1.0
            else:
                x1, x2 = DEFAULT_HORIZONTAL_MARGIN, 1.0 - DEFAULT_HORIZONTAL_MARGIN
            self._bbox = (x1, y_start, x2, y_end)
            self.subtitle_region = subtitle_region

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        裁剪字幕区域并缩放。

        Args:
            frame: BGR 格式的原始视频帧

        Returns:
            (cropped_original, processed_for_ocr)
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._bbox

        px1 = int(w * x1)
        py1 = int(h * y1)
        px2 = int(w * x2)
        py2 = int(h * y2)

        cropped = frame[py1:py2, px1:px2]

        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            cropped = frame.copy()

        if self.scale_factor != 1.0:
            new_w = int(cropped.shape[1] * self.scale_factor)
            new_h = int(cropped.shape[0] * self.scale_factor)
            processed = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            processed = cropped.copy()

        return cropped, processed

    def process_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        """便捷方法：直接返回适合 OCR 输入的图像（BGR 彩色）。"""
        _, processed = self.process(frame)
        return processed
