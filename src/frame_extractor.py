"""
模块 1: 帧提取器 (FrameExtractor)

从视频文件中按指定采样率抽取帧，并记录每帧的时间戳。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple


class FrameExtractor:
    """
    从视频文件中按采样率提取帧。

    Args:
        video_path: 视频文件路径
        fps: 采样帧率（帧/秒）。默认 1.0，即每秒提取1帧。
             设置越高精度越好但处理越慢。
    """

    def __init__(self, video_path: str, fps: float = 1.0):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        self.fps = fps
        self._cap = None
        self._video_fps = None
        self._total_frames = None
        self._duration_ms = None

    def _open(self):
        """打开视频文件并读取基本信息。"""
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_path}")

        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration_ms = (self._total_frames / self._video_fps) * 1000 if self._video_fps > 0 else 0

    def _close(self):
        """释放视频资源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def video_info(self) -> dict:
        """返回视频基本信息。"""
        if self._cap is None:
            self._open()
        return {
            "path": str(self.video_path),
            "filename": self.video_path.name,
            "video_fps": self._video_fps,
            "total_frames": self._total_frames,
            "duration_ms": self._duration_ms,
            "duration_str": self._ms_to_timestamp(self._duration_ms),
            "sample_fps": self.fps,
        }

    def extract(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        生成器：逐帧提取视频帧。

        Yields:
            Tuple[timestamp_ms, frame]:
                - timestamp_ms: 当前帧的时间戳（毫秒）
                - frame: BGR 格式的 numpy 数组
        """
        self._open()
        try:
            # 计算跳帧间隔：视频帧率 / 采样帧率
            frame_interval = max(1, int(round(self._video_fps / self.fps)))
            frame_idx = 0

            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                # 只在采样间隔处抽取帧
                if frame_idx % frame_interval == 0:
                    timestamp_ms = int((frame_idx / self._video_fps) * 1000)
                    yield timestamp_ms, frame

                frame_idx += 1
        finally:
            self._close()

    def get_total_sample_count(self) -> int:
        """估算总采样帧数（用于进度条）。"""
        if self._video_fps is None:
            self._open()
        duration_s = self._duration_ms / 1000
        return max(1, int(duration_s * self.fps))

    @staticmethod
    def _ms_to_timestamp(ms: int) -> str:
        """将毫秒转换为 HH:MM:SS 格式。"""
        s = int(ms / 1000)
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"
