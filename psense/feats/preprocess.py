"""信号预处理：带通滤波、重力/身体分量分离、时间戳重采样、滑窗。"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from scipy import signal

from ..data.schema import CHAN_COLS, RATE, WIN

SEG_GAP_MS = 200  # 间隔超过此值视为段断裂


def butter_bandpass(low: float = 0.3, high: float = 20.0, fs: int = RATE, order: int = 4):
    return signal.butter(order, [low, high], btype="band", fs=fs, output="sos")


def filter_channels(arr: np.ndarray, fs: int = RATE) -> np.ndarray:
    """arr: (n, 6) -> 逐通道带通滤波（0.3-20Hz）。"""
    return signal.sosfiltfilt(butter_bandpass(fs=fs), arr, axis=0)


def gravity_body_split(acc: np.ndarray, fs: int = RATE) -> tuple[np.ndarray, np.ndarray]:
    """加速度 -> (重力分量, 身体分量)。重力 = 0.3Hz 低通。"""
    sos = signal.butter(4, 0.3, btype="low", fs=fs, output="sos")
    gravity = signal.sosfiltfilt(sos, acc, axis=0)
    return gravity, acc - gravity


def resample_to_50hz(t_ms: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按时间戳线性插值到 50Hz 均匀网格。values 可为 (n,) 或 (n, c)。处理不规则采样。"""
    t_ms = np.asarray(t_ms, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    grid = np.arange(t_ms[0], t_ms[-1] + 1e-9, 1000.0 / RATE)
    if values.ndim == 1:
        return grid, np.interp(grid, t_ms, values)
    out = np.empty((len(grid), values.shape[1]))
    for k in range(values.shape[1]):
        out[:, k] = np.interp(grid, t_ms, values[:, k])
    return grid, out


def iter_segments(df: pd.DataFrame) -> Iterator[tuple[dict, np.ndarray]]:
    """按 (subject, activity, source, placement) 且时间连续切段。yield (meta, arr(n,6))。"""
    keys = ["subject", "activity", "source", "placement"]
    for meta, g in df.groupby(keys, sort=False):
        g = g.sort_values("t_ms")
        t = g["t_ms"].to_numpy()
        arr = g[CHAN_COLS].to_numpy(dtype=np.float64)
        breaks = np.where(np.diff(t) > SEG_GAP_MS)[0] + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [len(t)]])
        for s, e in zip(starts, ends):
            if e - s >= WIN:
                yield dict(zip(keys, meta)), arr[s:e]


def make_windows(
    df: pd.DataFrame,
    win: int = WIN,
    stride: int = 50,
    filter_fn=None,
) -> Iterator[dict]:
    """滑窗，测试集务必用 stride=win（窗口不重叠）。不足 win 的尾段丢弃。"""
    for meta, arr in iter_segments(df):
        n = len(arr)
        for s in range(0, n - win + 1, stride):
            x = arr[s:s + win]
            if filter_fn is not None:
                x = filter_fn(x)
            yield {**meta, "X": x}
