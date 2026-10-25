"""合成 IMU 数据生成器：仅用于管线冒烟自检，产出永远不进入实验结论。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ACTIVITIES_5, CHAN_COLS, RATE, WIN

SEG_PAD = 10

# 每类活动的信号签名：主频 Hz / 加速度幅值(g) / 噪声σ / 陀螺强度
# 注意：主频必须落在 0.3-20Hz 通带内（否则被滤波器抹掉），类间靠 (f, amp, gyro) 组合区分
SIGNS = {
    "walking":    dict(f=1.90, amp=1.00, noise=0.12, gyro=0.8),
    "upstairs":   dict(f=1.25, amp=1.35, noise=0.16, gyro=1.4),
    "downstairs": dict(f=1.50, amp=1.20, noise=0.18, gyro=1.3),
    "sitting":    dict(f=0.50, amp=0.05, noise=0.04, gyro=0.03),
    "standing":   dict(f=0.40, amp=0.16, noise=0.04, gyro=0.08),
}


def generate(config: dict) -> pd.DataFrame:
    """config: seed / n_subjects / windows_per_class（stride 按 50 计）。"""
    cfg = dict(seed=7, n_subjects=8, windows_per_class=40)
    cfg.update(config)
    rng = np.random.default_rng(cfg["seed"])
    frames = []
    for s in range(cfg["n_subjects"]):
        subject = "S%02d" % s
        f_mul = rng.uniform(0.85, 1.15)   # 被试间步频差异
        a_mul = rng.uniform(0.80, 1.20)   # 被试间幅值差异
        for act in ACTIVITIES_5:
            p = SIGNS[act]
            n = WIN + (cfg["windows_per_class"] - 1) * 50 + SEG_PAD
            t = np.arange(n) / RATE
            cols = np.zeros((n, 6))
            for k in range(3):
                phase = rng.uniform(0, 2 * np.pi)
                gain = [1.0, 0.75, 0.55][k]
                cols[:, k] = (
                    p["amp"] * a_mul * gain * np.sin(2 * np.pi * p["f"] * f_mul * t + phase)
                    + rng.normal(0, p["noise"], n)
                )
            for k in range(3):
                phase = rng.uniform(0, 2 * np.pi)
                cols[:, 3 + k] = (
                    0.2 * p["gyro"] * np.sin(2 * np.pi * p["f"] * f_mul * t + phase + k)
                    + rng.normal(0, p["noise"] * 0.5, n)
                )
            df = pd.DataFrame(cols, columns=CHAN_COLS)
            df.insert(0, "subject", subject)
            df.insert(1, "activity", act)
            df.insert(2, "t_ms", (np.arange(n) * (1000 // RATE)).astype("int64"))
            df.insert(3, "source", "synthetic")
            df.insert(4, "placement", "none")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[["subject", "activity", "t_ms", "source", "placement"] + CHAN_COLS]
