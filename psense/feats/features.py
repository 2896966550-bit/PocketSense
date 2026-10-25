"""27 维时频特征。输入窗口 (100, 6)，列序 ax..az, gx..gz。"""
from __future__ import annotations

import numpy as np

from ..data.schema import RATE, WIN

FEATURE_NAMES = (
    [f"{ax}_{stat}" for ax in ("ax", "ay", "az") for stat in ("mean", "std", "sma", "energy", "maxabs")]
    + ["corr_xy", "corr_xz", "corr_yz"]
    + ["mag_mean", "mag_std"]
    + ["dom_freq", "spec_entropy"]
    + ["gx_std", "gy_std", "gz_std"]
    + ["mag_zcr", "tilt_mean"]
)
assert len(FEATURE_NAMES) == 27


def window_features(x: np.ndarray) -> np.ndarray:
    """x: (win, 6) -> 27 维特征向量（顺序与 FEATURE_NAMES 一致）。"""
    acc = x[:, :3]
    gyr = x[:, 3:]
    n = len(x)
    feats: list[float] = []
    for k in range(3):
        v = acc[:, k]
        feats += [
            float(v.mean()),
            float(v.std()),
            float(np.abs(v).sum() / n),
            float((v ** 2).sum() / n),
            float(np.abs(v).max()),
        ]

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    feats += [
        safe_corr(acc[:, 0], acc[:, 1]),
        safe_corr(acc[:, 0], acc[:, 2]),
        safe_corr(acc[:, 1], acc[:, 2]),
    ]

    mag = np.sqrt((acc ** 2).sum(axis=1))
    feats += [float(mag.mean()), float(mag.std())]

    z = mag - mag.mean()
    # 按轴求和的功率谱：避免矢量幅值 |sin| 整流导致主频翻倍
    spec_sum = np.zeros(len(np.fft.rfftfreq(n, 1.0 / RATE)))
    for k in range(3):
        v = acc[:, k] - acc[:, k].mean()
        spec_sum += np.abs(np.fft.rfft(v)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    if spec_sum[1:].sum() > 1e-12:
        dom = float(freqs[1 + int(np.argmax(spec_sum[1:]))])
        p = spec_sum[1:] / spec_sum[1:].sum()
        ent = float(-(p * np.log(p + 1e-12)).sum())
    else:
        dom, ent = 0.0, 0.0
    feats += [dom, ent]

    feats += [float(gyr[:, k].std()) for k in range(3)]

    zc = float(np.sum(np.diff(np.sign(z)) != 0) / n)
    feats.append(zc)

    tilt = float(np.arccos(np.clip(acc[:, 2].mean() / (mag.mean() + 1e-12), -1.0, 1.0)))
    feats.append(tilt)

    out = np.asarray(feats, dtype=np.float64)
    assert out.shape == (27,), out.shape
    return out
