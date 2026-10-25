"""三个公开数据集解析器。输出统一约定：
- 加速度单位一律 g、且为"身体分量"（重力已分离）
- 陀螺单位 rad/s
- 长表列：subject/activity/t_ms/source/placement + ax..gz
- 标签统一到 5 类：walking/upstairs/downstairs/sitting/standing
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..feats.preprocess import gravity_body_split, resample_to_50hz
from .schema import CHAN_COLS, G_MS2

# ---------------- UCI-HAR：官方已按 2.56s 窗口（128 样本）给出，身体加速度（g）----------------

UCI_LABELS = {1: "walking", 2: "upstairs", 3: "downstairs", 4: "sitting", 5: "standing", 6: "laying"}
UCI_SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
               "body_gyro_x", "body_gyro_y", "body_gyro_z"]


def parse_uci(root: Path) -> dict:
    """返回 dict(split -> dict(X:(N,128,6), y:list[str], subject:list[str]))。"""
    base = root / "UCI HAR Dataset"
    out = {}
    for split in ("train", "test"):
        sig_dir = base / split / "Inertial Signals"
        chans = [pd.read_csv(sig_dir / f"{s}_{split}.txt", sep=r"\s+", header=None).to_numpy()
                 for s in UCI_SIGNALS]
        X = np.stack(chans, axis=-1).astype(np.float32)  # (N, 128, 6)
        y_num = pd.read_csv(base / split / f"y_{split}.txt", sep=r"\s+", header=None).to_numpy().ravel()
        subj = pd.read_csv(base / split / f"subject_{split}.txt", sep=r"\s+", header=None).to_numpy().ravel()
        out[split] = {
            "X": X,
            "y": [UCI_LABELS[int(v)] for v in y_num],
            "subject": [f"U{int(v):02d}" for v in subj],
        }
    return out


# ---------------- MotionSense：JSON 时序，userAcceleration（g，身体分量）----------------

MS_LABELS = {"dws": "downstairs", "ups": "upstairs", "wlk": "walking",
             "sit": "sitting", "std": "standing"}  # jog 弃用（5 类交集外）


def parse_motionsense(root: Path) -> pd.DataFrame:
    base = root / "A_DeviceMotion_data"
    if (base / "A_DeviceMotion_data").is_dir():  # 内层 zip 自带同名顶层目录
        base = base / "A_DeviceMotion_data"
    usecols = ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z",
               "rotationRate.x", "rotationRate.y", "rotationRate.z"]
    frames = []
    for trial_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        act = MS_LABELS.get(trial_dir.name.split("_")[0])
        if act is None:
            continue
        for cf in sorted(trial_dir.glob("sub_*.csv")):
            subject = "M" + cf.stem.split("_")[1]
            df = pd.read_csv(cf, usecols=usecols)
            df.columns = CHAN_COLS
            n = len(df)
            df.insert(0, "subject", subject)
            df.insert(1, "activity", act)
            df.insert(2, "t_ms", (np.arange(n) * 20).astype("int64"))  # 名义 50Hz
            df.insert(3, "source", "motionsense")
            df.insert(4, "placement", "pocket")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[["subject", "activity", "t_ms", "source", "placement"] + CHAN_COLS]


# ---------------- PAMAP2：100Hz 腕部 IMU，总加速度 m/s² → 分离重力 → g ----------------

PAMAP2_KEEP = {4: "walking", 12: "upstairs", 13: "downstairs", 2: "sitting", 3: "standing"}
# 列布局（0 基）：0 时间戳(s) 1 activityID 2 心率；腕部 4-6 acc16(m/s²) 10-12 gyro(rad/s)
COL_T, COL_ACT = 0, 1
COL_ACC = slice(4, 7)
COL_GYRO = slice(10, 13)


def parse_pamap2(root: Path) -> tuple[pd.DataFrame, float]:
    """返回 (长表, NaN 丢弃率)。"""
    frames, dropped, total = [], 0, 0
    for dat in sorted(root.rglob("subject*.dat")):
        if dat.parent.name != "Protocol":
            continue
        subject = "P" + dat.stem.replace("subject", "")
        arr = pd.read_csv(dat, sep=r"\s+", header=None).to_numpy(dtype=np.float64)
        total += len(arr)
        keep = ~np.isnan(arr[:, COL_ACC]).any(axis=1) & ~np.isnan(arr[:, COL_GYRO]).any(axis=1)
        dropped += int((~keep).sum())
        arr = arr[keep]
        if len(arr) == 0:
            continue
        act_mask = np.isin(arr[:, COL_ACT].astype(int), list(PAMAP2_KEEP))
        arr = arr[act_mask]
        if len(arr) == 0:
            continue
        # 按活动切连续段再处理（不同活动之间时间跳变）
        acts = arr[:, COL_ACT].astype(int)
        breaks = np.where(np.diff(acts) != 0)[0] + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [len(arr)]])
        for s, e in zip(starts, ends):
            seg = arr[s:e]
            if len(seg) < 200:
                continue
            t_ms = (seg[:, COL_T] - seg[0, COL_T]) * 1000.0
            acc_g = seg[:, COL_ACC] / G_MS2
            _, body = gravity_body_split(acc_g)
            gyro = seg[:, COL_GYRO]
            grid_t, res = resample_to_50hz(t_ms, np.column_stack([body, gyro]))
            df = pd.DataFrame(res, columns=CHAN_COLS)
            df.insert(0, "subject", subject)
            df.insert(1, "activity", PAMAP2_KEEP[int(acts[s])])
            df.insert(2, "t_ms", grid_t.astype("int64"))
            df.insert(3, "source", "pamap2")
            df.insert(4, "placement", "wrist")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[["subject", "activity", "t_ms", "source", "placement"] + CHAN_COLS], dropped / max(total, 1)
