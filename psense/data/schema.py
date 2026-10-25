"""统一数据 schema：长表（每行一个采样点），全项目唯一的物理单位为 g（重力加速度）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

META_COLS = ["subject", "activity", "t_ms", "source", "placement"]
CHAN_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
ALL_COLS = META_COLS + CHAN_COLS
ACTIVITIES_5 = ["walking", "upstairs", "downstairs", "sitting", "standing"]
RATE = 50  # Hz，全项目统一采样率
WIN = 100  # 2s @ 50Hz，全项目统一窗口长度
G_MS2 = 9.81  # PAMAP2 的 m/s² 转 g 用


def validate(df: pd.DataFrame) -> list[str]:
    """返回问题列表；空列表 = 合法。"""
    problems: list[str] = []
    for c in ALL_COLS:
        if c not in df.columns:
            problems.append("缺列 %s" % c)
    if problems:
        return problems
    for c in CHAN_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            problems.append("%s 非数值类型" % c)
    for c in ["subject", "activity"]:
        if df[c].isna().any() or (df[c].astype(str).str.len() == 0).any():
            problems.append("%s 含空值" % c)
    if df["t_ms"].isna().any():
        problems.append("t_ms 含 NaN")
    if len(df) == 0:
        problems.append("空表")
    return problems


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
