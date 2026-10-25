"""采集页 CSV 读取器。格式：
#subject=P07
#activity=walking
#placement=pocket
t_ms,ax,ay,az,gx,gy,gz
0,0.12,...
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import CHAN_COLS


def load_collector_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    meta: dict[str, str] = {}
    with open(p, encoding="utf-8") as f:
        for _ in range(10):
            line = f.readline()
            if not line.startswith("#"):
                break
            k, _, v = line.lstrip("#").strip().partition("=")
            meta[k] = v
    df = pd.read_csv(p, comment="#")
    df = df.rename(columns=str.strip)
    out = df[["t_ms"] + CHAN_COLS].copy()
    out.insert(0, "subject", meta.get("subject", "unknown"))
    out.insert(1, "activity", meta.get("activity", "unknown"))
    out.insert(2, "t_ms", out.pop("t_ms").astype("int64"))
    out.insert(3, "source", "collector")
    out.insert(4, "placement", meta.get("placement", "unknown"))
    return out[["subject", "activity", "t_ms", "source", "placement"] + CHAN_COLS]
