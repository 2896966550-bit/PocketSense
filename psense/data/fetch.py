"""数据集下载与解压（可复现）。三数据集均为 zip 套娃结构，统一展开到 data/raw/<name>/。"""
from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DL = RAW / "downloads"

SOURCES = {
    "uci_har": "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    "motionsense": "https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip",
    "pamap2": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str) -> Path:
    DL.mkdir(parents=True, exist_ok=True)
    dest = DL / f"{name}.zip"
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    print(f"[fetch] 下载 {name} …")
    resp = requests.get(SOURCES[name], stream=True, timeout=300)
    resp.raise_for_status()
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"[fetch] {name}: {dest.stat().st_size / 1e6:.1f} MB sha256={_sha256(dest)[:16]}…")
    return dest


def extract(name: str) -> Path:
    """展开（含嵌套 zip）到 data/raw/<name>/，幂等。"""
    out = RAW / name
    marker = out / ".extracted"
    if marker.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    src = download(name)
    with zipfile.ZipFile(src) as z:
        if name == "motionsense":
            inner_bytes = io.BytesIO(z.read("motion-sense-master/data/A_DeviceMotion_data.zip"))
            with zipfile.ZipFile(inner_bytes) as zi:
                zi.extractall(out / "A_DeviceMotion_data")
            (out / "data_subjects_info.csv").write_bytes(
                z.read("motion-sense-master/data/data_subjects_info.csv"))
        elif name == "pamap2":
            inner_bytes = io.BytesIO(z.read("PAMAP2_Dataset.zip"))
            with zipfile.ZipFile(inner_bytes) as zi:
                zi.extractall(out)
        else:  # uci_har 直接展开
            z.extractall(out)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"[fetch] {name} 已展开到 {out}")
    return out


def ensure_all() -> dict[str, Path]:
    return {name: extract(name) for name in SOURCES}


if __name__ == "__main__":
    for n, p in ensure_all().items():
        print(n, "->", p)
