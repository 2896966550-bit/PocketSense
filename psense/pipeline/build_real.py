"""S2 管线：三公开数据集解析入库 + UCI 官方划分真实基线（验收绊线：acc >= 0.80）。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..data.fetch import ensure_all
from ..data.parsers import parse_motionsense, parse_pamap2, parse_uci
from ..data.schema import save_parquet, validate
from ..feats.preprocess import WIN, make_windows
from ..models.train_rf import build_xy

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def uci_baseline(uci: dict) -> dict:
    def to_windows(split: dict) -> list[dict]:
        X = split["X"][:, :WIN, :].astype(np.float64)  # 128 样本(2.56s) 截前 100(2s)
        return [{"X": X[i], "activity": split["y"][i], "subject": split["subject"][i]}
                for i in range(len(X))]

    x_tr, y_tr = build_xy(to_windows(uci["train"]))
    x_te, y_te = build_xy(to_windows(uci["test"]))
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    classes = clf.classes_.tolist()
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "classes": classes,
        "confusion": confusion_matrix(y_te, pred, labels=clf.classes_).tolist(),
        "n_train": len(y_tr), "n_test": len(y_te),
        "test_subjects": sorted(set(uci["test"]["subject"])),
    }


def main() -> None:
    t0 = time.time()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    raw = ensure_all()
    report: dict = {"timing": {}}

    # 1) UCI-HAR：官方 21/9 被试划分基线
    t = time.time()
    uci = parse_uci(raw["uci_har"])
    base = uci_baseline(uci)
    report["uci_baseline"] = base
    report["timing"]["uci"] = round(time.time() - t, 1)
    print("[1/4] UCI-HAR 官方划分基线：acc=%.3f macroF1=%.3f（%d/%d 窗）"
          % (base["accuracy"], base["macro_f1"], base["n_train"], base["n_test"]))
    assert base["accuracy"] >= 0.80, "基线低于绊线，解析或特征层可能有 bug: %s" % base["accuracy"]
    np.savez_compressed(
        PROCESSED / "uci_windows.npz",
        X_train=uci["train"]["X"][:, :WIN, :], y_train=np.array(uci["train"]["y"]),
        s_train=np.array(uci["train"]["subject"]),
        X_test=uci["test"]["X"][:, :WIN, :], y_test=np.array(uci["test"]["y"]),
        s_test=np.array(uci["test"]["subject"]),
    )

    # 2) MotionSense：长表入库 + 统计
    t = time.time()
    ms = parse_motionsense(raw["motionsense"])
    problems = validate(ms)
    assert not problems, problems
    save_parquet(ms, PROCESSED / "motionsense_5class.parquet")
    n_win = sum(1 for _ in make_windows(ms, stride=50))
    report["motionsense"] = {
        "rows": len(ms), "subjects": int(ms["subject"].nunique()),
        "class_rows": ms["activity"].value_counts().to_dict(),
        "windows_stride50": n_win,
    }
    report["timing"]["motionsense"] = round(time.time() - t, 1)
    print("[2/4] MotionSense：%d 行 / %d 人 / %d 窗（stride50）"
          % (len(ms), ms["subject"].nunique(), n_win))

    # 3) PAMAP2：长表入库 + 统计（含 NaN 丢弃率）
    t = time.time()
    pmp, drop_rate = parse_pamap2(raw["pamap2"])
    problems = validate(pmp)
    assert not problems, problems
    save_parquet(pmp, PROCESSED / "pamap2_5class.parquet")
    n_win = sum(1 for _ in make_windows(pmp, stride=50))
    report["pamap2"] = {
        "rows": len(pmp), "subjects": int(pmp["subject"].nunique()),
        "class_rows": pmp["activity"].value_counts().to_dict(),
        "windows_stride50": n_win, "nan_drop_rate": round(drop_rate, 4),
    }
    report["timing"]["pamap2"] = round(time.time() - t, 1)
    print("[3/4] PAMAP2：%d 行 / %d 人 / %d 窗 / NaN 丢弃率 %.1f%%"
          % (len(pmp), pmp["subject"].nunique(), n_win, drop_rate * 100))

    report["elapsed_s"] = round(time.time() - t0, 1)
    (REPORTS / "s2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[4/4] S2 PASS，用时 %.1fs（报告：reports/s2_report.json）" % report["elapsed_s"])


if __name__ == "__main__":
    main()
