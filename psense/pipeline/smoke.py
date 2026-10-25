"""端到端冒烟：合成数据 -> schema 校验 -> 滤波 -> 滑窗(按人划分) -> RF/CNN -> ONNX INT8 -> onnxruntime 校验。

铁律：合成数据的准确率不作为质量指标，本管线只证明"全链贯通"。
"""
from __future__ import annotations

import json
import shutil
import time
from itertools import islice
from pathlib import Path

import numpy as np

from ..data import synthetic
from ..data.schema import save_parquet, validate
from ..export import export_onnx
from ..feats.preprocess import filter_channels, make_windows
from ..models import split as S
from ..models import train_cnn, train_rf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT / "configs" / "smoke.json"


def run(cfg: dict, out_dir: Path) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = synthetic.generate(cfg["synthetic"])
    problems = validate(df)
    assert not problems, problems
    save_parquet(df, out_dir / "synthetic.parquet")
    print("[1/6] 合成数据 %d 行，schema 校验通过" % len(df))

    test_subjects = S.pick_test_subjects(
        df["subject"], frac=cfg["split"]["test_frac"], seed=cfg["split"]["seed"])
    df_train, df_test = S.split_by_subject(df, test_subjects)
    print("[2/6] 按被试划分：test=%s（train %d 行 / test %d 行）"
          % (test_subjects, len(df_train), len(df_test)))

    rf_metrics = train_rf.run(
        df_train, df_test, out_dir / "models", filter_fn=filter_channels)
    print("[3/6] RF 基线：acc=%.3f macroF1=%.3f（仅供参考，非质量指标）"
          % (rf_metrics["accuracy"], rf_metrics["macro_f1"]))

    cnn_metrics = train_cnn.run(
        df_train, df_test, out_dir / "models",
        epochs=cfg["cnn"]["epochs"], batch=cfg["cnn"]["batch"],
        filter_fn=filter_channels)
    print("[4/6] 1D-CNN：acc=%.3f params=%d（仅供参考，非质量指标）"
          % (cnn_metrics["accuracy"], cnn_metrics["params"]))

    cal_windows = list(islice(
        make_windows(df_train, stride=100, filter_fn=filter_channels), 200))
    calibration = np.stack([w["X"] for w in cal_windows]).astype(np.float32).transpose(0, 2, 1)
    onnx_metrics = export_onnx.export(out_dir / "models" / "cnn.pt", out_dir / "models", calibration)
    print("[5/6] ONNX：INT8 %s KB（FP32 %s KB），argmax 一致率 %.3f"
          % (onnx_metrics["size_int8_kb"], onnx_metrics["size_fp32_kb"],
             onnx_metrics["int8_argmax_agreement"]))

    web_model = ROOT / "web" / "model_int8.onnx"
    if web_model.parent.is_dir():
        shutil.copy(out_dir / "models" / "cnn_int8.onnx", web_model)
        print("[6/6] 已复制 INT8 模型到 web/model_int8.onnx（网页端侧推理用）")

    report = {
        "elapsed_s": round(time.time() - t0, 1),
        "test_subjects": test_subjects,
        "rf": rf_metrics, "cnn": cnn_metrics, "onnx": onnx_metrics,
    }
    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    cfg = json.loads(DEFAULT_CFG.read_text(encoding="utf-8"))
    report = run(cfg, ROOT / "data" / "synthetic")
    print("SMOKE PASS，用时 %.1fs（报告：data/synthetic/smoke_report.json）" % report["elapsed_s"])


if __name__ == "__main__":
    main()
