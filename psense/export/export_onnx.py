"""ONNX 导出 + INT8 静态量化（校准数据驱动）+ 前后一致性校验。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from ..models.cnn import CNN1D


class _WindowReader(CalibrationDataReader):
    def __init__(self, data: np.ndarray):
        self._data = data.astype(np.float32)
        self._i = 0

    def get_next(self):
        if self._i >= len(self._data):
            return None
        item = {"x": self._data[self._i:self._i + 1]}
        self._i += 1
        return item


def export(state_dict_path: Path, out_dir: Path, calibration: np.ndarray | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "cnn_fp32.onnx"
    int8 = out_dir / "cnn_int8.onnx"

    model = CNN1D()
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.randn(1, 6, 100)
    torch.onnx.export(
        model, dummy, str(fp32),
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # 静态量化 + 无符号激活(QUInt8) + QOperator：生成 QLinearConv，
    # Python CPU 与浏览器 WASM 后端都支持（QInt8 激活的 ConvInteger 两端都可能缺内核）
    if calibration is None:
        calibration = np.random.default_rng(0).standard_normal((64, 6, 100))
    quantize_static(
        str(fp32), str(int8), _WindowReader(calibration),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    s_fp32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s_int8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    check_x = calibration[:64] if calibration is not None and len(calibration) >= 64 else \
        np.random.default_rng(0).standard_normal((64, 6, 100)).astype(np.float32)
    l1 = s_fp32.run(None, {"x": check_x})[0]
    l2 = s_int8.run(None, {"x": check_x})[0]
    agreement = float((l1.argmax(1) == l2.argmax(1)).mean())

    metrics = {
        "size_fp32_kb": round(fp32.stat().st_size / 1024, 1),
        "size_int8_kb": round(int8.stat().st_size / 1024, 1),
        "int8_argmax_agreement": agreement,
    }
    (out_dir / "onnx_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
