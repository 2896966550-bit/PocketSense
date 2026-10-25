"""端到端冒烟测试。阈值仅是'管线没坏'的回归绊线（合成数据本应近乎完美可分），
不代表任何模型质量结论。"""
from psense.pipeline.smoke import run


def test_smoke_end_to_end(tmp_path):
    cfg = {
        "synthetic": {"seed": 3, "n_subjects": 6, "windows_per_class": 10},
        "split": {"test_frac": 0.34, "seed": 0},
        "cnn": {"epochs": 15, "batch": 32},
        "rf": {},
    }
    report = run(cfg, tmp_path)
    assert (tmp_path / "smoke_report.json").exists()
    assert (tmp_path / "models" / "cnn_int8.onnx").exists()
    assert report["rf"]["accuracy"] > 0.6, report["rf"]
    assert report["cnn"]["accuracy"] >= 0.3, report["cnn"]
    assert report["onnx"]["int8_argmax_agreement"] > 0.8, report["onnx"]
