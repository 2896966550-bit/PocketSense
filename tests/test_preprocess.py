import numpy as np

from psense.data import synthetic
from psense.feats.preprocess import (
    filter_channels,
    gravity_body_split,
    make_windows,
    resample_to_50hz,
)

FS = 50


def test_resample_interpolation():
    t = np.array([0.0, 25.0, 60.0, 95.0])
    v = np.array([0.0, 1.0, 2.0, 3.0])
    grid, out = resample_to_50hz(t, v)
    assert len(grid) == 5 and abs(grid[1] - grid[0] - 20.0) < 1e-9
    assert out[0] == 0.0
    # 分段线性：v(80ms) 在 (60,2)-(95,3) 段内
    assert abs(out[-1] - (2.0 + 20.0 / 35.0)) < 1e-9
    assert np.all(np.diff(out) > 0)


def test_resample_2d():
    t = np.array([0.0, 30.0, 70.0])
    v = np.column_stack([np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])])
    grid, out = resample_to_50hz(t, v)
    assert out.shape == (len(grid), 2)
    assert abs(out[0, 0] - 0.0) < 1e-9 and abs(out[0, 1] - 2.0) < 1e-9


def test_bandpass_passes_2hz_rejects_dc():
    t = np.arange(0, 5, 1 / FS)
    x2 = np.sin(2 * np.pi * 2 * t)
    f2 = filter_channels(x2[:, None])[:, 0]
    assert np.abs(f2[100:]).max() > 0.7
    xdc = 1.0 + 0.05 * np.sin(2 * np.pi * 0.05 * t)
    fdc = filter_channels(xdc[:, None])[:, 0]
    assert np.abs(fdc[100:]).max() < 0.2  # 直流与近直流被 0.3Hz 高通去除


def test_gravity_body_split():
    t = np.arange(0, 30, 1 / FS)
    acc = 1.0 + 0.05 * np.sin(2 * np.pi * 0.1 * t) + 0.8 * np.sin(2 * np.pi * 2 * t)
    g, b = gravity_body_split(acc[:, None])
    assert abs(g[500:-500, 0].mean() - 1.0) < 0.05
    assert abs(b[500:-500, 0].std() - 0.8 / np.sqrt(2)) < 0.05


def test_window_counts_and_nonoverlap():
    df = synthetic.generate({"n_subjects": 2, "windows_per_class": 5})
    for stride, expect in ((50, 5), (100, 3)):  # 段长 = 100+4*50+10 = 310
        counts = {}
        for w in make_windows(df, stride=stride):
            key = (w["subject"], w["activity"])
            counts[key] = counts.get(key, 0) + 1
        assert all(c == expect for c in counts.values()), (stride, counts)
        # stride=win 时窗口起点间隔 >= win，结构上不重叠；计数公式断言即为其等价验证
