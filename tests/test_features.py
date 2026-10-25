import numpy as np

from psense.feats.features import FEATURE_NAMES, window_features


def test_feature_count():
    assert len(FEATURE_NAMES) == 27


def test_known_sine():
    t = np.arange(100) / 50
    x = np.zeros((100, 6))
    x[:, 0] = 1.0 * np.sin(2 * np.pi * 2 * t)
    x[:, 1] = 0.5 * np.sin(2 * np.pi * 2 * t + 0.5)
    x[:, 2] = 0.3 * np.sin(2 * np.pi * 2 * t + 1.0)
    f = window_features(x)
    assert abs(f[FEATURE_NAMES.index("dom_freq")] - 2.0) <= 0.75  # FFT 分辨率 0.5Hz
    assert abs(f[FEATURE_NAMES.index("ax_std")] - 1 / np.sqrt(2)) < 0.02
    assert np.isfinite(f).all()


def test_flat_window_is_finite():
    f = window_features(np.zeros((100, 6)))
    assert np.isfinite(f).all()
