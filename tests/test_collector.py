import numpy as np

from psense.data.collector import load_collector_csv
from psense.data.schema import validate
from psense.feats.preprocess import make_windows


def test_collector_csv_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    n = 260
    rows = ["#subject=P99", "#activity=walking", "#placement=前口袋",
            "#source=collector", "#rate=50", "t_ms,ax,ay,az,gx,gy,gz"]
    for i in range(n):
        vals = rng.normal(0, 0.3, 6).round(4)
        rows.append(str(i * 20) + "," + ",".join(map(str, vals)))
    p = tmp_path / "collect_test.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")

    df = load_collector_csv(p)
    assert validate(df) == []
    assert df["subject"].iloc[0] == "P99"
    assert df["activity"].iloc[0] == "walking"
    assert len(df) == n
    assert sum(1 for _ in make_windows(df, stride=50)) >= 3
