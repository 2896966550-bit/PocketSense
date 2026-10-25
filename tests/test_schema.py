import pandas as pd

from psense.data.schema import validate


def make_df():
    data = {
        "subject": ["S00"] * 4,
        "activity": ["walking"] * 4,
        "t_ms": [0, 20, 40, 60],
        "source": ["synthetic"] * 4,
        "placement": ["none"] * 4,
    }
    for c in ("ax", "ay", "az", "gx", "gy", "gz"):
        data[c] = [0.1, 0.2, 0.3, 0.4]
    return pd.DataFrame(data)


def test_valid_dataframe():
    assert validate(make_df()) == []


def test_missing_column_detected():
    df = make_df().drop(columns=["ax"])
    problems = validate(df)
    assert any("ax" in p for p in problems)


def test_nan_meta_detected():
    df = make_df()
    df.loc[0, "subject"] = None
    assert validate(df) != []


def test_non_numeric_channel_detected():
    df = make_df()
    df["gx"] = ["a", "b", "c", "d"]
    assert any("gx" in p for p in validate(df))
