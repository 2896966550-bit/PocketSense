import pandas as pd
import pytest

from psense.models.split import pick_test_subjects, split_by_subject


def make_df(subjects):
    return pd.DataFrame({"subject": subjects, "v": range(len(subjects))})


def test_pick_and_split_disjoint():
    subjects = [f"S{i:02d}" for i in range(6)] * 3
    test_subjects = pick_test_subjects(subjects, frac=1 / 3, seed=0)
    assert len(test_subjects) == 2
    assert set(test_subjects) <= set(subjects)
    df = make_df(subjects)
    train, test = split_by_subject(df, test_subjects)
    assert set(train["subject"]) & set(test_subjects) == set()
    assert set(test["subject"]) == set(test_subjects)


def test_pick_is_deterministic():
    a = pick_test_subjects([f"S{i:02d}" for i in range(10)], frac=0.3, seed=1)
    b = pick_test_subjects([f"S{i:02d}" for i in range(10)], frac=0.3, seed=1)
    assert a == b


def test_too_few_subjects_raises():
    with pytest.raises(ValueError):
        pick_test_subjects(["S00"], frac=0.5)
