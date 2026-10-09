import numpy as np
import pandas as pd

from creditrisk.woe import WoETransformer


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    num = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(1.5 * (cat == "c") - 1.0 * (cat == "a") + 0.8 * num - 0.3)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"cat": cat, "num": num}), pd.Series(y)


def test_risky_category_has_lower_woe():
    X, y = _data()
    t = WoETransformer().fit(X, y)
    woe = t.tables_["cat"].set_index("bin")["woe"]
    assert woe["c"] < woe["a"]  # c defaults more -> less "good evidence"


def test_numeric_binning_and_monotone_direction():
    X, y = _data()
    t = WoETransformer().fit(X, y)
    woe = t.tables_["num"]["woe"].to_numpy()
    assert 2 <= len(woe) <= 5
    assert woe[0] > woe[-1]  # higher num -> more default -> lower WoE


def test_transform_shape_and_unseen_category_is_neutral():
    X, y = _data()
    t = WoETransformer().fit(X, y)
    out = t.transform(pd.DataFrame({"cat": ["zzz"], "num": [0.0]}))
    assert list(out.columns) == ["cat", "num"]
    assert out.loc[0, "cat"] == 0.0


def test_iv_is_non_negative_and_ranked():
    X, y = _data()
    iv = WoETransformer().fit(X, y).iv_summary()
    assert (iv["iv"] >= 0).all()
    assert iv["iv"].is_monotonic_decreasing


def test_extreme_numeric_values_fall_in_outer_bins():
    X, y = _data()
    t = WoETransformer().fit(X, y)
    out = t.transform(pd.DataFrame({"cat": ["a", "a"], "num": [-1e9, 1e9]}))
    assert out["num"].iloc[0] == t.tables_["num"]["woe"].iloc[0]
    assert out["num"].iloc[1] == t.tables_["num"]["woe"].iloc[-1]
