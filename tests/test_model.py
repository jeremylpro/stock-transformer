from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import FeatureTransformer, feature_columns


def test_transforms_fit_on_train_and_apply_to_test():
    train = pd.DataFrame({"ret_x": [0.0, 1.0, 2.0, 3.0, 100.0]})
    val = pd.DataFrame({"ret_x": [-50.0, 200.0, np.nan]})
    transformer = FeatureTransformer.fit(train, ["ret_x"], (0.01, 0.99))
    out = transformer.transform(val)
    lo, hi = transformer.lower["ret_x"], transformer.upper["ret_x"]
    assert lo == train["ret_x"].quantile(0.01)
    assert hi == train["ret_x"].quantile(0.99)
    assert out["ret_x"].min() == lo
    assert out["ret_x"].max() == hi
    assert out.loc[2, "ret_x"] == train["ret_x"].median()
    # Test data must not have influenced the bounds.
    assert hi != val["ret_x"].max()


def test_transformer_roundtrip(tmp_path):
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
    transformer = FeatureTransformer.fit(train, ["a"])
    path = transformer.save(tmp_path / "imputation.json")
    loaded = FeatureTransformer.load(path)
    pd.testing.assert_frame_equal(transformer.transform(train), loaded.transform(train))


def test_feature_sets_select_prefixes_and_exclude_vader():
    df = pd.DataFrame(
        columns=["ret_lag_1", "ta_rsi_14", "sent_mean", "sent_vader_mean", "y_next_up"]
    )
    assert feature_columns(df, "combined") == ["ret_lag_1", "ta_rsi_14", "sent_mean"]
    assert feature_columns(df, "tech_only") == ["ret_lag_1", "ta_rsi_14"]
    assert feature_columns(df, "sent_only") == ["sent_mean"]
