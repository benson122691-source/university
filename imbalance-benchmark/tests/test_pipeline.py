"""
Pipeline 測試：最重要的是「沒有資料洩漏」。
"""

import numpy as np
import pytest
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold

from imbalance_lab.experiments import combos, run_cv
from imbalance_lab.models import is_compatible
from imbalance_lab.pipeline import build_pipeline


class SpySampler(RandomUnderSampler):
    """會「記錄自己被呼叫時看到幾筆資料」的 sampler，用來監視 Pipeline 的行為。"""

    seen_sizes: list = []

    def fit_resample(self, X, y, **kw):
        SpySampler.seen_sizes.append(len(y))
        return super().fit_resample(X, y, **kw)


def test_sampler_only_sees_training_fold(binary_ds):
    """
    重採樣只能看到訓練折：
    - fit 時，sampler 收到的筆數 = 訓練折筆數（不是全部資料）
    - predict 時，sampler 完全不會被呼叫，輸出筆數 = 驗證折筆數
    """
    SpySampler.seen_sizes = []
    pipe = build_pipeline(binary_ds, "logreg", "random_under", seed=0)
    pipe.set_params(sampler=SpySampler(random_state=0))  # 把真的 sampler 換成間諜

    for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(binary_ds.X, binary_ds.y):
        pipe.fit(binary_ds.X.iloc[tr], binary_ds.y[tr])
        n_calls_before = len(SpySampler.seen_sizes)
        proba = pipe.predict_proba(binary_ds.X.iloc[va])
        assert len(SpySampler.seen_sizes) == n_calls_before, "predict 時不應該重採樣"
        assert proba.shape[0] == len(va), "驗證折的筆數不應被改變"
        assert SpySampler.seen_sizes[-1] == len(tr), "sampler 應該只看到訓練折"


def test_smotenc_keeps_valid_category_codes(binary_ds):
    """SMOTENC 合成的樣本，類別欄只能是原本出現過的代碼（不能有 1.37 這種值）。"""
    pipe = build_pipeline(binary_ds, "logreg", "smote", seed=0)
    Xs = pipe.named_steps["scale"].fit_transform(binary_ds.X)
    X_res, y_res = pipe.named_steps["sampler"].fit_resample(Xs, binary_ds.y)
    cat_col = X_res[:, -1]  # 類別欄被排在最後一欄
    assert set(np.unique(cat_col)) <= {1, 2, 3}
    assert np.bincount(y_res)[0] == np.bincount(y_res)[1], "SMOTE 後兩類應該一樣多"


@pytest.mark.parametrize("strategy", ["none", "class_weight", "random_under", "tomek", "smote"])
@pytest.mark.parametrize("model", ["logreg", "knn", "random_forest", "xgboost", "balanced_rf"])
def test_every_compatible_combo_runs(binary_ds, multiclass_ds, model, strategy):
    """每個合理的組合，在二分類和多分類上都要能跑、輸出合法的機率。"""
    if not is_compatible(model, strategy):
        pytest.skip("不合理的組合，實驗中也會跳過")
    for ds in (binary_ds, multiclass_ds):
        pipe = build_pipeline(ds, model, strategy, seed=0)
        pipe.fit(ds.X, ds.y)
        proba = pipe.predict_proba(ds.X.iloc[:50])
        assert proba.shape == (50, len(ds.class_names))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_class_weight_actually_changes_xgboost(binary_ds):
    """class_weight 策略要真的生效：少數類的平均預測機率應該變高。"""
    p_none = build_pipeline(binary_ds, "xgboost", "none", 0).fit(binary_ds.X, binary_ds.y)
    p_cw = build_pipeline(binary_ds, "xgboost", "class_weight", 0).fit(binary_ds.X, binary_ds.y)
    assert p_cw.predict_proba(binary_ds.X)[:, 1].mean() > p_none.predict_proba(binary_ds.X)[:, 1].mean()


def test_run_cv_produces_paired_folds_and_complete_oof(binary_ds):
    scores, oof = run_cv(binary_ds, ["logreg", "knn"], ["none", "class_weight", "random_under"],
                         params={}, n_splits=3, n_repeats=2, seed=0, n_jobs=1)
    # 每個組合 3 折 × 2 次 = 6 列；KNN 不能用 class_weight → 2 + 3 = 5 個組合
    assert len(combos(["logreg", "knn"], ["none", "class_weight", "random_under"])) == 5
    assert len(scores) == 5 * 6
    # 配對：每個組合的 (repeat, fold) 集合要一模一樣
    keys = scores.groupby(["model", "strategy"]).apply(
        lambda g: tuple(sorted(zip(g.repeat, g.fold, strict=True))), include_groups=False)
    assert keys.nunique() == 1
    # OOF：每一筆資料都要有預測
    for proba in oof.values():
        assert not np.isnan(proba).any()


def test_step_order_depends_on_sampler(binary_ds):
    """SMOTENC 要在 one-hot 之前（需要類別代碼）；其他 sampler 要在 one-hot 之後（距離才有意義）。"""
    names = lambda s: [n for n, _ in build_pipeline(binary_ds, "logreg", s, 0).steps]  # noqa: E731
    assert names("smote") == ["scale", "sampler", "encode", "model"]
    assert names("tomek") == ["scale", "encode", "sampler", "model"]
    assert names("random_under") == ["scale", "encode", "sampler", "model"]
    assert names("none") == ["scale", "encode", "model"]


def test_smote_handles_tiny_classes():
    """最小類別只有 3 筆時（Yeast 的 ERL 就會這樣），SMOTE 要自動把 k 降低而不是報錯。"""
    import pandas as pd

    from imbalance_lab.data import Dataset
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 30 + [2] * 3)
    X = pd.DataFrame(rng.normal(size=(len(y), 4)) + y[:, None], columns=list("abcd"))
    ds = Dataset(key="tiny", display="tiny", X=X, y=y, class_names=["a", "b", "c"])
    pipe = build_pipeline(ds, "logreg", "smote", seed=0).fit(ds.X, ds.y)
    assert pipe.named_steps["sampler"].k_neighbors == 5, "設定值不應被永久改掉"
    assert pipe.predict_proba(ds.X).shape == (len(y), 3)


def test_knn_survives_more_neighbors_than_samples():
    """欠採樣後訓練資料可能比 n_neighbors 還少，KNN 要自動縮小 k 而不是報錯。"""
    from imbalance_lab.models import build_model
    rng = np.random.default_rng(0)
    knn = build_model("knn", class_weight=False, seed=0).set_params(n_neighbors=50)
    X, y = rng.normal(size=(30, 3)), np.array([0, 1, 2] * 10)
    proba = knn.fit(X, y).predict_proba(rng.normal(size=(5, 3)))
    assert proba.shape == (5, 3) and knn.n_neighbors == 50
