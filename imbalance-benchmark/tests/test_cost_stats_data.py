"""
成本、統計檢定、資料處理的測試：驗證數學公式寫對了。
"""

import numpy as np
import pandas as pd
import pytest

from imbalance_lab import cost, stats
from imbalance_lab.data import DEFAULT_DATA_DIR, SPECS, load_dataset, make_imbalanced
from imbalance_lab.metrics import compute_metrics

# ---------------------------------------------------------------------------
# cost.py
# ---------------------------------------------------------------------------


def test_bayes_threshold():
    # 漏抓違約成本是誤拒的 4 倍 → 只要違約機率 ≥ 20% 就該拒絕
    assert cost.bayes_threshold(c_fn=4, c_fp=1) == pytest.approx(0.2)


def test_expected_cost_by_hand():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.7, 0.1])
    # 門檻 0.5 → 拒絕第 1、3 筆：FN=1（第 2 筆）、FP=1（第 3 筆）
    assert cost.expected_cost(y, p, 0.5, c_fn=5, c_fp=1) == pytest.approx((5 + 1) / 4)


def test_optimal_threshold_matches_brute_force():
    """快速演算法的結果，要跟「每個門檻都試一遍」的暴力法一樣好。"""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 400), 0, 1)
    t = cost.optimal_threshold(y, p, c_fn=5, c_fp=1)
    brute = min(cost.expected_cost(y, p, c, 5, 1) for c in np.unique(np.r_[p, 1.01]))
    assert cost.expected_cost(y, p, t, 5, 1) == pytest.approx(brute)


def test_undersampling_correction():
    # beta = 1（沒有欠採樣）→ 機率不變
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(cost.correct_undersampled_proba(p, 1.0), p)
    # 欠採樣後模型說 50%，若負類只保留 1/4，真實機率應為 0.25·0.5/(0.25·0.5−0.5+1) = 0.2
    assert cost.correct_undersampled_proba(np.array([0.5]), 0.25)[0] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# stats.py
# ---------------------------------------------------------------------------


def test_corrected_ttest_is_more_conservative_than_naive():
    """修正後的 t 檢定 p 值應該比一般配對 t 檢定大（更保守）。"""
    from scipy.stats import ttest_rel
    rng = np.random.default_rng(1)
    a = rng.normal(0.60, 0.01, 15)
    b = a - rng.normal(0.004, 0.006, 15)
    _, p_corr = stats.corrected_resampled_ttest(a, b, test_train_ratio=0.25)
    assert p_corr > ttest_rel(a, b).pvalue


def test_holm_known_values():
    # 教科書例子：p = [0.01, 0.04, 0.03] → Holm = [0.03, 0.06, 0.06]
    np.testing.assert_allclose(stats.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_compare_to_baseline_detects_clear_improvement():
    rng = np.random.default_rng(2)
    rows = []
    for rep in range(3):
        for fold in range(5):
            base = 0.5 + rng.normal(0, 0.005)
            rows += [{"model": "m", "strategy": "none", "repeat": rep, "fold": fold, "pr_auc": base},
                     {"model": "m", "strategy": "good", "repeat": rep, "fold": fold, "pr_auc": base + 0.05},
                     {"model": "m", "strategy": "same", "repeat": rep, "fold": fold,
                      "pr_auc": base + rng.normal(0, 0.005)}]
    out = stats.compare_to_baseline(pd.DataFrame(rows), "pr_auc", n_splits=5).set_index("strategy")
    assert out.loc["good", "verdict"] == "better"
    assert out.loc["same", "verdict"] == "no sig. diff"


def test_average_ranks_gives_rank_1_to_the_best():
    """越大越好的指標：分數最高的策略要拿第 1 名（這個測試抓到過一次方向寫反的 bug）。"""
    rows = [{"model": m, "strategy": s, "repeat": 0, "fold": f, "pr_auc": v}
            for m in ["m1", "m2"] for f in range(3)
            for s, v in [("best", 0.9), ("mid", 0.5), ("worst", 0.1)]]
    ranks = stats.average_ranks(pd.DataFrame(rows), "pr_auc").set_index("strategy")
    assert ranks.loc["best", "avg_rank"] == 1.0
    assert ranks.loc["worst", "avg_rank"] == 3.0
    # 越小越好的指標（brier）方向相反
    rows_b = [{**r, "brier": r.pop("pr_auc")} for r in rows]
    ranks_b = stats.average_ranks(pd.DataFrame(rows_b), "brier").set_index("strategy")
    assert ranks_b.loc["worst", "avg_rank"] == 1.0


# ---------------------------------------------------------------------------
# data.py / metrics.py
# ---------------------------------------------------------------------------


def test_make_imbalanced_hits_target_ratio(binary_ds):
    for ratio in [6, 10, 20]:
        sub = make_imbalanced(binary_ds, ratio, seed=0)
        assert sub.imbalance_ratio == pytest.approx(ratio, rel=0.05)
        # 多數類完全保留
        assert (sub.y == 0).sum() == (binary_ds.y == 0).sum()


def test_make_imbalanced_rejects_impossible_ratio(binary_ds):
    with pytest.raises(ValueError):
        make_imbalanced(binary_ds, 1.5)  # 比原始 IR 還平衡，只靠丟資料做不到


def test_metrics_keys(binary_ds, multiclass_ds):
    y = binary_ds.y
    proba = np.column_stack([1 - y * 0.8, y * 0.8])
    m = compute_metrics(y, proba, "binary")
    assert m["recall"] == 1.0 and m["pr_auc"] == pytest.approx(1.0)
    y3 = multiclass_ds.y
    m3 = compute_metrics(y3, np.eye(3)[y3], "multiclass")
    assert m3["f1_macro"] == 1.0 and m3["min_recall"] == 1.0


@pytest.mark.skipif(not any(DEFAULT_DATA_DIR.rglob(SPECS["credit_card"].filename)),
                    reason="真實資料檔不存在（例如 CI 環境）就跳過")
def test_credit_card_cleaning():
    ds = load_dataset("credit_card")
    assert ds.X.shape == (30000, 23)
    assert set(ds.X.EDUCATION.unique()) <= {1, 2, 3, 4}   # 0/5/6 已併入 4
    assert set(ds.X.MARRIAGE.unique()) <= {1, 2, 3}       # 0 已併入 3
    assert "PAY_1" in ds.X and "PAY_0" not in ds.X
    assert ds.X.columns[-3:].tolist() == ["SEX", "EDUCATION", "MARRIAGE"]  # 類別欄在最後
    assert ds.y.sum() == 6636
