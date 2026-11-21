"""
統計檢定 (statistical comparison)
=================================

問題：策略 A 平均 PR-AUC 0.540、策略 B 0.536，A 真的比較好嗎？

舊版專案只有一個分數，沒辦法回答。現在每組有 15 個 CV 分數，可以做檢定。

但不能直接用一般的 paired t-test！
Repeated CV 的 15 個分數「不是互相獨立」的：不同折的訓練集有大量重疊
（5 折時任兩折的訓練集共用 75% 的資料），分數之間正相關，
一般 t-test 會低估變異數 → p 值太小 → 把雜訊誤判成顯著差異。

解法：Nadeau & Bengio (2003) 的 corrected resampled t-test，
把變異數乘上一個修正項 (1/k + n_test/n_train)，補償這個相關性。
這是機器學習比較模型時的標準做法之一（Bouckaert & Frank, 2004 也推薦）。

多重比較：同時比很多組時，總會有幾組「碰巧」顯著。
用 Holm 校正調整 p 值，控制整體的誤判率。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import LOWER_IS_BETTER


def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, test_train_ratio: float
                              ) -> tuple[float, float]:
    """
    Nadeau–Bengio 修正的配對 t 檢定。

    參數
    ----
    a, b : 兩種方法在「同樣的 k 個折」上的分數（順序要對應）
    test_train_ratio : 驗證折大小 / 訓練折大小；5 折 CV 時 = 1/4

    回傳 (t 統計量, 雙尾 p 值)
    """
    d = np.asarray(a) - np.asarray(b)   # 每一折的差異
    k = len(d)
    var = d.var(ddof=1)                 # 樣本變異數
    if var == 0:                        # 兩者完全一樣 → 沒有差異
        return 0.0, 1.0
    # 一般 t-test 用 var / k；修正版改成 var * (1/k + n_test/n_train)
    t = d.mean() / np.sqrt(var * (1.0 / k + test_train_ratio))
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def holm(pvals: np.ndarray) -> np.ndarray:
    """
    Holm-Bonferroni 校正。
    做法：p 值由小到大排序，第 i 小的乘上 (m - i)，再確保結果單調不減。
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adj[idx] = min(1.0, running_max)
    return adj


def compare_to_baseline(scores: pd.DataFrame, metric: str, n_splits: int,
                        baseline: str = "none", alpha: float = 0.05) -> pd.DataFrame:
    """
    對每個模型：每個策略 vs. baseline（none），計算差異、信賴區間與校正後 p 值。

    輸出欄位
    --------
    diff       : 平均差異（策略 − baseline），正數 = 策略比較好（brier 例外：負數較好）
    ci_low/high: 95% 信賴區間（同樣使用修正後的變異數）
    p_holm     : Holm 校正後的 p 值
    verdict    : better / worse / no sig. diff
    """
    ratio = 1.0 / (n_splits - 1)  # 驗證折 / 訓練折
    rows = []
    for model, g in scores.groupby("model"):
        base = g[g.strategy == baseline].sort_values(["repeat", "fold"])[metric].to_numpy()
        if len(base) == 0:
            continue  # 例如 balanced_rf 沒有 none 以外的策略，也就沒有東西可比
        for strategy, gs in g.groupby("strategy"):
            if strategy == baseline:
                continue
            vals = gs.sort_values(["repeat", "fold"])[metric].to_numpy()
            d = vals - base
            k = len(d)
            se = np.sqrt(d.var(ddof=1) * (1.0 / k + ratio))
            tcrit = stats.t.ppf(0.975, df=k - 1)
            _, p = corrected_resampled_ttest(vals, base, ratio)
            rows.append({"model": model, "strategy": strategy, "metric": metric,
                         "baseline_mean": base.mean(), "strategy_mean": vals.mean(),
                         "diff": d.mean(), "ci_low": d.mean() - tcrit * se,
                         "ci_high": d.mean() + tcrit * se, "p_raw": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    sign = -1 if metric in LOWER_IS_BETTER else 1

    def verdict(r):
        if r.p_holm >= alpha:
            return "no sig. diff"
        return "better" if sign * r["diff"] > 0 else "worse"

    out["verdict"] = out.apply(verdict, axis=1)
    return out


def average_ranks(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    平均排名：在每個 (模型, repeat, fold) 內，把各策略依分數排名（1 = 最好），
    再對所有區塊取平均。排名比原始分數更不受「各模型分數尺度不同」影響。
    只納入擁有全部策略的模型（例如 KNN 少了 class_weight 就不納入），排名才可比。
    """
    n_strat = scores.strategy.nunique()
    full = scores.groupby("model").strategy.nunique()
    keep = full[full == n_strat].index
    s = scores[scores.model.isin(keep)]
    # 越大越好的指標（PR-AUC 等）→ 由大到小排，最大的得第 1 名 → ascending=False
    # 越小越好的指標（brier）→ 由小到大排 → ascending=True
    ascending = metric in LOWER_IS_BETTER
    ranks = s.groupby(["model", "repeat", "fold"])[metric].rank(ascending=ascending)
    s = s.assign(rank=ranks)
    out = (s.groupby("strategy")["rank"].agg(["mean", "std"]).sort_values("mean")
           .rename(columns={"mean": "avg_rank", "std": "rank_std"}).reset_index())
    # Friedman 檢定：「所有策略表現都一樣」這個虛無假設是否成立
    # 注意：CV 各折並非完全獨立，Friedman 的 p 值偏樂觀，只當作輔助參考；
    # 個別策略的結論以 compare_to_baseline（修正 t 檢定）為準
    pivot = s.pivot_table(index=["model", "repeat", "fold"], columns="strategy", values=metric)
    if pivot.shape[1] >= 3:
        stat, p = stats.friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
        out.attrs["friedman"] = {"statistic": float(stat), "p": float(p),
                                 "models": list(keep)}
    return out
