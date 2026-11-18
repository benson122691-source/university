"""
成本導向決策 (cost-sensitive decision analysis)
===============================================

銀行要的不是最高的 F1，而是「最少的損失」。這個模組把模型輸出的機率，
轉成「核准 / 拒絕」的決策，並用成本來評估。

成本矩陣（單位是相對成本，FP 的成本定為 1）：

                    實際：正常還款        實際：違約
    預測：正常(核准)    0                 C_FN（核准了壞客戶 → 損失本金）
    預測：違約(拒絕)    C_FP（拒絕好客戶 → 少賺利息）   0

決策規則：違約機率 p ≥ 門檻 t → 拒絕。

三種選門檻的方式（policy）：
1. default  ：t = 0.5（大部分人直接用 predict() 就是這個）
2. bayes    ：t* = C_FP / (C_FP + C_FN)
              推導：拒絕的期望成本 = (1-p)·C_FP，核准的期望成本 = p·C_FN，
              拒絕比較划算 ⇔ p·C_FN ≥ (1-p)·C_FP ⇔ p ≥ C_FP/(C_FP+C_FN)。
              這只有在「機率準確（校準良好）」時才是最佳解。
3. tuned    ：在訓練集的 out-of-fold 預測上，直接搜尋讓總成本最低的門檻。
              不依賴機率是否校準，但需要額外資料來選（我們用 OOF，所以不碰 test）。

為什麼這跟「欠採樣」有關？
欠採樣把多數類丟掉，等於人為提高少數類的比例，模型預測出來的機率會整體「偏高」。
它的效果很大一部分就是「把門檻往下移」—— 而移門檻本身完全不需要丟資料。
`correct_undersampled_proba` 實作了 Dal Pozzolo et al. (2015) 的修正公式，
可以把欠採樣後偏高的機率換算回真實比例。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_cost(y: np.ndarray, p: np.ndarray, threshold: float,
                  c_fn: float, c_fp: float) -> float:
    """平均每位申請人的成本。"""
    reject = p >= threshold
    fn = np.sum((y == 1) & ~reject)   # 違約卻被核准
    fp = np.sum((y == 0) & reject)    # 正常卻被拒絕
    return float((c_fn * fn + c_fp * fp) / len(y))


def bayes_threshold(c_fn: float, c_fp: float) -> float:
    """機率校準良好時的理論最佳門檻。"""
    return c_fp / (c_fp + c_fn)


def optimal_threshold(y: np.ndarray, p: np.ndarray, c_fn: float, c_fp: float) -> float:
    """
    在資料上找讓成本最低的門檻。

    做法（向量化，O(n log n)）：
    把樣本依機率由大到小排序；門檻設在第 i 個樣本時，前 i 個被拒絕。
    用累加 (cumsum) 一次算出所有門檻下的 FP、FN 數量，取成本最小者。
    """
    order = np.argsort(-p)
    p_sorted, y_sorted = p[order], y[order]
    # 拒絕前 i 筆時：FP = 前 i 筆中的負類數；FN = 全部正類 − 前 i 筆中的正類數
    tp_cum = np.concatenate([[0], np.cumsum(y_sorted == 1)])
    fp_cum = np.concatenate([[0], np.cumsum(y_sorted == 0)])
    fn = tp_cum[-1] - tp_cum
    cost = c_fn * fn + c_fp * fp_cum
    i = int(np.argmin(cost))
    if i == 0:                        # 一個都不拒絕：門檻設在最大機率之上
        return float(p_sorted[0] + 1e-9)
    if i == len(p):                   # 全部拒絕
        return float(p_sorted[-1])
    # 門檻取「第 i 筆」與「第 i+1 筆」機率的中點，避免剛好落在某個樣本上
    return float((p_sorted[i - 1] + p_sorted[i]) / 2)


def correct_undersampled_proba(p_s: np.ndarray, beta: float) -> np.ndarray:
    """
    Dal Pozzolo et al. (2015)：把欠採樣模型的機率修正回原始分布。

    beta = 欠採樣時「每個負類樣本被保留的機率」= 保留的負類數 / 原本負類數
    公式：p = beta · p_s / (beta · p_s − p_s + 1)
    """
    return beta * p_s / (beta * p_s - p_s + 1)


def cost_table(oof: dict, test_proba: dict, y_train: np.ndarray, y_test: np.ndarray,
               c_fn: float, c_fp: float) -> pd.DataFrame:
    """
    對每個 (模型, 策略) × 三種門檻規則，計算 test set 上的成本。

    savings_vs_approve_all：相對於「全部核准、不用模型」能省下多少成本 (%)。
    這是給非技術主管看的指標：「用了模型每 100 位申請人省下多少損失」。
    """
    baseline_cost = c_fn * y_test.mean()  # 全部核准 → 所有違約者都變成 FN
    rows = []
    for key, p_test_full in test_proba.items():
        p_test = p_test_full[:, 1]
        p_oof = oof[key][:, 1]
        thresholds = {
            "default": 0.5,
            "bayes": bayes_threshold(c_fn, c_fp),
            "tuned": optimal_threshold(y_train, p_oof, c_fn, c_fp),
        }
        for policy, t in thresholds.items():
            reject = p_test >= t
            cost = expected_cost(y_test, p_test, t, c_fn, c_fp)
            rows.append({
                "model": key[0], "strategy": key[1], "policy": policy, "threshold": t,
                "cost_per_applicant": cost,
                "savings_vs_approve_all": 1 - cost / baseline_cost,
                "reject_rate": reject.mean(),
                "recall": reject[y_test == 1].mean(),          # 抓到多少違約者
                "false_reject_rate": reject[y_test == 0].mean(),  # 誤拒多少好客戶
            })
    return pd.DataFrame(rows)


def sensitivity(oof: dict, test_proba: dict, y_train: np.ndarray, y_test: np.ndarray,
                ratios: list[float], c_fp: float = 1.0, policy: str = "tuned") -> pd.DataFrame:
    """
    敏感度分析：C_FN 是 C_FP 的 1 倍、2 倍 … 20 倍時，各策略的成本如何變化？

    真實的成本比例很難精確估計（違約損失率、利率、回收率都會變），
    如果「哪個策略最好」會隨假設改變而翻盤，就必須在報告中說清楚。
    """
    frames = []
    for r in ratios:
        t = cost_table(oof, test_proba, y_train, y_test, c_fn=r * c_fp, c_fp=c_fp)
        t = t[t.policy == policy].assign(cost_ratio=r)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def cost_vs_baseline(oof: dict, test_proba: dict, y_train: np.ndarray, y_test: np.ndarray,
                     c_fn: float, c_fp: float, baseline: tuple[str, str] = ("none", "tuned"),
                     n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """
    每個 (模型, 策略, 門檻規則) 的 test 成本，與「同一模型 + none + tuned 門檻」比較，
    並用「配對 bootstrap」算 95% 信賴區間。

    為什麼需要？test set 只做一次評估，成本差 0.03 可能只是幾筆樣本的差別。
    German Credit 的 test set 只有 200 筆 —— 沒有信賴區間，就無法分辨真差異和運氣。

    配對 bootstrap 做法：
    1. 從 test set 有放回地抽 n 筆（同一組索引同時套用在兩個方法上 → 「配對」）
    2. 計算兩個方法在這組樣本上的平均成本差
    3. 重複 2000 次，取第 2.5 與 97.5 百分位數 = 95% 信賴區間
    區間不包含 0 → 差異大致可信；包含 0 → 跟 baseline 分不出高下。
    門檻是在訓練集 OOF 上選好的，bootstrap 時固定不變（test 不參與選門檻）。
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    boot_idx = rng.integers(0, n, size=(n_boot, n))  # 每一列是一次 bootstrap 抽樣

    def per_sample_cost(p, t):
        """每一位申請人各自的成本（0、C_FP 或 C_FN）。"""
        reject = p >= t
        return np.where((y_test == 1) & ~reject, c_fn, 0.0) + np.where((y_test == 0) & reject, c_fp, 0.0)

    thresholds = {}
    for key in test_proba:
        thresholds[(key, "default")] = 0.5
        thresholds[(key, "bayes")] = bayes_threshold(c_fn, c_fp)
        thresholds[(key, "tuned")] = optimal_threshold(y_train, oof[key][:, 1], c_fn, c_fp)

    rows = []
    for (model, strategy), proba in test_proba.items():
        base_key = (model, baseline[0])
        if base_key not in test_proba:
            continue
        base = per_sample_cost(test_proba[base_key][:, 1], thresholds[(base_key, baseline[1])])
        for policy in ("default", "bayes", "tuned"):
            if (strategy, policy) == baseline:
                continue
            this = per_sample_cost(proba[:, 1], thresholds[((model, strategy), policy)])
            d = this - base
            boots = d[boot_idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"model": model, "strategy": strategy, "policy": policy,
                         "cost": this.mean(), "baseline_cost": base.mean(), "diff": d.mean(),
                         "ci_low": lo, "ci_high": hi,
                         "verdict": "cheaper" if hi < 0 else ("costlier" if lo > 0 else "no clear diff")})
    return pd.DataFrame(rows)
