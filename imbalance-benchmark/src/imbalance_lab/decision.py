"""
多分類的「決策規則」分析 (multiclass decision rule)
================================================

二分類可以「調門檻」，多分類沒有單一門檻可調 —— 預設規則是「選機率最大的類別」(argmax)。
在極度不平衡的多分類資料上（Yeast 最小類只有 5 筆），argmax 幾乎永遠不會選到稀有類別，
這時 class weight / SMOTE 的效果，跟二分類一樣，可能主要是「把決策推向稀有類別」。

多分類版的「調門檻」= 先驗修正 (prior adjustment)：
    把模型輸出的機率除以「各類別在訓練集的比例」，再取 argmax：
        ŷ = argmax_k  p(k | x) / π_k
    等於問：「相對於這個類別本來就多常見，這筆樣本像它的程度有多高？」
    這正是 balanced（各類別一樣重要）的貝氏決策規則 —— 不用重新訓練、不用丟資料。

這裡比較（都用訓練集的 out-of-fold 預測，不碰 test）：
    none (argmax)、none + 先驗修正、class weight、SMOTE、random under
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


def prior_adjust(proba: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """p(k|x) / π_k，再重新正規化成總和 1（argmax 不受正規化影響，只是讓它仍像機率）。"""
    adj = proba / prior
    return adj / adj.sum(axis=1, keepdims=True)


def _scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "min_recall": recall_score(y, pred, average=None, zero_division=0).min()}


def decision_rule_table(oof: dict, y_train: np.ndarray, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """
    每個模型：各種做法的 macro-F1 / 最差類別 recall（OOF 預測彙總），
    並用配對 bootstrap 算「與 none + 先驗修正」差異的 95% 信賴區間。
    """
    prior = np.bincount(y_train, minlength=next(iter(oof.values())).shape[1]) / len(y_train)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(y_train), size=(n_boot, len(y_train)))
    rows = []
    for model in sorted({m for m, _ in oof}):
        # balanced_rf 的每棵樹本來就在「平衡過的」樣本上訓練，機率已經是平衡的；
        # 再除一次先驗等於重複修正，沒有意義 → 排除
        if (model, "none") not in oof or model == "balanced_rf":
            continue
        variants = {"none + prior adjustment": prior_adjust(oof[(model, "none")], prior).argmax(1)}
        for s in ("none", "class_weight", "smote", "random_under", "tomek"):
            if (model, s) in oof:
                variants[s] = oof[(model, s)].argmax(1)
        ref = variants["none + prior adjustment"]
        for name, pred in variants.items():
            row = {"model": model, "variant": name, **_scores(y_train, pred)}
            if name != "none + prior adjustment":
                # 配對 bootstrap：同一組抽樣同時算兩種做法，再看差異的分布
                diffs = np.array([[a - b for a, b in zip(_scores(y_train[i], pred[i]).values(),
                                                         _scores(y_train[i], ref[i]).values(), strict=True)]
                                  for i in boot_idx[:500]])  # 500 次即可：每次要算整份 F1，比較慢
                for j, metric in enumerate(("f1_macro", "min_recall")):
                    lo, hi = np.percentile(diffs[:, j], [2.5, 97.5])
                    row[f"{metric}_diff_ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
            rows.append(row)
    return pd.DataFrame(rows)
