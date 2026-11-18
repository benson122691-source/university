"""
評估指標 (evaluation metrics)
=============================

舊版只看 macro F1 和 ROC-AUC，對不平衡問題不夠：

二分類（信用卡違約）——我們真正關心的是「違約」這個正類：
- pr_auc        Precision-Recall 曲線下面積（= average precision）。
                不平衡時比 ROC-AUC 更誠實：ROC 的 FPR 分母是龐大的負類，
                誤報再多 FPR 都很小，曲線看起來很漂亮；PR 曲線不會被負類數量稀釋。
- roc_auc       排序能力（隨機挑一個違約者、一個正常戶，模型給違約者較高分的機率）
- recall        抓到的違約者 / 全部違約者
- precision     判為違約的人中，真的違約的比例
- f1            precision 與 recall 的調和平均（只算正類，不是 macro）
- mcc           Matthews 相關係數：-1~1，綜合混淆矩陣四格，對不平衡最穩健
- balanced_acc  各類 recall 的平均
- brier         機率預測的均方誤差，越小越好 —— 用來看「機率準不準」（校準）
- f1_best       所有門檻中「最好的那個 F1」。它不依賴 0.5 門檻，
                用來分辨：策略提升 F1，是因為模型真的變好，還是只是「門檻剛好移到對的位置」？
                （注意：這是在同一份資料上選門檻，是樂觀的上限，只拿來比較，不當作實際表現）

多分類（Dry Bean）：
- f1_macro、balanced_acc、mcc、roc_auc（one-vs-rest 平均）
- min_recall    最差那一類的 recall：不平衡處理最想改善的就是弱勢類別

注意：recall / precision / f1 / mcc 這些「硬分類」指標都依賴門檻。
這裡統一用 0.5（二分類）或「機率最大的類別」（多分類），
門檻的最佳化另外在 cost.py 處理。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 各任務的「主要指標」：結果表、統計檢定、排名都以它為準
PRIMARY_METRIC = {"binary": "pr_auc", "multiclass": "f1_macro"}

# 越小越好的指標（排名時要反過來）
LOWER_IS_BETTER = {"brier"}


def binary_metrics(y_true: np.ndarray, p_pos: np.ndarray, threshold: float = 0.5) -> dict:
    """p_pos：模型預測「是正類（違約）」的機率。"""
    y_pred = (p_pos >= threshold).astype(int)
    # 每個可能門檻下的 precision / recall → 算出 F1，取最大值
    prec, rec, _ = precision_recall_curve(y_true, p_pos)
    f1_curve = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    return {
        "pr_auc": average_precision_score(y_true, p_pos),
        "roc_auc": roc_auc_score(y_true, p_pos),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, p_pos),
        "f1_best": float(f1_curve.max()),
    }


def multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """proba：形狀 (n 筆, K 類) 的機率矩陣。"""
    y_pred = proba.argmax(axis=1)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "min_recall": float(per_class_recall.min()),
    }


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, task: str) -> dict:
    """統一入口：proba 一律是 predict_proba 的完整輸出 (n, K)。"""
    if task == "binary":
        return binary_metrics(y_true, proba[:, 1])
    return multiclass_metrics(y_true, proba)
