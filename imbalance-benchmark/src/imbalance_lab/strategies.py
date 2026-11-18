"""
不平衡處理策略 (imbalance-handling strategies)
==============================================

處理類別不平衡大致有三種思路，這裡各挑代表：

| 策略            | 類型     | 做法                                                   |
|-----------------|----------|--------------------------------------------------------|
| none            | 基準線   | 什麼都不做（所有方法都要跟它比，才知道有沒有幫助）     |
| class_weight    | 改損失   | 不動資料，訓練時讓少數類「答錯的懲罰」變大            |
| random_under    | 欠採樣   | 隨機丟掉多數類，直到各類一樣多                          |
| tomek           | 欠採樣   | 只刪除「緊貼著少數類」的多數類樣本，讓邊界更清楚        |
| smote           | 過採樣   | 在少數類樣本之間內插，合成新的少數類樣本                |

另外「調整決策門檻」也是一種處理方式，但它是在模型訓練完之後才做，
放在 cost.py 裡（成本導向的門檻選擇）。

⚠ 重要：重採樣只能用在「訓練資料」上。
本模組只負責「產生 sampler 物件」，真正套用的時機交給 imbalanced-learn 的
Pipeline：Pipeline 在 fit() 時才會重採樣，predict() 時會自動跳過 sampler，
所以驗證/測試資料永遠保持原始分布 —— 這就是避免資料洩漏 (data leakage) 的關鍵。
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# 每個策略的一句話說明（會輸出到結果表和 README）
STRATEGY_DESCRIPTIONS = {
    "none": "No resampling, no weighting (baseline)",
    "class_weight": "Cost-sensitive loss: minority errors weighted by inverse class frequency",
    "random_under": "Random undersampling of every non-minority class to the minority size",
    "tomek": "Remove majority samples that form Tomek links with minority samples",
    "smote": "SMOTE oversampling (SMOTENC when nominal categorical features exist)",
}

# 需要放一個 sampler 進 Pipeline 的策略
RESAMPLING_STRATEGIES = {"random_under", "tomek", "smote"}


def build_sampler(strategy: str, categorical_idx: list[int], seed: int):
    """
    依策略名稱回傳 sampler；不需要重採樣的策略（none / class_weight）回傳 None。

    參數
    ----
    categorical_idx : 名目類別欄位在「前處理後矩陣」中的位置（index）。
        SMOTE 的原理是「在兩個樣本之間取中間點」，對數值欄位沒問題，
        但對類別欄位會產生「性別 = 1.37」這種不存在的值。
        SMOTENC 會對類別欄位改用「鄰居中最常見的類別」，避免這個問題。
        （舊版專案直接用 SMOTE，就是這個錯誤。）
    """
    if strategy in ("none", "class_weight"):
        return None
    if strategy == "random_under":
        # sampling_strategy="auto"：所有非少數類都砍到跟少數類一樣多
        return RandomUnderSampler(sampling_strategy="auto", random_state=seed)
    if strategy == "tomek":
        # Tomek link：兩個不同類別的樣本互為最近鄰 → 刪掉其中多數類那一個
        # 它只清「邊界上的雜訊」，砍的數量很少，所以不會讓資料變平衡
        return TomekLinks(sampling_strategy="auto")
    if strategy == "smote":
        if categorical_idx:
            return SMOTENC(categorical_features=categorical_idx, random_state=seed)
        return SMOTE(random_state=seed)
    raise KeyError(f"未知的策略 {strategy!r}，可用：{list(STRATEGY_DESCRIPTIONS)}")
