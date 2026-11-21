"""
模型定義與超參數搜尋空間 (models & search spaces)
================================================

比較的模型：
- logreg        邏輯斯迴歸：信用評分業界最常用的基準，可解釋性高
- knn           K 近鄰：純粹靠距離，對資料分布變化最敏感，適合觀察重採樣的效果
- random_forest 隨機森林
- xgboost       梯度提升樹：表格資料的強力基準
- balanced_rf   平衡隨機森林：每棵樹在自己的 bootstrap 樣本裡先欠採樣，
                是「欠採樣 + 集成學習」的代表，不會整體浪費資料

為什麼拿掉舊版的 SVM？
RBF-SVM 的訓練時間大約隨樣本數平方成長；舊版在 SMOTE 後的 3.7 萬筆上
單次就要約 39 秒，放進 15 折重複 CV × 多種策略的實驗設計會讓總時間暴增，
而「線性模型」的角色已由邏輯斯迴歸擔任。
"""

from __future__ import annotations

from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

MODEL_DISPLAY = {
    "logreg": "Logistic Regression",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "balanced_rf": "Balanced RF",
}


class BalancedXGBClassifier(XGBClassifier):
    """
    支援 class weighting 的 XGBoost。

    XGBoost 沒有 sklearn 那種 class_weight="balanced" 參數；
    二分類可以用 scale_pos_weight，但多分類不行。
    最通用的做法是給每一筆樣本一個權重：權重 = 1 / 該類別的出現頻率。
    這個子類別在 fit() 時自動算好權重再交給原本的 XGBoost。
    """

    def fit(self, X, y, **kwargs):
        kwargs.setdefault("sample_weight", compute_sample_weight("balanced", y))
        return super().fit(X, y, **kwargs)


class SafeKNeighborsClassifier(KNeighborsClassifier):
    """
    鄰居數超過訓練筆數時自動縮小的 KNN。

    為什麼需要？Yeast 有 10 類、最小類只有 3 筆（在訓練折裡），
    隨機欠採樣會把每一類都砍到 3 筆 → 只剩 30 筆訓練資料，
    但調參選出的 n_neighbors 可能是 34 → 原本的 KNN 會直接報錯。
    這裡在「找鄰居」時把 k 限制在訓練筆數以內，設定值本身不會被改動。
    """

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        k = n_neighbors or self.n_neighbors
        k = min(k, self.n_samples_fit_ - (1 if X is None else 0))  # X=None 時不含自己
        return super().kneighbors(X, n_neighbors=k, return_distance=return_distance)


def is_compatible(model: str, strategy: str) -> bool:
    """
    判斷「模型 × 策略」這個組合是否有意義，不合理的組合直接跳過：
    - KNN 沒有損失函數可以加權 → 不能用 class_weight
    - balanced_rf 自己內建欠採樣 → 只跟 none 搭配，再加別的策略就重複了
    """
    if model == "knn" and strategy == "class_weight":
        return False
    if model == "balanced_rf" and strategy != "none":
        return False
    return True


def build_model(model: str, class_weight: bool, seed: int, n_jobs: int = 1):
    """
    建立一個「預設參數」的模型；調參後的參數會由 Pipeline.set_params 再覆寫。

    n_jobs 預設 1：平行化放在外層（同時跑多個 CV 折），
    避免外層和內層同時開多執行緒互相搶 CPU。
    """
    cw = "balanced" if class_weight else None
    if model == "logreg":
        return LogisticRegression(max_iter=3000, class_weight=cw)
    if model == "knn":
        return SafeKNeighborsClassifier(n_neighbors=15, n_jobs=n_jobs)
    if model == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight=cw,
                                      random_state=seed, n_jobs=n_jobs)
    if model == "xgboost":
        cls = BalancedXGBClassifier if class_weight else XGBClassifier
        return cls(n_estimators=300, tree_method="hist", random_state=seed,
                   n_jobs=n_jobs, verbosity=0)
    if model == "balanced_rf":
        # sampling_strategy="all"、replacement=True、bootstrap=False
        # 是 imbalanced-learn 官方建議的設定（也是未來版本的預設值）
        return BalancedRandomForestClassifier(
            n_estimators=200, sampling_strategy="all", replacement=True,
            bootstrap=False, random_state=seed, n_jobs=n_jobs,
        )
    raise KeyError(f"未知的模型 {model!r}，可用：{list(MODEL_DISPLAY)}")


# ---------------------------------------------------------------------------
# 超參數搜尋空間
# 參數名稱前綴 "model__" 代表「Pipeline 裡名為 model 的那一步」的參數
# ---------------------------------------------------------------------------
_TREE_SPACE = {
    "model__max_depth": [None, 8, 12, 16, 24],
    "model__min_samples_leaf": [1, 2, 5, 10, 20],
    "model__max_features": ["sqrt", "log2", 0.5],
}

SEARCH_SPACES = {
    "logreg": {"model__C": loguniform(1e-3, 1e2)},          # C 越小 = 正則化越強
    "knn": {
        "model__n_neighbors": randint(5, 76),                 # 5 ~ 75 個鄰居
        "model__weights": ["uniform", "distance"],            # 鄰居投票是否依距離加權
    },
    "random_forest": _TREE_SPACE,
    "balanced_rf": _TREE_SPACE,
    "xgboost": {
        "model__n_estimators": randint(100, 501),
        "model__learning_rate": loguniform(0.02, 0.3),
        "model__max_depth": randint(3, 9),
        "model__min_child_weight": randint(1, 11),
        "model__subsample": uniform(0.6, 0.4),                # 0.6 ~ 1.0
        "model__colsample_bytree": uniform(0.6, 0.4),
    },
}
