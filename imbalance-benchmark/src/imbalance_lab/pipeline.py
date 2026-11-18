"""
組裝完整的建模流程 (Pipeline)
=============================

每一個「模型 × 策略」組合都被組成同一種 4 步驟 Pipeline：

    原始特徵
      │
      ▼
    ① scale   ：數值欄標準化 (z-score)，類別欄原封不動
      │
      ▼
    ② sampler ：重採樣（只在 fit 時執行；predict 時自動跳過）
      │
      ▼
    ③ encode  ：類別欄 one-hot 編碼
      │
      ▼
    ④ model   ：分類器

為什麼順序是「先標準化 → 重採樣 → 再 one-hot」？
- SMOTE / Tomek 都靠「距離」找鄰居。沒標準化的話，信用額度 (LIMIT_BAL，數十萬)
  會完全主宰距離，年齡、繳款狀態幾乎沒影響 → 所以要先標準化。
- SMOTENC 需要知道哪幾欄是類別，而且類別欄要維持「代碼」形式，
  所以 one-hot 必須放在重採樣之後。

為什麼一定要用 Pipeline？
因為交叉驗證時，scaler 的平均數/標準差、sampler 的抽樣，都必須「只看訓練折」。
把它們全部包進 Pipeline，再交給 cross-validation，sklearn 會保證每一折都重新 fit，
驗證折的資訊不會流進訓練過程。
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import Dataset
from .models import build_model
from .strategies import build_sampler


def build_pipeline(ds: Dataset, model: str, strategy: str, seed: int,
                   params: dict | None = None, n_jobs: int = 1) -> Pipeline:
    """
    建立一條完整 Pipeline。

    參數
    ----
    ds       : 資料集（需要它的欄位資訊：哪些是數值、哪些是類別）
    model    : 模型名稱，見 models.MODEL_DISPLAY
    strategy : 策略名稱，見 strategies.STRATEGY_DESCRIPTIONS
    params   : 調參結果，例如 {"model__C": 0.5}；None 表示用預設值
    """
    n_num, n_cat = len(ds.numeric), len(ds.categorical)

    # ① 數值欄標準化；類別欄 passthrough（直接通過）。
    #    ColumnTransformer 輸出的欄位順序 = 數值欄在前、類別欄在後
    scale = ColumnTransformer(
        [("num", StandardScaler(), ds.numeric),
         ("cat", "passthrough", ds.categorical)],
        verbose_feature_names_out=False,
    )

    # ② 經過 ① 之後，類別欄位在矩陣的最後 n_cat 欄
    cat_idx = list(range(n_num, n_num + n_cat))
    sampler = build_sampler(strategy, cat_idx, seed)

    # ③ one-hot：把類別代碼展開成 0/1 欄位；沒有類別欄就什麼都不做
    #    handle_unknown="ignore"：驗證集若出現訓練集沒看過的代碼，不會報錯
    if n_cat:
        encode = ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx)],
            remainder="passthrough",
        )
    else:
        encode = "passthrough"

    # ④ 分類器；strategy == "class_weight" 時開啟類別加權
    clf = build_model(model, class_weight=(strategy == "class_weight"), seed=seed, n_jobs=n_jobs)

    steps = [("scale", scale)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps += [("encode", encode), ("model", clf)]

    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe
