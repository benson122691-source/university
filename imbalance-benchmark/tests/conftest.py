"""
測試用的共用資料 (fixtures)。

測試一律使用「人造資料」(synthetic data)：
- 不依賴真實資料檔，CI（GitHub Actions）上也能跑
- 跑得快（幾秒內完成）
- 我們知道「正確答案」，比較容易驗證
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from imbalance_lab.data import Dataset


@pytest.fixture
def binary_ds() -> Dataset:
    """1,200 筆、IR ≈ 4 的二分類資料，含 2 個數值欄 + 1 個名目類別欄。"""
    X, y = make_classification(n_samples=1200, n_features=6, n_informative=4,
                               weights=[0.8, 0.2], random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    rng = np.random.default_rng(0)
    df["color"] = rng.choice([1, 2, 3], size=len(df))   # 名目類別：代碼 1/2/3
    return Dataset(key="toy_bin", display="toy binary", X=df, y=y,
                   class_names=["neg", "pos"], categorical=["color"])


@pytest.fixture
def multiclass_ds() -> Dataset:
    X, y = make_classification(n_samples=900, n_features=5, n_informative=4, n_redundant=0,
                               n_classes=3, weights=[0.6, 0.3, 0.1], random_state=1)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    return Dataset(key="toy_mc", display="toy multiclass", X=df, y=y,
                   class_names=["a", "b", "c"])
