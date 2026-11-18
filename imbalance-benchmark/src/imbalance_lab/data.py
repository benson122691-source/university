"""
資料載入與清理 (data loading & cleaning)
=========================================

這個模組負責三件事：
1. 找到資料檔（本機 data/ 資料夾；沒有的話嘗試從 UCI 下載）
2. 驗證檔案沒被改過（比對 SHA-256 雜湊值）
3. 清理資料，輸出統一格式的 `Dataset` 物件給後面的實驗使用

另外提供 `make_imbalanced()`：把二分類資料的少數類「人為抽少」，
製造指定的不平衡比 (imbalance ratio, IR)，用於受控實驗。
"""

from __future__ import annotations

import hashlib
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# 專案根目錄下的 data/ 資料夾（data.py 位於 src/imbalance_lab/，往上三層就是專案根目錄）
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# 資料集規格：每份資料集的「說明書」
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    key: str                 # 程式內部用的名稱，例如 "credit_card"
    display: str             # 圖表上顯示的名稱
    filename: str            # data/ 底下的檔名
    sha256: str              # 官方檔案的雜湊值，用來確認檔案完整、沒被竄改
    uci_url: str             # 找不到檔案時的下載網址（UCI 官方 zip）
    read_kwargs: dict = field(default_factory=dict)  # 傳給 pd.read_excel 的額外參數


SPECS: dict[str, DatasetSpec] = {
    "credit_card": DatasetSpec(
        key="credit_card",
        display="Taiwan Credit Card Default",
        filename="default of credit card clients.xls",
        sha256="5cbcb12c989ec42c3ad8fdc315380a128c00fe9521e388d3f16e1ef9b2e32189",
        uci_url="https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        read_kwargs={"header": 1},  # 這個 .xls 第一列是說明文字，真正的欄名在第二列
    ),
    "dry_bean": DatasetSpec(
        key="dry_bean",
        display="Dry Bean",
        filename="Dry_Bean_Dataset.xlsx",
        sha256="a9efa69741c6c5d95167c962a13da23a456f7e940402a5add12639790e01f714",
        uci_url="https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
    ),
}


# ---------------------------------------------------------------------------
# 清理後的統一格式
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """實驗用的資料集。後面所有模組都只吃這個格式。"""

    key: str
    display: str
    X: pd.DataFrame                  # 特徵（欄位順序：數值欄在前、類別欄在後）
    y: np.ndarray                    # 標籤，已轉成整數 0..K-1
    class_names: list[str]           # y 的整數對應到的原始類別名稱
    categorical: list[str] = field(default_factory=list)  # 名目類別欄位（不能拿來做數值內插）

    @property
    def task(self) -> str:
        """'binary'（二分類）或 'multiclass'（多分類）。"""
        return "binary" if len(self.class_names) == 2 else "multiclass"

    @property
    def numeric(self) -> list[str]:
        return [c for c in self.X.columns if c not in self.categorical]

    @property
    def imbalance_ratio(self) -> float:
        """不平衡比 = 最多類別的筆數 / 最少類別的筆數。"""
        counts = np.bincount(self.y)
        return float(counts.max() / counts.min())


# ---------------------------------------------------------------------------
# 找檔案 / 驗證 / 下載
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 每次讀 1 MB，大檔案也不會吃光記憶體
            h.update(chunk)
    return h.hexdigest()


def _locate_file(spec: DatasetSpec, data_dir: Path) -> Path:
    """回傳資料檔路徑。本機找不到就從 UCI 下載並解壓。"""
    data_dir.mkdir(parents=True, exist_ok=True)

    # 先在 data/ 底下（含子資料夾）找同名檔案
    hits = sorted(data_dir.rglob(spec.filename))
    if hits:
        return hits[0]

    # 找不到 → 下載官方 zip
    zip_path = data_dir / f"{spec.key}.zip"
    print(f"[data] 本機沒有 {spec.filename}，從 UCI 下載中 ...")
    try:
        urllib.request.urlretrieve(spec.uci_url, zip_path)
    except OSError as e:  # 網路不通、被防火牆擋等
        raise FileNotFoundError(
            f"找不到 {spec.filename}，下載也失敗（{e}）。\n"
            f"請手動下載 {spec.uci_url} 並把 {spec.filename} 放到 {data_dir}/"
        ) from e
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir / spec.key)
    hits = sorted(data_dir.rglob(spec.filename))
    if not hits:
        raise FileNotFoundError(f"zip 裡找不到 {spec.filename}")
    return hits[0]


def _verify(path: Path, spec: DatasetSpec) -> None:
    """雜湊值不符時只發警告不中斷：讓使用者知道結果可能跟 README 不同。"""
    actual = _sha256(path)
    if actual != spec.sha256:
        warnings.warn(
            f"{path.name} 的 SHA-256 與官方版本不同，結果可能與 README 不一致。",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# 各資料集的清理邏輯
# ---------------------------------------------------------------------------
def _clean_credit_card(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    台灣信用卡違約資料的清理。

    已知的資料問題（官方說明文件沒提到，但資料裡真的有）：
    - EDUCATION 只定義了 1~4（研究所/大學/高中/其他），資料裡卻出現 0、5、6
      → 全部歸到 4「其他」
    - MARRIAGE 只定義了 1~3（已婚/單身/其他），資料裡有 0
      → 歸到 3「其他」
    - 還款狀態欄位名稱從 PAY_0 直接跳到 PAY_2（沒有 PAY_1）
      → 把 PAY_0 改名成 PAY_1，讓 PAY_1..PAY_6 對應最近 1~6 個月
    """
    df = df.drop(columns=["ID"]).rename(
        columns={"PAY_0": "PAY_1", "default payment next month": "default"}
    )
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    y = df.pop("default")
    # SEX / EDUCATION / MARRIAGE 是「名目」類別：數字只是代碼，2 不代表比 1「多」
    # PAY_x 雖然是代碼，但有順序意義（延遲月數），所以當作數值處理
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    return df, y, categorical, ["no_default", "default"]


def _clean_dry_bean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Dry Bean 資料本身很乾淨：16 個數值形狀特徵，沒有缺值。"""
    y = df.pop("Class")
    class_names = sorted(y.unique())
    return df, y, [], class_names


_CLEANERS = {"credit_card": _clean_credit_card, "dry_bean": _clean_dry_bean}


def load_dataset(key: str, data_dir: Path | str | None = None) -> Dataset:
    """
    載入並清理資料集。

    >>> ds = load_dataset("credit_card")
    >>> ds.X.shape, ds.task
    ((30000, 23), 'binary')
    """
    if key not in SPECS:
        raise KeyError(f"未知的資料集 {key!r}，可用：{list(SPECS)}")
    spec = SPECS[key]
    path = _locate_file(spec, Path(data_dir) if data_dir else DEFAULT_DATA_DIR)
    _verify(path, spec)

    raw = pd.read_excel(path, **spec.read_kwargs)
    X, y_raw, categorical, class_names = _CLEANERS[key](raw)

    # 把文字標籤轉成整數：class_names[i] 對應整數 i
    # 二分類時 1 一定是「正類」（違約），這對 precision/recall 的定義很重要
    if key == "credit_card":
        y = y_raw.to_numpy().astype(int)
    else:
        mapping = {name: i for i, name in enumerate(class_names)}
        y = y_raw.map(mapping).to_numpy().astype(int)

    # 欄位排序：數值欄在前、類別欄在後。前處理 (preprocessing) 會依賴這個順序
    numeric = [c for c in X.columns if c not in categorical]
    X = X[numeric + categorical].reset_index(drop=True)

    return Dataset(key=key, display=spec.display, X=X, y=y,
                   class_names=list(class_names), categorical=categorical)


# ---------------------------------------------------------------------------
# 受控不平衡：人為製造指定的 IR
# ---------------------------------------------------------------------------
def make_imbalanced(ds: Dataset, ratio: float, seed: int = 0) -> Dataset:
    """
    從二分類資料中「隨機丟掉部分少數類」，讓 多數類 : 少數類 = ratio : 1。

    為什麼要這樣做？
    舊版專案用「兩份不同資料集」比較不平衡程度的影響，但兩份資料除了 IR 不同，
    類別數、特徵、領域也都不同，根本分不出是哪個因素造成差異。
    這裡固定同一份資料、只改變 IR，其他條件完全相同 → 才是乾淨的對照實驗。

    多數類全部保留，只抽少少數類，所以 ratio 不能比原始 IR 還小。
    """
    if ds.task != "binary":
        raise ValueError("make_imbalanced 只支援二分類資料")

    counts = np.bincount(ds.y)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    n_major = counts[majority]
    n_minor_target = int(round(n_major / ratio))
    if n_minor_target > counts[minority]:
        # 容許 1% 的誤差：例如原始 IR=3.52，要求 3.5 時就直接保留全部少數類
        if ratio >= 0.99 * ds.imbalance_ratio:
            n_minor_target = int(counts[minority])
        else:
            raise ValueError(
                f"ratio={ratio} 比原始 IR={ds.imbalance_ratio:.2f} 還小，無法只靠抽樣達成")

    rng = np.random.default_rng(seed)
    idx_minor = np.flatnonzero(ds.y == minority)
    idx_major = np.flatnonzero(ds.y == majority)
    keep_minor = rng.choice(idx_minor, size=n_minor_target, replace=False)
    keep = np.sort(np.concatenate([idx_major, keep_minor]))  # 排序：維持原本的列順序

    return Dataset(
        key=f"{ds.key}_ir{ratio:g}",
        display=f"{ds.display} (IR={ratio:g})",
        X=ds.X.iloc[keep].reset_index(drop=True),
        y=ds.y[keep],
        class_names=ds.class_names,
        categorical=ds.categorical,
    )
