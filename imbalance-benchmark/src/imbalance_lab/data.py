"""
資料載入與清理 (data loading & cleaning)
=========================================

這個模組負責三件事：
1. 找到資料檔：本機 data/ 資料夾 → 沒有的話依序嘗試下載來源（官方 / GitHub 鏡像）
2. 驗證檔案沒被改過：比對 SHA-256 雜湊值
3. 清理資料，輸出統一格式的 `Dataset` 物件給後面的實驗使用

另外提供 `make_imbalanced()`：把二分類資料的少數類「人為抽少」，
製造指定的不平衡比 (imbalance ratio, IR)，用於受控實驗。

資料集分兩組：
- 主要資料集 (primary)：credit_card、dry_bean —— 做完整分析（含成本、校準、IR 實驗）
- 驗證資料集 (validation)：另外 5 份不同領域、不同不平衡程度的資料，
  用「同一套流程」重跑，檢驗主要結論是否只是那兩份資料的特例
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import warnings
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# 專案根目錄下的 data/ 資料夾（data.py 位於 src/imbalance_lab/，往上三層就是專案根目錄）
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# 清理後的統一格式
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """實驗用的資料集。後面所有模組都只吃這個格式。"""

    key: str
    display: str
    X: pd.DataFrame                  # 特徵（欄位順序：數值欄在前、類別欄在後；類別欄已轉成整數代碼）
    y: np.ndarray                    # 標籤，已轉成整數 0..K-1；二分類時 1 = 我們關心的「正類」
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


# 清理函式的統一介面：吃原始 DataFrame，吐出 (X, y, 類別欄名單, 類別名稱)
Cleaned = tuple[pd.DataFrame, np.ndarray, list[str], list[str]]


# ---------------------------------------------------------------------------
# 資料集規格：每份資料集的「說明書」
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    key: str                         # 程式內部用的名稱，例如 "credit_card"
    display: str                     # 圖表上顯示的名稱
    filename: str                    # data/ 底下的檔名
    sha256: str                      # 檔案雜湊值：確認拿到的跟作者用的是同一份
    sources: tuple[str, ...]         # 下載來源，依序嘗試；.zip 會自動解壓找 filename
    reader: Callable[[Path], pd.DataFrame]
    cleaner: Callable[[pd.DataFrame], Cleaned]
    domain: str = ""                 # 領域說明（給 README / 圖表用）
    reference: str = ""              # 原始出處


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 每次讀 1 MB，大檔案也不會吃光記憶體
            h.update(chunk)
    return h.hexdigest()


def _codes(s: pd.Series) -> pd.Series:
    """
    把文字類別轉成整數代碼（依字母排序：例如 'married'→1、'single'→2）。
    代碼只是「名字」，大小沒有意義；後面的 Pipeline 會再做 one-hot。
    之所以先轉整數，是因為標準化、SMOTENC 等步驟需要純數值矩陣。
    """
    cats = sorted(s.astype(str).unique())
    return s.astype(str).map({c: i for i, c in enumerate(cats)}).astype(int)


def _encode_labels(y_raw: pd.Series, positive=None) -> tuple[np.ndarray, list[str]]:
    """
    文字/數字標籤 → 整數 0..K-1。
    二分類時指定 positive（正類），保證它被編成 1：
    precision / recall / PR-AUC 都是針對「1」計算，正類搞錯整個分析就反了。
    """
    if positive is not None:
        y = (y_raw == positive).astype(int).to_numpy()
        names = sorted(str(v) for v in y_raw.unique() if v != positive) + [str(positive)]
        return y, names
    names = sorted(str(v) for v in y_raw.unique())
    mapping = {n: i for i, n in enumerate(names)}
    return y_raw.astype(str).map(mapping).to_numpy().astype(int), names


# ---------------------------------------------------------------------------
# 各資料集的讀檔 + 清理
# ---------------------------------------------------------------------------
def _clean_credit_card(df: pd.DataFrame) -> Cleaned:
    """
    台灣信用卡違約資料。

    已知的資料問題（官方說明文件沒提到，但資料裡真的有）：
    - EDUCATION 只定義了 1~4（研究所/大學/高中/其他），資料裡卻出現 0、5、6 → 全部歸到 4「其他」
    - MARRIAGE 只定義了 1~3（已婚/單身/其他），資料裡有 0 → 歸到 3「其他」
    - 還款狀態欄位名稱從 PAY_0 直接跳到 PAY_2 → 把 PAY_0 改名成 PAY_1
    """
    df = df.drop(columns=["ID"]).rename(columns={"PAY_0": "PAY_1", "default payment next month": "default"})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    y, names = _encode_labels(df.pop("default"), positive=1)
    # SEX / EDUCATION / MARRIAGE 是「名目」類別；PAY_x 雖是代碼但有順序意義（延遲月數），當數值
    return df, y, ["SEX", "EDUCATION", "MARRIAGE"], ["no_default", "default"]


def _clean_dry_bean(df: pd.DataFrame) -> Cleaned:
    """Dry Bean：16 個數值形狀特徵，沒有缺值。"""
    y, names = _encode_labels(df.pop("Class"))
    return df, y, [], names


def _clean_bank_marketing(df: pd.DataFrame) -> Cleaned:
    """
    葡萄牙銀行電話行銷：預測客戶會不會訂閱定期存款 (y = yes)。

    - 移除 duration（通話秒數）：UCI 官方說明特別警告，通話長度要「打完電話才知道」，
      而且 duration = 0 時一定是 no —— 拿它來預測是典型的資料洩漏 (target leakage)。
    - pdays = 999 代表「之前從未聯絡過」，不是真的 999 天：
      另外加一欄 prev_contacted（0/1），並把 999 改成 -1，避免模型把它當成很大的數字。
    - 'unknown' 保留為一個獨立類別（「不肯回答」本身可能就有資訊）。
    """
    df = df.drop(columns=["duration"])
    df["prev_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)
    cat = ["job", "marital", "education", "default", "housing", "loan",
           "contact", "month", "day_of_week", "poutcome"]
    for c in cat:
        df[c] = _codes(df[c])
    y, names = _encode_labels(df.pop("y"), positive="yes")
    return df, y, cat, names


# German Credit 的欄位名稱（UCI 文件 Attribute 1~20）
_GERMAN_COLS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "target",
]


def _clean_german_credit(df: pd.DataFrame) -> Cleaned:
    """
    德國信用評等 (Statlog German Credit)：1 = 好客戶，2 = 壞客戶。

    這份資料特別之處：UCI 官方附了成本矩陣 ——
    把壞客戶誤判成好客戶的成本是 5，把好客戶誤判成壞客戶是 1。
    剛好跟我們信用卡分析「假設」的 C_FN=5、C_FP=1 一樣，可以用來驗證成本分析。
    A11、A12… 這類代碼全部是名目類別。
    """
    df.columns = _GERMAN_COLS
    # 文字欄（A11、A34 …）就是類別欄；數字欄（金額、年齡 …）是數值欄
    cat = [c for c in df.columns if c != "target" and not pd.api.types.is_numeric_dtype(df[c])]
    for c in cat:
        df[c] = _codes(df[c])
    y, _ = _encode_labels(df.pop("target"), positive=2)
    return df, y, cat, ["good", "bad"]


def _clean_mammography(df: pd.DataFrame) -> Cleaned:
    """
    乳房 X 光微鈣化點偵測 (Woods et al., 1993)：6 個已標準化的影像特徵。
    標籤 '1' = 惡性鈣化點（正類，約 2.3%），'-1' = 正常。
    """
    df.columns = [f"f{i}" for i in range(6)] + ["label"]
    label = df.pop("label").astype(str).str.strip("'")
    y, _ = _encode_labels(label, positive="1")
    return df, y, [], ["normal", "calcification"]


def _clean_yeast(df: pd.DataFrame) -> Cleaned:
    """
    酵母菌蛋白質定位 (UCI Yeast)：8 個數值特徵，預測蛋白質在細胞中的位置，共 10 類。
    最大類 CYT 有 463 筆，最小類 ERL 只有 5 筆（IR ≈ 93）—— 極端多分類不平衡。
    第一欄是序列名稱（每列都不同，沒有預測價值）→ 刪除。
    """
    df.columns = ["name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "site"]
    df = df.drop(columns=["name"])
    y, names = _encode_labels(df.pop("site"))
    return df, y, [], names


def _clean_page_blocks(df: pd.DataFrame) -> Cleaned:
    """
    文件版面區塊分類 (UCI Page Blocks)：10 個區塊形狀特徵，5 類
    （1 文字 / 2 水平線 / 3 圖片 / 4 垂直線 / 5 圖形）。文字區塊佔 90%，圖片只有 28 筆 → IR ≈ 175。
    """
    df.columns = ["height", "length", "area", "eccen", "p_black", "p_and",
                  "mean_tr", "blackpix", "blackand", "wb_trans", "block_type"]
    names_map = {1: "text", 2: "horiz_line", 3: "graphic", 4: "vert_line", 5: "picture"}
    y, names = _encode_labels(df.pop("block_type").map(names_map))
    return df, y, [], names


_WS = {"sep": r"\s+", "header": None}  # 以空白分隔、沒有標題列的 UCI 純文字格式

SPECS: dict[str, DatasetSpec] = {
    # ---------------- 主要資料集 ----------------
    "credit_card": DatasetSpec(
        key="credit_card", display="Taiwan Credit Card Default",
        filename="default of credit card clients.xls",
        sha256="5cbcb12c989ec42c3ad8fdc315380a128c00fe9521e388d3f16e1ef9b2e32189",
        sources=("https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",),
        reader=lambda p: pd.read_excel(p, header=1),  # 第一列是說明文字，欄名在第二列
        cleaner=_clean_credit_card, domain="consumer credit",
        reference="Yeh & Lien (2009), UCI #350",
    ),
    "dry_bean": DatasetSpec(
        key="dry_bean", display="Dry Bean", filename="Dry_Bean_Dataset.xlsx",
        sha256="a9efa69741c6c5d95167c962a13da23a456f7e940402a5add12639790e01f714",
        sources=("https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",),
        reader=pd.read_excel, cleaner=_clean_dry_bean, domain="agriculture / vision",
        reference="Koklu & Ozkan (2020), UCI #602",
    ),
    # ---------------- 驗證資料集 ----------------
    # sources 第二個以後是 GitHub 上的鏡像：UCI 有時連不上，鏡像一樣會驗證 SHA-256
    "bank_marketing": DatasetSpec(
        key="bank_marketing", display="Bank Marketing", filename="bank-additional-full.csv",
        sha256="74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8",
        sources=("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
                 "https://raw.githubusercontent.com/Njuraita/Bank-Marketing-Subscription-Predictor/main/"
                 "data/bank-additional-full.csv"),
        reader=lambda p: pd.read_csv(p, sep=";"),
        cleaner=_clean_bank_marketing, domain="banking / marketing",
        reference="Moro et al. (2014), UCI #222",
    ),
    "german_credit": DatasetSpec(
        key="german_credit", display="German Credit", filename="german.data",
        sha256="b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871",
        sources=("https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
                 "https://raw.githubusercontent.com/tunguz/TabularBenchmarks/main/datasets/credit-g/input/german.data"),
        reader=lambda p: pd.read_csv(p, sep=" ", header=None),
        cleaner=_clean_german_credit, domain="consumer credit (official cost matrix)",
        reference="Hofmann (1994), UCI #144",
    ),
    "mammography": DatasetSpec(
        key="mammography", display="Mammography", filename="mammography.csv",
        sha256="7d3dea3f075f30bbdbb8980e7725684059b4fe3d0f850fadd0f635f6993d8730",
        sources=("https://raw.githubusercontent.com/jbrownlee/Datasets/master/mammography.csv",),
        reader=lambda p: pd.read_csv(p, header=None),
        cleaner=_clean_mammography, domain="medical imaging",
        reference="Woods et al. (1993); OpenML #310",
    ),
    "yeast": DatasetSpec(
        key="yeast", display="Yeast", filename="yeast.data",
        sha256="7cf61776fc04f527f93bf57a327b863893a1225d82df02d457e8950173218258",
        sources=("https://archive.ics.uci.edu/static/public/110/yeast.zip",
                 "https://raw.githubusercontent.com/kingfengji/mGBDT/master/datasets/uci_yeast/yeast.data"),
        reader=lambda p: pd.read_csv(p, **_WS),
        cleaner=_clean_yeast, domain="biology",
        reference="Nakai & Kanehisa (1991), UCI #110",
    ),
    "page_blocks": DatasetSpec(
        key="page_blocks", display="Page Blocks", filename="page-blocks.data",
        sha256="52058b6e19ba797b5e80ae6454ebb3b7a25016787e668dd3203393c236d22c78",
        sources=("https://archive.ics.uci.edu/static/public/78/page+blocks+classification.zip",
                 "https://raw.githubusercontent.com/dimonenka/DEDPUL/master/UCI/pageblock/page-blocks.data"),
        reader=lambda p: pd.read_csv(p, **_WS),
        cleaner=_clean_page_blocks, domain="document analysis",
        reference="Malerba et al. (1995), UCI #78",
    ),
}

PRIMARY_DATASETS = ["credit_card", "dry_bean"]
VALIDATION_DATASETS = ["bank_marketing", "german_credit", "mammography", "yeast", "page_blocks"]


# ---------------------------------------------------------------------------
# 找檔案 / 下載 / 驗證
# ---------------------------------------------------------------------------
def _download(spec: DatasetSpec, url: str, dest_dir: Path) -> Path | None:
    """從一個來源下載。成功回傳檔案路徑，失敗回傳 None（讓呼叫端試下一個來源）。"""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            payload = r.read()
    except OSError as e:  # 網路不通、被防火牆擋、404 等
        print(f"[data]   ✗ {url}  ({e})")
        return None
    target = dest_dir / spec.filename
    if url.endswith(".zip"):
        # 有些 UCI zip 裡面還包一層 zip（例如 bank marketing），所以遞迴找
        def find_in_zip(data: bytes) -> bytes | None:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if name.endswith("/" + spec.filename) or name == spec.filename:
                        return zf.read(name)
                for name in zf.namelist():
                    if name.endswith(".zip"):
                        found = find_in_zip(zf.read(name))
                        if found:
                            return found
            return None

        payload = find_in_zip(payload)
        if payload is None:
            print(f"[data]   ✗ {url}  (zip 裡找不到 {spec.filename})")
            return None
    target.write_bytes(payload)
    return target


def _locate_file(spec: DatasetSpec, data_dir: Path) -> Path:
    """回傳資料檔路徑：先找本機，找不到就依序嘗試每個下載來源。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    hits = sorted(data_dir.rglob(spec.filename))
    if hits:
        return hits[0]
    print(f"[data] 本機沒有 {spec.filename}，嘗試下載 ...")
    for url in spec.sources:
        path = _download(spec, url, data_dir)
        if path is None:
            continue
        if _sha256(path) != spec.sha256:
            # 下載到的內容跟預期不同 → 不可信，刪掉換下一個來源
            print(f"[data]   ✗ {url}  (SHA-256 不符，已刪除)")
            path.unlink()
            continue
        print(f"[data]   ✓ {url}")
        return path
    raise FileNotFoundError(
        f"找不到 {spec.filename}，所有來源都下載失敗。\n請手動下載並放到 {data_dir}/\n來源：{spec.sources}")


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
    if _sha256(path) != spec.sha256:  # 本機檔案被改過：警告但不中斷
        warnings.warn(f"{path.name} 的 SHA-256 與預期不同，結果可能與 README 不一致。", stacklevel=2)

    X, y, categorical, class_names = spec.cleaner(spec.reader(path))

    # 欄位排序：數值欄在前、類別欄在後。前處理 (pipeline.py) 依賴這個順序
    numeric = [c for c in X.columns if c not in categorical]
    X = X[numeric + categorical].reset_index(drop=True).astype(float)

    return Dataset(key=key, display=spec.display, X=X, y=np.asarray(y, dtype=int),
                   class_names=list(class_names), categorical=list(categorical))


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
