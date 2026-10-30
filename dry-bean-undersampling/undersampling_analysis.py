# -*- coding: utf-8 -*-
"""
欠採樣於不平衡分類分析 (Undersampling for Imbalanced Classification)
====================================================================
畢業專題（期末報告）
指導老師：林真如 教授

資料集：Dry Bean Dataset（UCI）
  - 13,611 筆、16 個特徵、7 個類別
  - 類別分布：DERMASON 3,546 (26.05%)、SIRA 2,636 (19.37%)、
             SEKER 2,027 (14.89%)、HOROZ 1,928 (14.17%)、
             CALI 1,630 (11.98%)、BARBUNYA 1,322 (9.71%)、BOMBAY 522 (3.84%)

研究流程（對應簡報）：
  1. 資料選取與類別分布分析
  2. 資料分割：80% 訓練集 / 20% 測試集（分層抽樣）
  3. 標準化：StandardScaler（僅以訓練集擬合，測試集使用相同參數）
  4. 欠採樣：
       方案 A：僅對最多數組欠採樣（可指定比例 1:1 ~ 1:5）
       方案 B：對所有非最少組欠採樣（全部類別降至最少數組數量）
  5. 分類器：KNN、隨機森林、SVM、XGBoost
  6. 方法比較：SMOTE、欠採樣、超參數最佳化、欠採樣+最佳化
  7. 評估指標：Precision、Recall、F1-Score、ROC AUC（macro 平均）

執行方式：
    python undersampling_analysis.py

輸出：
    figures/  — 6 張圖表（PNG）
    results/  — 4 個結果表格（CSV）
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 不開啟視窗，直接存檔
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- 全域設定 ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
LOCAL_XLSX_CANDIDATES = [
    DATA_DIR / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx",
    DATA_DIR / "Dry_Bean_Dataset.xlsx",
]
LOCAL_CSV_CANDIDATES = [
    DATA_DIR / "DryBean_Dataset.csv",
    DATA_DIR / "dry_bean.csv",
]

CLASSIFIER_NAMES = ["KNN", "Random Forest", "SVM", "XGBoost"]

# 方法順序（對應簡報 F1 比較圖與分類成果表）
METHOD_ORDER = [
    ("smote", "SMOTE"),
    ("undersample+optimized", "下採樣+最優化"),
    ("undersample", "下採樣"),
    ("optimized", "最優化"),
]

# 簡報中經最佳化後的分類器參數
OPTIMIZED_PARAMS = {
    "KNN": dict(metric="euclidean", n_neighbors=9, weights="distance"),
    "Random Forest": dict(max_depth=None, max_features="log2", min_samples_leaf=1,
                          min_samples_split=10, n_estimators=100),
    "SVM": dict(C=10, gamma=0.1, kernel="rbf"),
    "XGBoost": dict(colsample_bytree=1.0, learning_rate=0.2, max_depth=6,
                    n_estimators=100, subsample=1.0),
}
DEFAULT_PARAMS = {
    "KNN": dict(n_neighbors=5),
    "Random Forest": dict(n_estimators=100),
    "SVM": dict(kernel="rbf"),
    "XGBoost": dict(n_estimators=100),
}

# 中文圖表字型（Windows 微軟正黑體）
rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Noto Sans CJK TC",
                               "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass


# ---------- 1. 資料取得 ----------
def ensure_dataset() -> pd.DataFrame:
    """優先使用本機 data/ 下的檔案，否則自動從 UCI 下載並解壓。"""
    DATA_DIR.mkdir(exist_ok=True)

    for p in LOCAL_XLSX_CANDIDATES + LOCAL_CSV_CANDIDATES:
        if p.exists():
            print(f"[資料] 使用本機檔案：{p}")
            if p.suffix == ".csv":
                return pd.read_csv(p)
            return pd.read_excel(p)

    print("[資料] 本機無資料檔，從 UCI 下載 Dry Bean Dataset ...")
    import urllib.request
    import zipfile

    zip_path = DATA_DIR / "dry+bean+dataset.zip"
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
    print(f"[資料] 下載完成：{zip_path.name}（{zip_path.stat().st_size / 1e6:.1f} MB）")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    for p in LOCAL_XLSX_CANDIDATES:
        if p.exists():
            return pd.read_excel(p)
    raise FileNotFoundError("無法在解壓後的資料夾中找到 Dry_Bean_Dataset.xlsx")


# ---------- 2. 欠採樣輔助函式 ----------
def scheme_a_undersample(X: pd.DataFrame, y: pd.Series, ratio: float = 1.0,
                         random_state: int = RANDOM_STATE):
    """方案 A：僅對最多數組欠採樣。
    最多數類別的目標數量 = 最少數類別數量 × ratio（例如 1:1 ~ 1:5）。
    """
    counts = y.value_counts()
    minority = counts.idxmin()
    majority = counts.idxmax()
    n_min = int(counts.min())
    target = int(round(n_min * ratio))
    target = min(target, int(counts.max()))

    rng = np.random.RandomState(random_state)
    idx = np.arange(len(y))
    keep = []
    for cls in counts.index:
        cls_idx = idx[y.to_numpy() == cls]
        if cls == majority:
            keep.append(rng.choice(cls_idx, size=target, replace=False))
        else:
            keep.append(cls_idx)
    keep = np.concatenate(keep)
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def scheme_b_undersample(X: pd.DataFrame, y: pd.Series,
                         random_state: int = RANDOM_STATE):
    """方案 B：對所有非最少組欠採樣（全部類別降至最少數組數量）。"""
    counts = y.value_counts()
    n_min = int(counts.min())
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(y))
    keep = []
    for cls in counts.index:
        cls_idx = idx[y.to_numpy() == cls]
        keep.append(rng.choice(cls_idx, size=n_min, replace=False))
    keep = np.concatenate(keep)
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """SMOTE 過採樣（合成少數類樣本）。"""
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=random_state)
    Xr, yr = smote.fit_resample(X, y)
    return pd.DataFrame(Xr, columns=X.columns), pd.Series(yr)


# ---------- 3. 分類器 ----------
def build_classifier(name: str, optimized: bool):
    """建立分類器；optimized=True 時使用簡報最佳化參數。"""
    params = OPTIMIZED_PARAMS[name] if optimized else DEFAULT_PARAMS[name]
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params, n_jobs=-1)
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "SVM":
        from sklearn.svm import SVC
        from sklearn.multiclass import OneVsRestClassifier
        return OneVsRestClassifier(
            SVC(**params, random_state=RANDOM_STATE), n_jobs=-1
        )
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1,
                             verbosity=0)
    raise ValueError(f"未知分類器：{name}")


def get_scores(model, X) -> np.ndarray:
    """回傳 (n, n_classes) 的分數矩陣（機率或決策函數）。"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def compute_auc(y_true: np.ndarray, scores: np.ndarray, classes: list) -> float:
    """逐類別二元 ROC AUC 之 macro 平均（支援機率或決策函數分數）。"""
    from sklearn.metrics import roc_auc_score as ras
    from sklearn.preprocessing import label_binarize

    classes_sorted = sorted(classes)
    y_bin = label_binarize(y_true, classes=classes_sorted)
    aucs = []
    for i in range(len(classes_sorted)):
        n_pos = int(y_bin[:, i].sum())
        if n_pos == 0 or n_pos == len(y_bin):  # 該類別在測試集無樣本時略過
            continue
        aucs.append(ras(y_bin[:, i], scores[:, i]))
    return float(np.mean(aucs))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray,
             classes: list) -> dict:
    """計算 macro 平均的 Precision / Recall / F1 / AUC。"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    return {
        "Precision": round(precision_score(y_true, y_pred, average="macro",
                                            labels=classes, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="macro",
                                     labels=classes, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, average="macro",
                             labels=classes, zero_division=0), 4),
        "AUC": round(compute_auc(y_true, scores, classes), 4),
    }


# ---------- 4. 圖表 ----------
def macro_roc_curve(y_true: np.ndarray, scores: np.ndarray, classes: list):
    """以各類別 ROC 插值平均出 macro 曲線。"""
    from sklearn.metrics import roc_curve, auc as roc_auc
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=sorted(classes))
    n_classes = len(classes)
    fpr_grid = np.linspace(0.0, 1.0, 200)
    tpr_list = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_list.append(tpr_interp)
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    return fpr_grid, mean_tpr, roc_auc(fpr_grid, mean_tpr)


def plot_class_distribution(df: pd.DataFrame, classes: list, save_path: Path):
    """類別分布長條圖（含比例）。"""
    counts = df["Class"].value_counts().reindex(classes)
    props = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(counts)), counts.values, color="#4C72B0")
    for i, (c, p) in enumerate(zip(counts.values, props.values)):
        ax.text(i, c, f"{p:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.set_title("Class Distribution in Dry Bean Dataset with Proportions")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc_comparison(models, X_test, y_test, classes, title, save_path):
    """多分類器 macro ROC 曲線比較圖。"""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = ["#55A868", "#4C72B0", "#C44E52", "#8172B2"]
    for (name, model), color in zip(models.items(), colors):
        scores = get_scores(model, X_test)
        fpr, tpr, mean_auc = macro_roc_curve(y_test, scores, classes)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {mean_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("1-Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_f1_method_comparison(result_df: pd.DataFrame, save_path: Path):
    """F1 分數比較折線圖（不同方法 × 分類器）。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"SMOTE": "#F0C419", "下採樣+最優化": "#E67E22",
               "下採樣": "#C0392B", "最優化": "#E91E8C"}
    classifiers = CLASSIFIER_NAMES
    for key, label in METHOD_ORDER:
        sub = result_df[result_df["Method_Key"] == key]
        sub = sub.set_index("Classifier").reindex(classifiers)
        ax.plot(classifiers, sub["F1"], marker="o", lw=2, color=palette[label],
                label=label)
    ax.axhline(0.85, ls="--", color="gray", lw=1, label="0.85 F1 Score")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Classifiers")
    ax.set_ylim(0.80, 1.00)
    ax.set_title("F1 Score Comparison for Different Methods and Classifiers")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_scheme_ab_comparison(df_ab: pd.DataFrame, save_path: Path):
    """方案 A（僅對最多數組）vs 方案 B（對所有非最少組）長條圖。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(df_ab))
    width = 0.36
    ax.bar(x - width / 2, df_ab["SchemeA_F1"], width, label="方案A：僅對最多數組",
           color="#4C72B0")
    ax.bar(x + width / 2, df_ab["SchemeB_F1"], width, label="方案B：對所有非最少組",
           color="#DD8452")
    for xi, v in zip(x - width / 2, df_ab["SchemeA_F1"]):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + width / 2, df_ab["SchemeB_F1"]):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c.lower()}" for c in df_ab["Classifier"]])
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.85, 0.95)
    ax.set_title("樣本最多的類別欠採樣與全部類別欠採樣比較表")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_ratio_analysis(df_ratio: pd.DataFrame, save_path: Path):
    """不同欠採樣比例（1:1 ~ 1:5）F1 分析長條圖。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [f"1:{r}" for r in df_ratio["Ratio"]]
    ax.bar(range(len(df_ratio)), df_ratio["F1"], color="#55A868")
    for i, v in enumerate(df_ratio["F1"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(df_ratio)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_ylim(0.85, 0.95)
    ax.set_title("樣本最多的類別欠採樣比例分析表")
    ax.set_xlabel("樣本最多的類別欠採樣比例")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------- 5. 實驗 ----------
def experiment_method_comparison(X_train, y_train, X_test, y_test,
                                 classes) -> pd.DataFrame:
    """四種分類器 × 四種方法 的 Precision/Recall/F1/AUC 比較。"""
    rows = []
    for name in CLASSIFIER_NAMES:
        for key, label in METHOD_ORDER:
            t0 = time.time()
            use_optimized = key in ("optimized", "undersample+optimized")
            resample = "smote" if key == "smote" else (
                "schemeA" if key.startswith("undersample") else None)

            Xr, yr = X_train, y_train
            if resample == "smote":
                Xr, yr = apply_smote(X_train, y_train)
            elif resample == "schemeA":
                Xr, yr = scheme_a_undersample(X_train, y_train, ratio=1.0)

            model = build_classifier(name, optimized=use_optimized)
            model.fit(Xr, yr)
            y_pred = model.predict(X_test)
            scores = get_scores(model, X_test)
            m = evaluate(y_test.to_numpy(), y_pred, scores, classes)
            dt = round(time.time() - t0, 1)
            rows.append({"Classifier": name, "Method_Key": key, "Method": label,
                         **m, "TrainSize": len(yr), "TimeSec": dt})
            print(f"  [完成] {name:>14s} × {label:<8s}  "
                  f"F1={m['F1']:.4f}  AUC={m['AUC']:.4f}  ({dt}s, train={len(yr)})")
    return pd.DataFrame(rows)


def experiment_scheme_ab(X_train, y_train, X_test, y_test, classes) -> pd.DataFrame:
    """最佳化參數下，方案 A vs 方案 B 的 F1 比較。"""
    rows = []
    for name in CLASSIFIER_NAMES:
        Xa, ya = scheme_a_undersample(X_train, y_train, ratio=1.0)
        Xb, yb = scheme_b_undersample(X_train, y_train)
        fa = fit_and_f1(name, Xa, ya, X_test, y_test, classes)
        fb = fit_and_f1(name, Xb, yb, X_test, y_test, classes)
        rows.append({"Classifier": name, "SchemeA_F1": fa, "SchemeB_F1": fb})
        print(f"  [完成] {name:>14s}  方案A F1={fa:.4f}  方案B F1={fb:.4f}")
    return pd.DataFrame(rows)


def fit_and_f1(name, Xr, yr, X_test, y_test, classes) -> float:
    from sklearn.metrics import f1_score
    model = build_classifier(name, optimized=True)
    model.fit(Xr, yr)
    y_pred = model.predict(X_test)
    return round(f1_score(y_test, y_pred, average="macro", labels=classes,
                          zero_division=0), 4)


def experiment_ratio_analysis(X_train, y_train, X_test, y_test, classes) -> pd.DataFrame:
    """XGBoost（最佳化）於不同欠採樣比例（1:1 ~ 1:5）的 F1。"""
    from sklearn.metrics import f1_score
    rows = []
    counts = y_train.value_counts()
    n_min = int(counts.min())
    n_max = int(counts.max())
    for ratio in [1, 2, 3, 4, 5]:
        Xr, yr = scheme_a_undersample(X_train, y_train, ratio=ratio)
        model = build_classifier("XGBoost", optimized=True)
        model.fit(Xr, yr)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro", labels=classes,
                      zero_division=0)
        target = min(int(round(n_min * ratio)), n_max)
        rows.append({"Ratio": ratio,
                     "MinorityCount": n_min,
                     "MajorityTarget": target,
                     "F1": round(f1, 4)})
        print(f"  [完成] 比例 1:{ratio}  多數組目標={target:>4d}  F1={f1:.4f}")
    return pd.DataFrame(rows)


# ---------- 6. 主流程 ----------
def main():
    total_t0 = time.time()
    print("=" * 68)
    print("欠採樣於不平衡分類分析 — Dry Bean Dataset")
    print("=" * 68)

    # 1. 資料
    df = ensure_dataset()
    df = df.dropna().reset_index(drop=True)
    classes = list(df["Class"].value_counts().index)
    print(f"[1/7] 資料載入完成：{df.shape[0]} 筆、{df.shape[1] - 1} 個特徵、"
          f"{len(classes)} 個類別")
    for cls in classes:
        print(f"      {cls:<10s} {int((df['Class'] == cls).sum()):>5d} 筆")

    # 2. 分割
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"[2/7] 資料分割：訓練集 {len(X_train)}（80%）/ 測試集 {len(X_test)}（20%）")

    # 類別標籤轉為整數編碼 0 ~ n_classes-1（XGBoost 3.x 不接受字串標籤）
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y)
    y_train = pd.Series(le.transform(y_train), name="Class")
    y_test = pd.Series(le.transform(y_test), name="Class")
    classes_codes = list(range(len(classes)))

    # 3. 標準化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    print("[3/7] 標準化完成（z = (x - μ) / σ，僅以訓練集擬合）")

    # 4. 類別分布圖
    FIG_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    plot_class_distribution(df, classes, FIG_DIR / "01_class_distribution.png")
    df["Class"].value_counts().reindex(classes).to_csv(
        RESULT_DIR / "class_distribution.csv", encoding="utf-8-sig")
    print("[4/7] 類別分布圖 → figures/01_class_distribution.png")

    # 5. 方法比較（四分類器 × 四方法）
    print("[5/7] 方法比較：四種分類器 × 四種方法（SMOTE/下採樣/最優化/下採樣+最優化）...")
    res = experiment_method_comparison(X_train_s, y_train, X_test_s, y_test,
                                       classes_codes)
    res.to_csv(RESULT_DIR / "method_comparison_f1.csv", index=False,
               encoding="utf-8-sig")

    # ROC 曲線比較
    print("      繪製 ROC 曲線比較圖 ...")
    models_opt = {n: build_classifier(n, optimized=True) for n in CLASSIFIER_NAMES}
    for n, m in models_opt.items():
        m.fit(X_train_s, y_train)
    plot_roc_comparison(models_opt, X_test_s, y_test, classes_codes,
                        "ROC Curve Comparison for KNN, Random Forest, SVM, and "
                        "XGBoost (Optimized Parameters)",
                        FIG_DIR / "02_roc_optimized.png")

    models_us_opt = {}
    for n in CLASSIFIER_NAMES:
        Xr, yr = scheme_a_undersample(X_train_s, y_train, ratio=1.0)
        m = build_classifier(n, optimized=True)
        m.fit(Xr, yr)
        models_us_opt[n] = m
    plot_roc_comparison(models_us_opt, X_test_s, y_test, classes_codes,
                        "ROC Curve Comparison with Random Undersampling and "
                        "Optimized for KNN, Random Forest, SVM, and XGBoost",
                        FIG_DIR / "03_roc_undersample_optimized.png")

    plot_f1_method_comparison(res, FIG_DIR / "04_f1_method_comparison.png")
    print("      ROC 圖與 F1 比較圖已輸出至 figures/")

    # 6. 方案 A vs 方案 B
    print("[6/7] 方案比較：僅對最多數組（A）vs 對所有非最少組（B）...")
    df_ab = experiment_scheme_ab(X_train_s, y_train, X_test_s, y_test,
                                 classes_codes)
    df_ab.to_csv(RESULT_DIR / "scheme_ab_comparison.csv", index=False,
                 encoding="utf-8-sig")
    plot_scheme_ab_comparison(df_ab, FIG_DIR / "05_scheme_ab_comparison.png")

    # 7. 比例分析
    print("[7/7] 比例分析：XGBoost 於欠採樣比例 1:1 ~ 1:5 ...")
    df_ratio = experiment_ratio_analysis(X_train_s, y_train, X_test_s, y_test,
                                         classes_codes)
    df_ratio.to_csv(RESULT_DIR / "ratio_analysis.csv", index=False,
                    encoding="utf-8-sig")
    plot_ratio_analysis(df_ratio, FIG_DIR / "06_ratio_analysis.png")

    # 摘要輸出
    print("\n" + "=" * 68)
    print("結果摘要")
    print("=" * 68)
    pivot = res.pivot_table(index="Classifier", columns="Method", values="F1")
    print("\nF1-Score 比較表：")
    print(pivot.to_string())
    print("\n方案 A/B 比較（最佳化參數）：")
    print(df_ab.to_string(index=False))
    best_ratio = df_ratio.loc[df_ratio["F1"].idxmax()]
    print("\n比例分析（XGBoost 最佳化）：")
    print(df_ratio.to_string(index=False))
    print(f"\n最佳比例：1:{best_ratio['Ratio']}（F1 = {best_ratio['F1']}）")
    print(f"\n全部完成，耗時 {time.time() - total_t0:.0f} 秒。")
    print(f"圖表目錄：{FIG_DIR}")
    print(f"結果目錄：{RESULT_DIR}")


if __name__ == "__main__":
    main()
