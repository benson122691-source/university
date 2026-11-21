"""
實驗流程 (experiment stages)
============================

整體設計（每一步只用該用的資料，test set 從頭到尾只在最後碰一次）：

    全部資料
      ├── 80% 訓練集 (train) ───────────────────────────────┐
      │     ├─ 階段 1  調參：在 train 內部做 3 折 CV 選參數  │
      │     ├─ 階段 2  基準：在 train 上做 5×3 Repeated CV   │  ← 用來「比較策略」
      │     │          （同時保存 out-of-fold 預測，給成本分析選門檻）
      │     └─ 階段 3  用全部 train 訓練最終模型 ───────────┘
      └── 20% 測試集 (test)  → 階段 3 最終評估、成本分析的「期末考」

舊版的問題對照：
- 舊版只切一次 train/test，看不出誤差 → 現在每組有 15 個分數
- 舊版的「最佳參數」寫死、找不到調參過程 → 現在調參有程式、只用 train
- 舊版兩份資料共用同一組參數 → 現在每份資料、每個模型各自調參
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from .data import Dataset, make_imbalanced
from .metrics import PRIMARY_METRIC, compute_metrics
from .models import SEARCH_SPACES, is_compatible
from .pipeline import build_pipeline

# sklearn 的 scoring 名稱：調參時要最佳化的目標
_SKLEARN_SCORING = {"binary": "average_precision", "multiclass": "f1_macro"}


# ---------------------------------------------------------------------------
# 工具：取子集合
# ---------------------------------------------------------------------------
def subset(ds: Dataset, idx: np.ndarray) -> Dataset:
    """取出部分列，回傳新的 Dataset（欄位資訊不變）。"""
    return Dataset(key=ds.key, display=ds.display,
                   X=ds.X.iloc[idx].reset_index(drop=True), y=ds.y[idx],
                   class_names=ds.class_names, categorical=ds.categorical)


def split_train_test(ds: Dataset, test_size: float, seed: int) -> tuple[Dataset, Dataset]:
    """分層切分 (stratified)：train 和 test 的類別比例與原始資料一致。"""
    idx_train, idx_test = train_test_split(
        np.arange(len(ds.y)), test_size=test_size, stratify=ds.y, random_state=seed)
    return subset(ds, idx_train), subset(ds, idx_test)


def combos(models: list[str], strategies: list[str]) -> list[tuple[str, str]]:
    """列出所有「合理的」模型 × 策略組合（不合理的見 models.is_compatible）。"""
    return [(m, s) for m in models for s in strategies if is_compatible(m, s)]


# ---------------------------------------------------------------------------
# 階段 1：調參
# ---------------------------------------------------------------------------
def tune(train: Dataset, models: list[str], cfg: dict, n_jobs: int = 2) -> dict:
    """
    每個模型用 RandomizedSearchCV 在「訓練集」上調參。

    取捨說明：參數是在 strategy="none" 下調的，再套用到所有策略。
    理想做法是每個策略各自調參，但計算量會變成 5 倍；
    這個簡化對所有策略一視同仁（大家都用同一組參數），比較仍然公平，README 有註明。
    balanced_rf 只搭配 none，本來就是在自己的設定下調參。
    """
    seed = cfg["seed"]
    inner_cv = StratifiedKFold(cfg["tuning"]["inner_splits"], shuffle=True, random_state=seed)
    out = {}
    for model in models:
        t0 = time.time()
        search = RandomizedSearchCV(
            build_pipeline(train, model, "none", seed),
            SEARCH_SPACES[model],
            n_iter=cfg["tuning"]["n_iter"],
            scoring=_SKLEARN_SCORING[train.task],
            cv=inner_cv, random_state=seed, n_jobs=n_jobs,
        )
        search.fit(train.X, train.y)
        # numpy 型別轉成 Python 原生型別，才能存成 JSON
        best = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        out[model] = {"params": best, "inner_cv_score": float(search.best_score_)}
        print(f"  [tune] {model:<14s} {_SKLEARN_SCORING[train.task]}={search.best_score_:.4f} "
              f"({time.time() - t0:.0f}s)  {best}")
    return out


# ---------------------------------------------------------------------------
# 階段 2：Repeated CV 基準
# ---------------------------------------------------------------------------
@dataclass
class _Job:
    model: str
    strategy: str
    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def _run_one(ds: Dataset, job: _Job, params: dict, seed: int) -> dict:
    """跑一個 (模型, 策略, 折)：在訓練折上 fit，在驗證折上評估。"""
    tr, va = subset(ds, job.train_idx), subset(ds, job.val_idx)
    pipe = build_pipeline(ds, job.model, job.strategy, seed, params.get(job.model))
    t0 = time.time()
    pipe.fit(tr.X, tr.y)          # ← 重採樣只發生在這裡，而且只看得到訓練折
    fit_sec = time.time() - t0
    proba = pipe.predict_proba(va.X)  # ← 驗證折維持原始分布，不會被重採樣
    return {
        "model": job.model, "strategy": job.strategy, "repeat": job.repeat, "fold": job.fold,
        **compute_metrics(va.y, proba, ds.task),
        "fit_sec": fit_sec,
        "_val_idx": job.val_idx, "_proba": proba,  # 暫存，給 out-of-fold 預測用
    }


def run_cv(train: Dataset, models: list[str], strategies: list[str], params: dict,
           n_splits: int, n_repeats: int, seed: int, n_jobs: int = 2,
           ) -> tuple[pd.DataFrame, dict]:
    """
    在訓練集上做 Repeated Stratified K-Fold。

    回傳
    ----
    scores : 每一列 = 一個 (模型, 策略, repeat, fold) 的各項指標
    oof    : {(模型, 策略): 機率矩陣}，取第 0 次重複的 out-of-fold 預測。
             out-of-fold = 每一筆資料的預測，都來自「沒看過它」的模型，
             所以可以拿來選門檻而不會作弊（不用碰到 test set）。

    注意：所有組合使用「同一組」切分（同一個 random_state），
    這樣不同策略在同一折上的分數可以「配對」比較，統計檢定才成立。
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(rskf.split(train.X, train.y))
    jobs = [
        _Job(m, s, i // n_splits, i % n_splits, tr_idx, va_idx)
        for (m, s) in combos(models, strategies)
        for i, (tr_idx, va_idx) in enumerate(splits)
    ]
    print(f"  [cv] {len(jobs)} fits ({len(combos(models, strategies))} combos × "
          f"{n_splits}×{n_repeats} folds)")

    # 平行執行：每個 worker 跑一個 fold（verbose 印進度）
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one)(train, job, params, seed) for job in jobs)

    # 組出 out-of-fold 機率（只取 repeat 0：每筆資料剛好被預測一次）
    n_classes = len(train.class_names)
    oof: dict[tuple[str, str], np.ndarray] = {}
    for r in rows:
        if r["repeat"] != 0:
            continue
        key = (r["model"], r["strategy"])
        oof.setdefault(key, np.full((len(train.y), n_classes), np.nan))
        oof[key][r["_val_idx"]] = r["_proba"]

    scores = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    return scores, oof


# ---------------------------------------------------------------------------
# 階段 3：最終測試
# ---------------------------------------------------------------------------
def evaluate_test(train: Dataset, test: Dataset, models: list[str], strategies: list[str],
                  params: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """
    用「全部訓練集」重新訓練每個組合，在 test set 上評估一次。
    同時記錄重採樣後的訓練筆數，讓讀者看到每種策略丟掉/合成了多少資料。
    """
    rows, test_proba = [], {}
    for model, strategy in combos(models, strategies):
        pipe = build_pipeline(train, model, strategy, seed, params.get(model), n_jobs=2)
        t0 = time.time()
        pipe.fit(train.X, train.y)
        fit_sec = time.time() - t0
        proba = pipe.predict_proba(test.X)
        test_proba[(model, strategy)] = proba

        # 重採樣後的訓練筆數：把前兩步（scale → sampler）單獨跑一次來計算
        n_after = len(train.y)
        if "sampler" in pipe.named_steps:
            Xs = pipe.named_steps["scale"].transform(train.X)
            _, y_res = pipe.named_steps["sampler"].fit_resample(Xs, train.y)
            n_after = len(y_res)

        rows.append({"model": model, "strategy": strategy,
                     **compute_metrics(test.y, proba, test.task),
                     "train_size": n_after, "fit_sec": fit_sec})
        m = PRIMARY_METRIC[test.task]
        print(f"  [test] {model:<14s} × {strategy:<13s} {m}={rows[-1][m]:.4f}  "
              f"train_size={n_after}")
    return pd.DataFrame(rows), test_proba


# ---------------------------------------------------------------------------
# 受控不平衡程度實驗
# ---------------------------------------------------------------------------
def run_ir_experiment(ds: Dataset, ratios: list[float], models: list[str],
                      strategies: list[str], params: dict, n_splits: int, n_repeats: int,
                      seed: int, n_jobs: int = 2) -> pd.DataFrame:
    """
    對每一個 IR：先用 make_imbalanced 製造資料，再跑 Repeated CV。

    額外計算 pr_auc_lift = PR-AUC / 正類比例：
    隨機猜測的 PR-AUC 剛好等於正類比例，IR 越大這個基準越低，
    直接比 PR-AUC 會誤以為「模型變爛」；除以基準後才能跨 IR 比較。
    """
    frames = []
    for ratio in ratios:
        sub = make_imbalanced(ds, ratio, seed=seed)
        prevalence = sub.y.mean()
        print(f"  [ir] IR={ratio:g}: n={len(sub.y)}, positives={int(sub.y.sum())}")
        scores, _ = run_cv(sub, models, strategies, params, n_splits, n_repeats, seed, n_jobs)
        scores["ir"] = ratio
        scores["prevalence"] = prevalence
        scores["pr_auc_lift"] = scores["pr_auc"] / prevalence
        frames.append(scores)
    return pd.concat(frames, ignore_index=True)
