"""
命令列入口 (command-line interface)
===================================

用法：
    imbalance-lab run                                  # 全部資料集、全部階段
    imbalance-lab run --dataset credit_card            # 只跑信用卡
    imbalance-lab run --stages cost                    # 只重跑成本分析（讀取先前存好的預測）
    imbalance-lab run --config configs/quick.yaml      # 換一份設定（例如快速版）

階段 (stages)：
    tune → cv → stats → test → cost → ir
    （stats 只重新分析已存好的 CV 分數，不重新訓練）
每個階段的輸出都存成檔案，後面的階段直接讀檔，
所以改了成本假設只要重跑 cost，不用重新訓練所有模型。

輸出位置：
    results/<dataset>/   CSV / JSON 結果
    figures/<dataset>/   PNG 圖
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import cost as cost_mod
from . import plots
from .data import load_dataset
from .experiments import evaluate_test, run_cv, run_ir_experiment, split_train_test, tune
from .metrics import PRIMARY_METRIC
from .stats import average_ranks, compare_to_baseline

ROOT = Path(__file__).resolve().parents[2]  # 專案根目錄
ALL_STAGES = ["tune", "cv", "stats", "test", "cost", "ir"]


# ---------------------------------------------------------------------------
# 存取「機率預測」：dict[(模型, 策略)] ⇄ .npz 檔
# ---------------------------------------------------------------------------
def _save_proba(d: dict, path: Path) -> None:
    np.savez_compressed(path, **{f"{m}|{s}": v for (m, s), v in d.items()})


def _load_proba(path: Path) -> dict:
    with np.load(path) as z:
        return {tuple(k.split("|")): z[k] for k in z.files}


def _fmt_csv(df: pd.DataFrame, path: Path) -> None:
    """數字四捨五入到小數點 4 位再存，讓 CSV 在 GitHub 上好讀。"""
    df.round(4).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# 單一資料集的完整流程
# ---------------------------------------------------------------------------
def run_dataset(key: str, cfg: dict, stages: list[str], n_jobs: int) -> None:
    res_dir, fig_dir = ROOT / "results" / key, ROOT / "figures" / key
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(key)
    train, test = split_train_test(ds, cfg["test_size"], cfg["seed"])
    metric = PRIMARY_METRIC[ds.task]
    models, strategies = cfg["models"], cfg["strategies"]
    seed, cv = cfg["seed"], cfg["cv"]
    print(f"\n=== {ds.display}: {len(ds.y)} rows, {ds.X.shape[1]} features, "
          f"{len(ds.class_names)} classes, IR={ds.imbalance_ratio:.2f}, primary metric={metric}")

    plots.class_distribution(ds.y, ds.class_names, ds.display, fig_dir / "01_class_distribution.png")

    # ---- 階段 1：調參 ----
    params_path = res_dir / "tuned_params.json"
    if "tune" in stages:
        tuned = tune(train, models, cfg, n_jobs=n_jobs)
        params_path.write_text(json.dumps(tuned, indent=2))
    tuned = json.loads(params_path.read_text())
    params = {m: v["params"] for m, v in tuned.items()}

    # ---- 階段 2：Repeated CV ----
    if "cv" in stages:
        scores, oof = run_cv(train, models, strategies, params, cv["n_splits"],
                             cv["n_repeats"], seed, n_jobs)
        _fmt_csv(scores, res_dir / "cv_scores.csv")
        _save_proba(oof, res_dir / "oof_proba.npz")

    # ---- 階段 2b：CV 結果的統計分析（只讀 cv_scores.csv，不重新訓練）----
    if "cv" in stages or "stats" in stages:
        analyze_cv(ds.display, ds.task, res_dir, fig_dir, cv["n_splits"])

    # ---- 階段 3：最終 test set ----
    if "test" in stages:
        test_scores, test_proba = evaluate_test(train, test, models, strategies, params, seed)
        _fmt_csv(test_scores, res_dir / "test_scores.csv")
        _save_proba(test_proba, res_dir / "test_proba.npz")
        if ds.task == "binary":
            plots.pr_curves(test.y, test_proba, "xgboost",
                            f"{ds.display}: XGBoost precision-recall on test set",
                            fig_dir / "04_pr_curves_xgboost.png")

    # ---- 階段 4：成本分析（只有二分類有意義）----
    if "cost" in stages and ds.task == "binary":
        oof = _load_proba(res_dir / "oof_proba.npz")
        test_proba = _load_proba(res_dir / "test_proba.npz")
        c = cfg["cost"]
        costs = cost_mod.cost_table(oof, test_proba, train.y, test.y, c["fn"], c["fp"])
        _fmt_csv(costs, res_dir / "cost_policies.csv")
        sens = cost_mod.sensitivity(oof, test_proba, train.y, test.y, c["ratios"], c["fp"])
        _fmt_csv(sens, res_dir / "cost_sensitivity.csv")

        plots.threshold_policies(
            costs, "xgboost",
            f"XGBoost: cost by threshold rule (C_FN={c['fn']:g}, C_FP={c['fp']:g})",
            fig_dir / "05_threshold_policies.png")
        plots.cost_sensitivity(sens, "xgboost",
                               "XGBoost (tuned threshold): savings across cost assumptions",
                               fig_dir / "06_cost_sensitivity.png")

        # 校準圖：欠採樣讓機率偏高，用 Dal Pozzolo 公式修正後應該回到對角線附近
        beta = train.y.sum() / (len(train.y) - train.y.sum())  # 保留的負類比例
        p = {s: test_proba[("xgboost", s)][:, 1] for s in ["none", "random_under", "smote", "class_weight"]}
        corrected = cost_mod.correct_undersampled_proba(p["random_under"], beta)
        curves = {
            "none": (p["none"], "none", "-"),
            "class weight": (p["class_weight"], "class_weight", "-"),
            "random undersampling": (p["random_under"], "random_under", "-"),
            "random undersampling, prior-corrected": (corrected, "random_under", "--"),
            "SMOTENC": (p["smote"], "smote", "-"),
        }
        plots.calibration(test.y, curves, "XGBoost: are the predicted probabilities honest?",
                          fig_dir / "07_calibration.png")
        brier = {k: float(np.mean((v[0] - test.y) ** 2)) for k, v in curves.items()}
        (res_dir / "calibration_brier.json").write_text(json.dumps(brier, indent=2))


def analyze_cv(display: str, task: str, res_dir: Path, fig_dir: Path, n_splits: int) -> None:
    """
    從 cv_scores.csv 產生：平均 ± 標準差、vs. baseline 檢定、平均排名、圖。
    獨立成一個函式，是為了「改了統計方法只要重跑這一步」（--stages stats）。
    """
    scores = pd.read_csv(res_dir / "cv_scores.csv")
    metric = PRIMARY_METRIC[task]

    # 彙總：每個組合的平均 ± 標準差
    summary = (scores.drop(columns=["repeat", "fold"]).groupby(["model", "strategy"])
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    _fmt_csv(summary.reset_index(), res_dir / "cv_summary.csv")

    # 統計檢定：每個策略 vs. none。
    # 二分類另外檢定 f1（門檻 0.5）與 f1_best（不依賴門檻），用來分辨「門檻效應」
    metrics = [metric] + (["f1", "f1_best"] if task == "binary" else ["min_recall"])
    comp = pd.concat([compare_to_baseline(scores, m, n_splits) for m in metrics], ignore_index=True)
    _fmt_csv(comp, res_dir / "cv_vs_baseline.csv")

    ranks = average_ranks(scores, metric)
    _fmt_csv(ranks, res_dir / "cv_avg_ranks.csv")
    (res_dir / "friedman.json").write_text(json.dumps(ranks.attrs.get("friedman", {}), indent=2))

    plots.cv_dotplot(scores, metric, f"{display}: {metric} by model and strategy",
                     fig_dir / "02_cv_dotplot.png")
    plots.diff_forest(comp[comp.metric == metric], f"{display}: does the strategy beat doing nothing?",
                      fig_dir / "03_diff_vs_baseline.png")


def run_ir(cfg: dict, n_jobs: int) -> None:
    """受控不平衡程度實驗（使用信用卡資料調好的參數）。"""
    ir = cfg["ir_experiment"]
    key = ir["dataset"]
    res_dir, fig_dir = ROOT / "results" / "ir_experiment", ROOT / "figures" / "ir_experiment"
    res_dir.mkdir(parents=True, exist_ok=True)
    tuned = json.loads((ROOT / "results" / key / "tuned_params.json").read_text())
    params = {m: v["params"] for m, v in tuned.items()}

    ds = load_dataset(key)
    scores = run_ir_experiment(ds, ir["ratios"], ir["models"], ir["strategies"], params,
                               cfg["cv"]["n_splits"], ir["n_repeats"], cfg["seed"], n_jobs)
    _fmt_csv(scores, res_dir / "ir_scores.csv")
    summary = (scores.groupby(["ir", "model", "strategy"])
               [["pr_auc", "pr_auc_lift", "roc_auc", "recall", "precision", "f1", "f1_best", "balanced_acc"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    _fmt_csv(summary.reset_index(), res_dir / "ir_summary.csv")
    plots.ir_small_multiples(scores, ["pr_auc_lift", "f1", "f1_best"],
                             "Same data, increasing imbalance: how each strategy holds up",
                             fig_dir / "ir_small_multiples.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="imbalance-lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="執行實驗")
    run.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    run.add_argument("--dataset", nargs="+", help="只跑指定資料集（預設：設定檔裡全部）")
    run.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    run.add_argument("--n-jobs", type=int, default=2, help="平行執行的 CPU 數")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    t0 = time.time()
    per_dataset = [s for s in args.stages if s != "ir"]
    if per_dataset:
        for key in args.dataset or cfg["datasets"]:
            run_dataset(key, cfg, per_dataset, args.n_jobs)
    if "ir" in args.stages:
        run_ir(cfg, args.n_jobs)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
