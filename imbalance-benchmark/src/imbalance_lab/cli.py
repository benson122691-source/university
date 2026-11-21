"""
命令列入口 (command-line interface)
===================================

用法：
    imbalance-lab run                                  # 全部資料集、全部階段
    imbalance-lab run --dataset credit_card            # 只跑信用卡
    imbalance-lab run --stages cost                    # 只重跑成本分析（讀取先前存好的預測）
    imbalance-lab run --config configs/quick.yaml      # 換一份設定（例如快速版）

階段 (stages)：
    tune → cv → stats → test → cost → ir → summary → figures
    （stats、summary、figures 只讀已存好的結果，不重新訓練；
      例如只想重畫所有圖：imbalance-lab run --stages figures summary）
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
from . import decision, plots
from .data import load_dataset
from .experiments import evaluate_test, run_cv, run_ir_experiment, split_train_test, tune
from .metrics import PRIMARY_METRIC
from .stats import average_ranks, compare_to_baseline

ROOT = Path(__file__).resolve().parents[2]  # 專案根目錄
ALL_STAGES = ["tune", "cv", "stats", "test", "cost", "ir", "summary", "figures"]


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

    # ---- 階段 4（多分類版）：決策規則分析 —— 先驗修正 vs. 重採樣 ----
    if "cost" in stages and ds.task == "multiclass":
        oof = _load_proba(res_dir / "oof_proba.npz")
        table = decision.decision_rule_table(oof, train.y, seed=seed)
        _fmt_csv(table, res_dir / "decision_rule.csv")
        print(table.round(3).to_string(index=False))

    # ---- 階段 4：成本分析（只有二分類有意義）----
    if "cost" in stages and ds.task == "binary":
        oof = _load_proba(res_dir / "oof_proba.npz")
        test_proba = _load_proba(res_dir / "test_proba.npz")
        c = cfg["cost"]
        costs = cost_mod.cost_table(oof, test_proba, train.y, test.y, c["fn"], c["fp"])
        _fmt_csv(costs, res_dir / "cost_policies.csv")
        # 跟「none + tuned 門檻」比較，附 bootstrap 95% 信賴區間
        cvb = cost_mod.cost_vs_baseline(oof, test_proba, train.y, test.y, c["fn"], c["fp"], seed=seed)
        _fmt_csv(cvb, res_dir / "cost_vs_baseline.csv")
        sens = cost_mod.sensitivity(oof, test_proba, train.y, test.y, c["ratios"], c["fp"])
        _fmt_csv(sens, res_dir / "cost_sensitivity.csv")

        # 校準：欠採樣讓機率偏高，用 Dal Pozzolo 公式修正後應該回到真實比例附近
        curves = _calibration_curves(test_proba, train.y)
        brier = {k: float(np.mean((v[0] - test.y) ** 2)) for k, v in curves.items()}
        (res_dir / "calibration_brier.json").write_text(json.dumps(brier, indent=2))

    # ---- 圖表：只讀已存好的結果來畫，任何階段跑完都會更新；也可單獨執行 --stages figures ----
    if set(stages) & {"cv", "stats", "test", "cost", "figures"}:
        make_figures(ds, train, test, res_dir, fig_dir, cfg)


def _calibration_curves(test_proba: dict, y_train: np.ndarray) -> dict:
    """校準圖要畫的幾條線：{圖例: (正類機率, 策略, 線型)}。"""
    beta = y_train.sum() / (len(y_train) - y_train.sum())  # 欠採樣時「保留的負類比例」
    p = {s: test_proba[("xgboost", s)][:, 1] for s in ["none", "random_under", "smote", "class_weight"]}
    return {
        "none": (p["none"], "none", "-"),
        "class weight": (p["class_weight"], "class_weight", "-"),
        "random undersampling": (p["random_under"], "random_under", "-"),
        "random undersampling, prior-corrected": (
            cost_mod.correct_undersampled_proba(p["random_under"], beta), "random_under", "--"),
        "SMOTE / SMOTENC": (p["smote"], "smote", "-"),
    }


def make_figures(ds, train, test, res_dir: Path, fig_dir: Path, cfg: dict) -> None:
    """
    依「已經存在的結果檔」重畫這份資料集的所有圖。
    缺哪個結果檔就跳過那張圖，所以任何時候都可以安全地呼叫。
    """
    metric = PRIMARY_METRIC[ds.task]
    plots.class_distribution(ds.y, ds.class_names, ds.display, fig_dir / "01_class_distribution")

    if (res_dir / "cv_scores.csv").exists() and (res_dir / "cv_vs_baseline.csv").exists():
        scores = pd.read_csv(res_dir / "cv_scores.csv")
        comp = pd.read_csv(res_dir / "cv_vs_baseline.csv")
        plots.cv_dotplot(scores, metric, f"{ds.display}: {metric} by model and strategy",
                         fig_dir / "02_cv_dotplot")
        plots.diff_forest(comp[comp.metric == metric], f"{ds.display}: does the strategy beat doing nothing?",
                          fig_dir / "03_diff_vs_baseline")

    if ds.task != "binary" or not (res_dir / "test_proba.npz").exists():
        return
    test_proba = _load_proba(res_dir / "test_proba.npz")
    plots.pr_curves(test.y, test_proba, "xgboost", f"{ds.display}: XGBoost precision–recall on the test set",
                    fig_dir / "04_pr_curves_xgboost")
    if (res_dir / "cost_policies.csv").exists():
        c = cfg["cost"]
        plots.threshold_policies(pd.read_csv(res_dir / "cost_policies.csv"), "xgboost",
                                 f"{ds.display} · XGBoost: test cost by threshold rule "
                                 f"(C_FN = {c['fn']:g}, C_FP = {c['fp']:g})",
                                 fig_dir / "05_threshold_policies")
        plots.cost_sensitivity(pd.read_csv(res_dir / "cost_sensitivity.csv"), "xgboost",
                               f"{ds.display} · XGBoost, tuned thresholds: is any strategy cheaper than none?",
                               fig_dir / "06_cost_sensitivity")
    plots.calibration(test.y, _calibration_curves(test_proba, train.y),
                      f"{ds.display} · XGBoost: are the predicted probabilities honest?",
                      fig_dir / "07_calibration")


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


def run_ir(cfg: dict, n_jobs: int) -> None:
    """受控不平衡程度實驗（使用信用卡資料調好的參數）。"""
    ir = cfg["ir_experiment"]
    key = ir["dataset"]
    res_dir = ROOT / "results" / "ir_experiment"
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
    plot_ir()


def plot_ir() -> None:
    """IR 實驗的圖（只讀 ir_scores.csv）。"""
    f = ROOT / "results" / "ir_experiment" / "ir_scores.csv"
    if f.exists():
        plots.ir_small_multiples(pd.read_csv(f), ["pr_auc_lift", "f1", "f1_best"],
                                 "Same data, increasing imbalance: how each strategy holds up",
                                 ROOT / "figures" / "ir_experiment" / "ir_small_multiples")


def summarize() -> None:
    """
    跨資料集總結：讀取 results/ 底下「每一份」資料集的結果，回答：
    這些策略相對於「什麼都不做」，在多少個 (資料集 × 模型) 上顯著較好 / 沒差 / 顯著較差？

    只讀檔、不訓練，所以任何時候都可以重跑：imbalance-lab run --stages summary
    """
    from .data import SPECS

    out_res, out_fig = ROOT / "results" / "summary", ROOT / "figures" / "summary"
    out_res.mkdir(parents=True, exist_ok=True)
    keys = [k for k in SPECS if (ROOT / "results" / k / "cv_vs_baseline.csv").exists()]

    rows, info = [], []
    for k in keys:
        ds = load_dataset(k)
        info.append({"dataset": k, "display": ds.display, "task": ds.task, "n": len(ds.y),
                     "features": ds.X.shape[1], "classes": len(ds.class_names),
                     "ir": ds.imbalance_ratio, "minority_n": int(np.bincount(ds.y).min())})
        # cv_vs_baseline.csv 裡有：主要指標、(二分類) f1 與 f1_best、(多分類) min_recall
        rows.append(pd.read_csv(ROOT / "results" / k / "cv_vs_baseline.csv").assign(dataset=k))
    info = pd.DataFrame(info)
    _fmt_csv(info, out_res / "datasets.csv")
    comp = pd.concat(rows, ignore_index=True).merge(info[["dataset", "display", "ir", "task"]], on="dataset")
    _fmt_csv(comp, out_res / "all_vs_baseline.csv")

    # 1) 勝 / 平 / 負 統計：依「指標種類」分開
    def kind(r):
        if r.metric in ("pr_auc", "f1_macro"):
            return "primary (threshold-free)"
        if r.metric in ("f1_best", "min_recall"):
            return "secondary (threshold-free)" if r.metric == "f1_best" else "worst-class recall"
        return "F1 at threshold 0.5"

    comp["kind"] = comp.apply(kind, axis=1)
    wtl = (comp.groupby(["kind", "strategy"]).verdict.value_counts().unstack(fill_value=0)
           .reindex(columns=["better", "no sig. diff", "worse"], fill_value=0).reset_index())
    _fmt_csv(wtl, out_res / "win_tie_loss.csv")
    print("\n[summary] wins / ties / losses vs. no handling:")
    print(wtl.to_string(index=False))

    # 2) 熱圖：主要指標，每格 = 各模型平均差異 + 勝平負數
    prim = comp[comp.metric.isin(["pr_auc", "f1_macro"])]
    cells = (prim.groupby(["dataset", "display", "ir", "metric", "strategy"])
             .agg(mean_diff=("diff", "mean"),
                  better=("verdict", lambda v: (v == "better").sum()),
                  tie=("verdict", lambda v: (v == "no sig. diff").sum()),
                  worse=("verdict", lambda v: (v == "worse").sum()))
             .reset_index())
    _fmt_csv(cells, out_res / "heatmap_cells.csv")
    plots.summary_heatmap(cells, "Across 7 datasets: does any strategy beat doing nothing?",
                          out_fig / "summary_heatmap")

    # 3) 成本 & 校準（只有二分類資料有）：XGBoost，test set
    cost_rows, brier_rows = [], []
    for k in keys:
        f = ROOT / "results" / k / "cost_policies.csv"
        if f.exists():
            c = pd.read_csv(f)
            c = c[c.model == "xgboost"].assign(dataset=k)
            cost_rows.append(c)
        f = ROOT / "results" / k / "calibration_brier.json"
        if f.exists():
            brier_rows.append({"dataset": k, **json.loads(f.read_text())})
    if cost_rows:
        cost = pd.concat(cost_rows)
        pivot = cost.pivot_table(index=["dataset", "strategy"], columns="policy",
                                 values="savings_vs_approve_all").reset_index()
        _fmt_csv(pivot, out_res / "cost_savings_xgboost.csv")
        rej = cost[cost.policy == "bayes"].pivot_table(index="dataset", columns="strategy",
                                                       values="reject_rate").reset_index()
        _fmt_csv(rej, out_res / "bayes_threshold_reject_rate_xgboost.csv")
    if brier_rows:
        _fmt_csv(pd.DataFrame(brier_rows), out_res / "calibration_brier_xgboost.csv")

    # 4) 成本差異的 bootstrap 結論：所有模型 × 策略，跟 none + tuned 門檻比
    cvb = [pd.read_csv(ROOT / "results" / k / "cost_vs_baseline.csv").assign(dataset=k)
           for k in keys if (ROOT / "results" / k / "cost_vs_baseline.csv").exists()]
    if cvb:
        cvb = pd.concat(cvb, ignore_index=True)
        _fmt_csv(cvb, out_res / "cost_vs_baseline_all.csv")
        table = (cvb.groupby(["policy", "strategy"]).verdict.value_counts().unstack(fill_value=0)
                 .reindex(columns=["cheaper", "no clear diff", "costlier"], fill_value=0).reset_index())
        _fmt_csv(table, out_res / "cost_win_tie_loss.csv")
        print("\n[summary] test-set cost vs. none + tuned threshold (bootstrap 95% CI):")
        print(table.to_string(index=False))


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
    per_dataset = [s for s in args.stages if s not in ("ir", "summary")]
    if "figures" in args.stages:
        plot_ir()
    if per_dataset:
        for key in args.dataset or cfg["datasets"]:
            run_dataset(key, cfg, per_dataset, args.n_jobs)
    if "ir" in args.stages and "ir_experiment" in cfg:  # 驗證用設定檔沒有 IR 實驗
        run_ir(cfg, args.n_jobs)
    if "summary" in args.stages:
        summarize()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
