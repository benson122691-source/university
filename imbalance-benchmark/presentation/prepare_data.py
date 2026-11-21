"""
從 results/ 讀出簡報需要的數字，存成 presentation/deck_data.json。
簡報上的每個數字都從這裡來，不手打 —— 結果更新後重跑一次就同步。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
out = {}

# 1) 信用卡 XGBoost：門檻效應（CV 平均）
s = pd.read_csv(R / "credit_card/cv_summary.csv").query("model == 'xgboost'").set_index("strategy")
out["threshold_effect"] = {k: {"f1": round(s.loc[k, "f1_mean"], 3), "f1_best": round(s.loc[k, "f1_best_mean"], 3),
                               "recall": round(s.loc[k, "recall_mean"], 3)}
                           for k in ["none", "class_weight", "random_under", "smote"]}

# 2) 信用卡 XGBoost：三種門檻規則的 test 成本
c = pd.read_csv(R / "credit_card/cost_policies.csv").query("model == 'xgboost'")
out["cost"] = {st: {p: round(c[(c.strategy == st) & (c.policy == p)].cost_per_applicant.iloc[0], 3)
                    for p in ["default", "bayes", "tuned"]}
               for st in ["none", "class_weight", "random_under", "tomek", "smote"]}
none_t = c[(c.strategy == "none") & (c.policy == "tuned")].iloc[0]
out["cost_none_tuned_savings"] = round(100 * none_t.savings_vs_approve_all, 1)
out["bayes_reject"] = {st: round(100 * c[(c.strategy == st) & (c.policy == "bayes")].reject_rate.iloc[0], 1)
                       for st in ["none", "class_weight", "random_under"]}
out["brier"] = {k: round(v, 3) for k, v in json.loads((R / "credit_card/calibration_brier.json").read_text()).items()}

# 3) 跨資料集勝 / 平 / 負
w = pd.read_csv(R / "summary/win_tie_loss.csv").groupby("kind")[["better", "no sig. diff", "worse"]].sum()
out["wtl"] = {k: [int(x) for x in w.loc[k]] for k in w.index}
cw = pd.read_csv(R / "summary/cost_win_tie_loss.csv").groupby("policy")[["cheaper", "no clear diff", "costlier"]].sum()
out["cost_wtl"] = {k: [int(x) for x in cw.loc[k]] for k in cw.index}

# 4) 熱圖：每格的數值與顏色（跟 figures/summary 的配色完全相同：紅 = 較差、藍 = 較好）
cells = pd.read_csv(R / "summary/heatmap_cells.csv")
lim = cells.mean_diff.abs().max()
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
norm = TwoSlopeNorm(0, -lim, lim)
rows = cells.drop_duplicates("dataset").sort_values("ir")
heat = []
for _, r in rows.iterrows():
    row = {"display": r.display, "metric": r.metric, "ir": round(r.ir), "cells": []}
    for st in ["class_weight", "random_under", "tomek", "smote"]:
        x = cells[(cells.dataset == r.dataset) & (cells.strategy == st)].iloc[0]
        rgb = cmap(norm(x.mean_diff))[:3]
        lum = np.dot(rgb, [0.2126, 0.7152, 0.0722])
        row["cells"].append({"v": f"{x.mean_diff:+.3f}", "wtl": f"{x.better:.0f} / {x.tie:.0f} / {x.worse:.0f}",
                             "fill": to_hex(rgb)[1:].upper(), "text": "FFFFFF" if lum < 0.45 else "0B0B0B"})
    heat.append(row)
out["heatmap"] = heat

# 5) Yeast 先驗修正
d = pd.read_csv(R / "yeast/decision_rule.csv")
out["yeast"] = {f"{r.model}|{r.variant}": [round(r.f1_macro, 3), round(r.min_recall, 3)] for r in d.itertuples()}

# 6) v1 舊結果：直接從 git 歷史讀 v1 的 CSV（Dry Bean，「不處理」vs「欠採樣」，兩者都用 v1 的參數）
import io
import subprocess

raw = subprocess.run(["git", "show", "42c59a8:dry-bean-undersampling/results/dry_bean/method_comparison_f1.csv"],
                     cwd=ROOT, capture_output=True, text=True, check=True).stdout.lstrip("\ufeff")
v1 = pd.read_csv(io.StringIO(raw)).pivot(index="Classifier", columns="Method_Key", values="F1")
out["v1"] = {m: [round(float(v1.loc[m, "optimized"]), 3), round(float(v1.loc[m, "undersample+optimized"]), 3)]
             for m in v1.index}

# 7) 簡報專用的大字版圖片：同一套畫圖程式，只把基準字級從 9.5 調到 15，
#    放到投影片上（約 7 吋寬）時字才不會太小。輸出到 presentation/img/
from imbalance_lab import plots  # noqa: E402
from imbalance_lab.cli import _calibration_curves, _load_proba  # noqa: E402
from imbalance_lab.data import load_dataset  # noqa: E402
from imbalance_lab.experiments import split_train_test  # noqa: E402

IMG = Path(__file__).parent / "img"
plots.BASE_FONT_SIZE = 15
comp = pd.read_csv(R / "credit_card/cv_vs_baseline.csv")
plots.diff_forest(comp[comp.metric == "pr_auc"], "Credit card · Δ PR-AUC vs. no handling", IMG / "forest")
ds = load_dataset("credit_card")
train, test = split_train_test(ds, 0.2, 42)
plots.calibration(test.y, _calibration_curves(_load_proba(R / "credit_card/test_proba.npz"), train.y),
                  "Credit card · XGBoost calibration", IMG / "calibration")
# IR 圖在簡報只放兩列（排序能力、門檻 0.5 的 F1），完整三列版本在 README
plots.ir_small_multiples(pd.read_csv(R / "ir_experiment/ir_scores.csv"), ["pr_auc_lift", "f1"],
                         "Same data, increasing imbalance", IMG / "ir", panel_size=(4.6, 3.6))
for f in IMG.glob("*.svg"):  # 簡報只用 PNG
    f.unlink()
out["images"] = {}
for k in ["forest", "calibration", "ir"]:
    w_, h_ = Image.open(IMG / f"{k}.png").size
    out["images"][k] = {"path": f"img/{k}.png", "ratio": w_ / h_}  # 相對於 presentation/

(Path(__file__).parent / "deck_data.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k not in ("heatmap", "images", "yeast")}, ensure_ascii=False, indent=1))
print(out["yeast"])
