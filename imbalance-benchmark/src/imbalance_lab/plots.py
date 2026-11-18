"""
圖表 (figures)
==============

設計原則：
- 每個策略固定一個顏色（不會因圖表不同而換色），基準線 none 一律用灰色
- 一律畫「平均 ± 誤差」，不再只畫單一分數
- 只用一個 y 軸；格線淡化，讓資料是主角
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不開視窗，直接存檔（伺服器 / CI 上也能跑）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .models import MODEL_DISPLAY  # noqa: E402

# 固定的策略配色：顏色跟著「策略」走，所有圖都一樣
STRATEGY_COLORS = {
    "none": "#8a8984",          # 灰色 = 基準線
    "class_weight": "#2a78d6",  # 藍
    "random_under": "#eb6834",  # 橘
    "tomek": "#1baf7a",         # 青綠
    "smote": "#6250d6",         # 紫
}
STRATEGY_LABEL = {
    "none": "none (baseline)", "class_weight": "class weight",
    "random_under": "random undersampling", "tomek": "Tomek links", "smote": "SMOTE / SMOTENC",
}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def class_distribution(y: np.ndarray, class_names: list[str], title: str, path: Path) -> None:
    """類別分布長條圖，標上百分比；由多到少排序。"""
    counts = pd.Series(np.bincount(y), index=class_names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(counts.index, counts.values, color="#2a78d6", width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v / counts.sum():.1%}", ha="center", va="bottom", color=TEXT_2, fontsize=9)
    ax.set_ylabel("samples")
    ax.set_title(f"{title} — imbalance ratio {counts.max() / counts.min():.1f} : 1")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, path)


def cv_dotplot(scores: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """
    每個模型一列，每個策略一個點：點 = 15 折平均，橫線 = ± 1 標準差。
    一眼看出「點之間的差距」是否大於「誤差線長度」。
    """
    agg = scores.groupby(["model", "strategy"])[metric].agg(["mean", "std"]).reset_index()
    models = [m for m in MODEL_DISPLAY if m in agg.model.unique()]
    strategies = [s for s in STRATEGY_COLORS if s in agg.strategy.unique()]
    offsets = np.linspace(-0.3, 0.3, len(strategies))

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(models) + 1.4))
    for off, s in zip(offsets, strategies, strict=True):
        sub = agg[agg.strategy == s].set_index("model")
        ys = [i + off for i, m in enumerate(models) if m in sub.index]
        ms = [m for m in models if m in sub.index]
        ax.errorbar(sub.loc[ms, "mean"], ys, xerr=sub.loc[ms, "std"], fmt="o", ms=6,
                    color=STRATEGY_COLORS[s], ecolor=STRATEGY_COLORS[s], elinewidth=1.5,
                    capsize=0, label=STRATEGY_LABEL[s])
    ax.set_yticks(range(len(models)), [MODEL_DISPLAY[m] for m in models])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"{metric}  (mean ± 1 sd over repeated CV folds)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)
    _save(fig, path)


def diff_forest(comp: pd.DataFrame, title: str, path: Path) -> None:
    """
    森林圖：每個「策略 − baseline」的平均差異與 95% 信賴區間。
    區間跨過 0 = 跟什麼都不做沒有顯著差別。空心點 = Holm 校正後不顯著。
    """
    comp = comp.copy()
    order = [m for m in MODEL_DISPLAY if m in comp.model.unique()]
    comp["m_rank"] = comp.model.map({m: i for i, m in enumerate(order)})
    comp["s_rank"] = comp.strategy.map({s: i for i, s in enumerate(STRATEGY_COLORS)})
    comp = comp.sort_values(["m_rank", "s_rank"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(comp) + 1.4))
    for i, r in comp.iterrows():
        c = STRATEGY_COLORS[r.strategy]
        sig = r.verdict != "no sig. diff"
        ax.plot([r.ci_low, r.ci_high], [i, i], color=c, lw=2)
        ax.plot(r["diff"], i, "o", ms=7, color=c, mfc=c if sig else "white", mew=2)
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(range(len(comp)),
                  [f"{MODEL_DISPLAY[m]} · {STRATEGY_LABEL[s]}"
                   for m, s in zip(comp.model, comp.strategy, strict=True)], fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Δ {comp.metric.iloc[0]} vs. no handling  (95% CI, corrected t-test)")
    ax.set_title(title)
    note = "bars: unadjusted 95% CI · hollow marker: not significant after Holm correction"
    ax.text(1.0, -0.6 / (0.32 * len(comp) + 1.4), note, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=TEXT_2)
    _save(fig, path)


def pr_curves(y_test: np.ndarray, test_proba: dict, model: str, title: str, path: Path) -> None:
    """同一個模型、不同策略在 test set 上的 Precision-Recall 曲線。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for s in STRATEGY_COLORS:
        if (model, s) not in test_proba:
            continue
        p = test_proba[(model, s)][:, 1]
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, lw=2, color=STRATEGY_COLORS[s], label=STRATEGY_LABEL[s])
    ax.axhline(y_test.mean(), color=TEXT_2, ls="--", lw=1)
    ax.text(0.99, y_test.mean() + 0.01, "random guess", ha="right", va="bottom",
            fontsize=8, color=TEXT_2)
    ax.set_xlabel("recall (share of defaulters caught)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, path)


def calibration(y_test: np.ndarray, curves: dict[str, tuple[np.ndarray, str, str]], title: str,
                path: Path) -> None:
    """
    校準圖 (reliability diagram)：把預測機率分成 10 組，
    x = 該組的平均預測機率，y = 該組實際違約比例。完美校準 = 對角線。
    curves：{圖例文字: (正類機率, 策略名稱, 線型)}；顏色跟著策略走，與其他圖一致。
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5)
    for label, (p, strategy, ls) in curves.items():
        frac, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=4, lw=2, color=STRATEGY_COLORS[strategy], ls=ls, label=label)
    ax.set_xlabel("mean predicted default probability")
    ax.set_ylabel("observed default rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, path)


def cost_sensitivity(sens: pd.DataFrame, model: str, title: str, path: Path) -> None:
    """成本比例 (C_FN / C_FP) vs. 相對「全部核准」省下的成本。"""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    sub = sens[sens.model == model]
    for s in STRATEGY_COLORS:
        g = sub[sub.strategy == s].sort_values("cost_ratio")
        if g.empty:
            continue
        ax.plot(g.cost_ratio, 100 * g.savings_vs_approve_all, marker="o", ms=5, lw=2,
                color=STRATEGY_COLORS[s], label=STRATEGY_LABEL[s])
    ax.set_xlabel("cost ratio  C_FN / C_FP  (cost of approving a defaulter vs. rejecting a payer)")
    ax.set_ylabel("cost saved vs. approve-all (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    _save(fig, path)


def threshold_policies(costs: pd.DataFrame, model: str, title: str, path: Path) -> None:
    """同一模型：三種門檻規則 × 各策略的每人平均成本（越低越好）。"""
    sub = costs[costs.model == model]
    policies = ["default", "bayes", "tuned"]
    strategies = [s for s in STRATEGY_COLORS if s in sub.strategy.unique()]
    width = 0.8 / len(strategies)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for j, s in enumerate(strategies):
        vals = [sub[(sub.strategy == s) & (sub.policy == p)].cost_per_applicant.iloc[0]
                for p in policies]
        xs = np.arange(len(policies)) - 0.4 + width * (j + 0.5)
        ax.bar(xs, vals, width=width * 0.9, color=STRATEGY_COLORS[s], label=STRATEGY_LABEL[s])
    ax.set_xticks(range(len(policies)),
                  ["threshold 0.5", "Bayes threshold\nC_FP/(C_FP+C_FN)", "threshold tuned\non OOF predictions"])
    ax.set_ylabel("expected cost per applicant (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    _save(fig, path)


def ir_small_multiples(ir_scores: pd.DataFrame, metrics: list[str], title: str, path: Path) -> None:
    """
    受控不平衡實驗：列 = 指標、欄 = 模型；x 軸 = 不平衡比 (log 尺度)。
    陰影 = ± 1 標準差。
    """
    models = [m for m in MODEL_DISPLAY if m in ir_scores.model.unique()]
    fig, axes = plt.subplots(len(metrics), len(models), figsize=(4.2 * len(models), 3.2 * len(metrics)),
                             sharex=True, squeeze=False)
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            ax = axes[i][j]
            sub = ir_scores[ir_scores.model == model]
            for s in STRATEGY_COLORS:
                g = sub[sub.strategy == s].groupby("ir")[metric].agg(["mean", "std"])
                if g.empty:
                    continue
                c = STRATEGY_COLORS[s]
                ax.plot(g.index, g["mean"], marker="o", ms=4, lw=2, color=c, label=STRATEGY_LABEL[s])
                ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.12, lw=0)
            ax.set_xscale("log")
            ax.set_xticks(sorted(ir_scores.ir.unique()), [f"{r:g}" for r in sorted(ir_scores.ir.unique())])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(MODEL_DISPLAY[model])
            if j == 0:
                ax.set_ylabel(metric)
            if i == len(metrics) - 1:
                ax.set_xlabel("imbalance ratio (majority : minority)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(title, fontweight="bold", color=TEXT)
    fig.tight_layout()
    _save(fig, path)


def summary_heatmap(cells: pd.DataFrame, title: str, path: Path) -> None:
    """
    跨資料集總覽：列 = 資料集（依不平衡比排序），欄 = 策略。
    顏色 = 各模型「策略 − none」差異的平均（藍 = 比較好、紅 = 比較差、灰白 = 差不多）；
    格子裡的文字 = 平均差異，以及「顯著較好 / 無差異 / 顯著較差」的模型數。
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    rows = cells.drop_duplicates("dataset").sort_values("ir")
    datasets = rows.dataset.tolist()
    labels = [f"{d}\n{m}, IR {ir:.0f}" for d, m, ir in zip(rows.display, rows.metric, rows.ir, strict=True)]
    strategies = [s for s in STRATEGY_COLORS if s in cells.strategy.unique()]
    grid = cells.pivot(index="dataset", columns="strategy", values="mean_diff").reindex(
        index=datasets, columns=strategies)

    lim = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0
    # 發散色階：紅（較差）— 灰白（0）— 藍（較好）
    cmap = LinearSegmentedColormap.from_list("div", ["#c23b3a", "#f4f3f0", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(2.0 * len(strategies) + 2.6, 0.85 * len(datasets) + 1.6))
    ax.imshow(grid.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i, d in enumerate(datasets):
        for j, s in enumerate(strategies):
            r = cells[(cells.dataset == d) & (cells.strategy == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.text(j, i, f"{r.mean_diff:+.3f}\n{r.better:.0f} / {r.tie:.0f} / {r.worse:.0f}",
                    ha="center", va="center", fontsize=8.5, color=TEXT)
    ax.set_xticks(range(len(strategies)), [STRATEGY_LABEL[s] for s in strategies], fontsize=9)
    ax.set_yticks(range(len(datasets)), labels, fontsize=8.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    ax.text(0.0, -0.9 / (0.85 * len(datasets) + 1.6),
            "cell: mean Δ vs. no handling (averaged over models) · counts = models significantly "
            "better / no sig. diff / significantly worse (Holm-corrected)",
            transform=ax.transAxes, fontsize=8, color=TEXT_2, va="top")
    _save(fig, path)
