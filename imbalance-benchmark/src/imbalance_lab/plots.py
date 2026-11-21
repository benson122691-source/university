"""
圖表 (figures)
==============

設計規範（依照 dataviz 設計指引，不是憑感覺挑）：

顏色 —— 每個顏色只做一件事
- 策略 = 類別色 (categorical)，依「固定順序」取用第 1~4 號色，不跳號：
  class weight → 1 藍、random under → 2 橘、Tomek → 3 青綠、SMOTE → 4 黃。
  這個順序經過色盲模擬驗證（相鄰顏色在紅綠色盲下仍可區分），換順序就不保證了。
- 基準線 none = 灰色（「對照組」用低調的灰，讓有顏色的策略成為焦點）。
- 熱圖 = 發散色 (diverging)：藍（較好）— 中性灰（0）— 紅（較差）。
- 黃色在淺色背景上對比度不足 3:1，所以每張圖都有圖例，數字也都能在 README 表格查到。

標記 (marks)
- 線 2px、點 ≥ 8px 並加一圈背景色外框（重疊時仍看得清楚）
- 長條最寬 24px、資料端 4px 圓角、基線端保持直角
- 格線：1px 實線、比背景深一階，退到背景裡；不用虛線（虛線會被誤讀成「門檻」或「預測」）
- 文字一律用文字色（主要 / 次要 / 淡），不用資料的顏色

輸出
- 每張圖輸出三個檔：
  <name>.svg       淺色主題，向量圖：放大不會糊（README 用）
  <name>-dark.svg  深色主題（GitHub 深色模式自動切換）
  <name>.png       淺色、高解析度點陣圖（放進簡報用）
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不開視窗，直接存檔（伺服器 / CI 上也能跑）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .models import MODEL_DISPLAY  # noqa: E402

# ---------------------------------------------------------------------------
# 主題：淺色 / 深色。所有色值都來自設計指引的調色盤，沒有目測挑的顏色。
# ---------------------------------------------------------------------------
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],   # 類別色第 1~4 號
        "baseline": "#898781",
        "diverging": ("#e34948", "#f0efec", "#2a78d6"),           # 由小到大：紅（較差）— 中性灰 — 藍（較好）
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "baseline": "#898781",
        "diverging": ("#e66767", "#383835", "#3987e5"),
    },
}

# 策略 → 類別色號（固定，所有圖一致）。none 不在這裡：它用基準灰。
STRATEGY_ORDER = ["none", "class_weight", "random_under", "tomek", "smote"]
_SLOT = {"class_weight": 0, "random_under": 1, "tomek": 2, "smote": 3}
STRATEGY_LABEL = {
    "none": "none (baseline)", "class_weight": "class weight",
    "random_under": "random undersampling", "tomek": "Tomek links", "smote": "SMOTE / SMOTENC",
}

PT_PER_PX = 0.75   # 設計規格用 px，matplotlib 用 pt（1px = 0.75pt）
LINE = 2 * PT_PER_PX
MARKER = 8 * PT_PER_PX
RING = 2 * PT_PER_PX


def color(strategy: str, T: dict) -> str:
    """策略的顏色：none 用基準灰，其餘依固定色號。"""
    return T["baseline"] if strategy == "none" else T["series"][_SLOT[strategy]]


def _rc(T: dict) -> dict:
    """matplotlib 全域樣式：字型、文字色、格線。"""
    return {
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 9.5,
        "svg.fonttype": "path",  # 文字轉成向量路徑：在沒有 Inter 字型的電腦上看也一樣
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"],
        "savefig.facecolor": T["surface"], "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
        "text.color": T["ink"], "axes.labelcolor": T["ink2"], "axes.titlecolor": T["ink"],
        "axes.titleweight": "semibold", "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.titlepad": 12, "axes.edgecolor": T["axis"], "axes.linewidth": 0.75,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": T["ink2"], "ytick.color": T["ink2"],
        "xtick.labelcolor": T["ink2"], "ytick.labelcolor": T["ink2"],
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
        "axes.grid": True, "grid.color": T["grid"], "grid.linewidth": 0.75, "grid.linestyle": "-",
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
        "legend.labelcolor": T["ink2"], "lines.linewidth": LINE, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }


def render(draw: Callable[[dict], plt.Figure], path: Path) -> None:
    """
    用淺色、深色兩種主題各畫一次並存檔。

    draw 是「畫圖函式」：收到主題字典 T，回傳畫好的 Figure。
    這樣同一份畫圖邏輯只寫一次，換主題只是換顏色。
    """
    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, T in THEMES.items():
        with plt.rc_context(_rc(T)):
            fig = draw(T)
            if name == "light":
                fig.savefig(f"{path}.svg")
                fig.savefig(f"{path}.png", dpi=220)
            else:
                fig.savefig(f"{path}-dark.svg")
            plt.close(fig)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _ringed_marker_kw(c: str, T: dict, filled: bool = True) -> dict:
    """點標記：≥ 8px，外圈 2px 背景色，重疊在線上也看得清楚。空心點表示「不顯著」。"""
    return {"marker": "o", "ms": MARKER + 2 * RING, "mfc": c if filled else T["surface"],
            "mec": T["surface"] if filled else c, "mew": RING if filled else LINE}


def _rounded_bars(ax, positions, values, thickness, colors, horizontal: bool = False,
                  base: float = 0.0) -> None:
    """
    畫「資料端圓角、基線端直角」的長條。

    matplotlib 內建的 bar() 只能畫直角矩形，所以這裡自己用路徑 (Path) 畫：
    圓角半徑 4px 要換算成「資料座標」—— x、y 兩軸的單位長度不同，所以分開換算。
    必須在座標軸範圍 (xlim / ylim) 設定好之後才呼叫，換算才準確。
    """
    fig = ax.figure
    pos = ax.get_position()
    w_pt = pos.width * fig.get_figwidth() * 72
    h_pt = pos.height * fig.get_figheight() * 72
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    per_pt_x = abs(x1 - x0) / w_pt
    per_pt_y = abs(y1 - y0) / h_pt
    r_pt = 4 * PT_PER_PX
    k = 0.5523  # 用三次貝茲曲線近似四分之一圓的係數
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CLOSEPOLY]
    for p, v, c in zip(positions, values, colors, strict=True):
        lo, hi, end = p - thickness / 2, p + thickness / 2, v
        if horizontal:
            # 長條沿 x 方向：圓角在右端（資料端）
            rx = min(r_pt * per_pt_x, abs(v - base) / 2)
            ry = min(r_pt * per_pt_y, thickness / 2)
            verts = [(base, lo), (end - rx, lo), (end - rx + k * rx, lo), (end, lo + ry - k * ry),
                     (end, lo + ry), (end, hi - ry), (end, hi - ry + k * ry), (end - rx + k * rx, hi),
                     (end - rx, hi), (base, hi), (base, lo)]
        else:
            # 長條沿 y 方向：圓角在上端（資料端）
            rx = min(r_pt * per_pt_x, thickness / 2)
            ry = min(r_pt * per_pt_y, abs(v - base) / 2)
            verts = [(lo, base), (lo, end - ry), (lo, end - ry + k * ry), (lo + rx - k * rx, end),
                     (lo + rx, end), (hi - rx, end), (hi - rx + k * rx, end), (hi, end - ry + k * ry),
                     (hi, end - ry), (hi, base), (lo, base)]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=c, edgecolor="none", zorder=3))


def _max_thickness(ax, axis: str, px: float = 24) -> float:
    """24px 換算成資料座標的長度（長條寬度上限）。"""
    fig = ax.figure
    pos = ax.get_position()
    if axis == "x":
        span, size = abs(np.subtract(*ax.get_xlim())), pos.width * fig.get_figwidth() * 72
    else:
        span, size = abs(np.subtract(*ax.get_ylim())), pos.height * fig.get_figheight() * 72
    return px * PT_PER_PX * span / size


def _legend_top(ax, ncol: int = 5) -> None:
    """圖例放在標題下方、繪圖區上方，一列排開。"""
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, handlelength=2.0,
                    columnspacing=1.4, borderaxespad=0.3, numpoints=1)
    # 圖例裡的點不加背景色外框（外框會把線切斷，看起來像虛線）
    for h in leg.legend_handles:
        if hasattr(h, "set_markeredgecolor") and h.get_marker() not in (None, "None", ""):
            h.set_markeredgecolor(h.get_markerfacecolor())


def _footnote(fig, text: str, T: dict) -> None:
    """圖下方的小字註解（符號說明）。"""
    fig.text(0.01, 0.0, text, ha="left", va="top", fontsize=8, color=T["muted"],
             transform=fig.transFigure)


# ---------------------------------------------------------------------------
# 各張圖
# ---------------------------------------------------------------------------
def class_distribution(y: np.ndarray, class_names: list[str], title: str, path: Path) -> None:
    """類別分布：水平長條、由多到少排序、單一顏色（名目類別不用依數值變色）。"""
    counts = pd.Series(np.bincount(y), index=class_names).sort_values()
    total = counts.sum()

    def draw(T):
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(counts) + 1.2))
        ax.set_ylim(-0.6, len(counts) - 0.4)
        ax.set_xlim(0, counts.max() * 1.25)
        th = min(0.7, _max_thickness(ax, "y"))
        _rounded_bars(ax, range(len(counts)), counts.values, th, [T["series"][0]] * len(counts),
                      horizontal=True)
        for i, v in enumerate(counts.values):  # 數值標在長條尾端
            ax.text(v + counts.max() * 0.015, i, f"{v:,}  ({v / total:.1%})", va="center",
                    fontsize=8.5, color=T["ink2"])
        ax.set_yticks(range(len(counts)), counts.index)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("samples")
        ax.set_title(f"{title}: class sizes (imbalance ratio {counts.max() / counts.min():.1f} : 1)")
        return fig

    render(draw, path)


def cv_dotplot(scores: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    """每個模型一列、每個策略一個點：點 = CV 平均，橫線 = ± 1 標準差。"""
    agg = scores.groupby(["model", "strategy"])[metric].agg(["mean", "std"]).reset_index()
    models = [m for m in MODEL_DISPLAY if m in agg.model.unique()]
    strategies = [s for s in STRATEGY_ORDER if s in agg.strategy.unique()]

    # 每個模型的點「以模型名稱為中心」排開：只依該模型實際有的策略數量計算位置
    # （KNN 沒有 class weight、Balanced RF 只有 none，點的數量不同）
    y_of = {}
    for i, m in enumerate(models):
        have = [s for s in strategies if ((agg.model == m) & (agg.strategy == s)).any()]
        offs = np.linspace(-0.28, 0.28, len(have)) if len(have) > 1 else [0.0]
        y_of.update({(m, s): i + o for s, o in zip(have, offs, strict=True)})

    def draw(T):
        fig, ax = plt.subplots(figsize=(7.6, 0.85 * len(models) + 1.3))
        for s in strategies:
            sub = agg[agg.strategy == s].set_index("model")
            ms = [m for m in models if m in sub.index]
            ys = [y_of[(m, s)] for m in ms]
            c = color(s, T)
            ax.errorbar(sub.loc[ms, "mean"], ys, xerr=sub.loc[ms, "std"], fmt="none", ecolor=c,
                        elinewidth=LINE, capsize=0, zorder=2)
            ax.plot(sub.loc[ms, "mean"], ys, ls="none", zorder=3, label=STRATEGY_LABEL[s],
                    **_ringed_marker_kw(c, T))
        ax.set_yticks(range(len(models)), [MODEL_DISPLAY[m] for m in models])
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"{metric}  ·  mean ± 1 sd over repeated CV folds")
        ax.set_title(title, pad=30)
        _legend_top(ax)
        return fig

    render(draw, path)


def diff_forest(comp: pd.DataFrame, title: str, path: Path) -> None:
    """
    森林圖：每個「策略 − baseline」的平均差異與 95% 信賴區間。
    實心點 = Holm 校正後顯著；空心點 = 不顯著。灰色直線 = 0（跟什麼都不做一樣）。
    """
    comp = comp.copy()
    order = [m for m in MODEL_DISPLAY if m in comp.model.unique()]
    comp["m_rank"] = comp.model.map({m: i for i, m in enumerate(order)})
    comp["s_rank"] = comp.strategy.map({s: i for i, s in enumerate(STRATEGY_ORDER)})
    comp = comp.sort_values(["m_rank", "s_rank"]).reset_index(drop=True)
    metric = comp.metric.iloc[0]

    def draw(T):
        fig, ax = plt.subplots(figsize=(7.6, 0.27 * len(comp) + 1.5))
        ax.axvline(0, color=T["ink2"], lw=1, zorder=1)
        y, ticks, labels, prev = 0.0, [], [], None
        for _, r in comp.iterrows():
            if prev is not None and r.model != prev:
                y += 0.6  # 模型之間留空隙，分組更清楚
            c = color(r.strategy, T)
            ax.plot([r.ci_low, r.ci_high], [y, y], color=c, lw=LINE, zorder=2)
            ax.plot(r["diff"], y, zorder=3, **_ringed_marker_kw(c, T, r.verdict != "no sig. diff"))
            ticks.append(y)
            labels.append(f"{MODEL_DISPLAY[r.model]}  ·  {STRATEGY_LABEL[r.strategy]}")
            prev, y = r.model, y + 1
        ax.set_yticks(ticks, labels, fontsize=8.5)
        ax.set_ylim(y - 0.4, -0.6)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"Δ {metric} vs. no handling  ·  95% CI (corrected resampled t-test)")
        ax.set_title(title)
        _footnote(fig, "filled = significant after Holm correction  ·  hollow = not significant", T)
        return fig

    render(draw, path)


def pr_curves(y_test: np.ndarray, test_proba: dict, model: str, title: str, path: Path) -> None:
    """同一個模型、不同策略在 test set 上的 Precision-Recall 曲線。"""
    curves = {s: precision_recall_curve(y_test, test_proba[(model, s)][:, 1])
              for s in STRATEGY_ORDER if (model, s) in test_proba}
    prevalence = y_test.mean()

    def draw(T):
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.axhline(prevalence, color=T["muted"], lw=1, zorder=1)
        ax.text(0.99, prevalence + 0.015, f"random guess (= prevalence {prevalence:.0%})",
                ha="right", va="bottom", fontsize=8, color=T["muted"])
        for s, (prec, rec, _) in curves.items():
            ax.plot(rec, prec, color=color(s, T), label=STRATEGY_LABEL[s], zorder=3 if s == "none" else 2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("recall (share of positives caught)")
        ax.set_ylabel("precision")
        ax.set_title(title, pad=30)
        _legend_top(ax, ncol=3)
        return fig

    render(draw, path)


def calibration(y_test: np.ndarray, curves: dict[str, tuple[np.ndarray, str, str]], title: str,
                path: Path) -> None:
    """
    校準圖：把預測機率分成 10 組，x = 平均預測機率，y = 實際發生比例。對角線 = 完美校準。
    curves：{圖例文字: (正類機率, 策略名稱, 線型)}；顏色跟著策略走。
    """
    pts = {label: (calibration_curve(y_test, p, n_bins=10, strategy="quantile"), s, ls)
           for label, (p, s, ls) in curves.items()}

    def draw(T):
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        ax.plot([0, 1], [0, 1], color=T["muted"], lw=1, zorder=1)
        ax.text(0.97, 0.93, "perfect calibration", ha="right", fontsize=8, color=T["muted"])
        for label, ((frac, mean_pred), s, ls) in pts.items():
            c = color(s, T)
            ax.plot(mean_pred, frac, color=c, ls=ls, zorder=2, label=label,
                    marker="o", ms=MARKER, mfc=c, mec=T["surface"], mew=RING)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability (10 quantile bins)")
        ax.set_ylabel("observed rate")
        ax.set_title(title)
        # 左上角（對角線上方）沒有資料，圖例放這裡不會擋到線
        leg = ax.legend(loc="upper left", handlelength=2.4, borderaxespad=0.8)
        for h in leg.legend_handles:
            h.set_markeredgecolor(h.get_markerfacecolor())
        return fig

    render(draw, path)


def cost_sensitivity(sens: pd.DataFrame, model: str, title: str, path: Path) -> None:
    """
    成本敏感度：每個策略的「省下比例」減去 none 的省下比例（百分點）。

    原本直接畫各策略的省下比例，5 條線幾乎完全重疊、什麼都看不出來。
    改畫「跟基準線的差距」：0 = 跟什麼都不做一樣好；負數 = 比較差。
    """
    sub = sens[sens.model == model]
    base = sub[sub.strategy == "none"].set_index("cost_ratio").savings_vs_approve_all
    strategies = [s for s in STRATEGY_ORDER if s != "none" and s in sub.strategy.unique()]

    def draw(T):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.axhline(0, color=T["ink2"], lw=1, zorder=1)
        for s in strategies:
            g = sub[sub.strategy == s].set_index("cost_ratio").savings_vs_approve_all.sort_index()
            d = 100 * (g - base.loc[g.index])
            c = color(s, T)
            ax.plot(d.index, d.values, color=c, label=STRATEGY_LABEL[s], zorder=2,
                    marker="o", ms=MARKER, mfc=c, mec=T["surface"], mew=RING)
        ax.set_xscale("log")
        ticks = sorted(sub.cost_ratio.unique())
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("cost ratio  C_FN / C_FP   (approving a defaulter vs. rejecting a good payer)")
        ax.set_ylabel("Δ cost saved vs. no handling\n(pct. points · 0 = same as no handling)")
        ax.set_title(title, pad=30)
        _legend_top(ax, ncol=4)
        return fig

    render(draw, path)


def threshold_policies(costs: pd.DataFrame, model: str, title: str, path: Path) -> None:
    """同一模型：三種門檻規則 × 各策略的每人平均成本（越低越好）。"""
    sub = costs[costs.model == model]
    policies = ["default", "bayes", "tuned"]
    names = ["threshold 0.5", "Bayes threshold\nC_FP / (C_FP + C_FN)", "threshold tuned\non OOF predictions"]
    strategies = [s for s in STRATEGY_ORDER if s in sub.strategy.unique()]
    vals = {s: [sub[(sub.strategy == s) & (sub.policy == p)].cost_per_applicant.iloc[0] for p in policies]
            for s in strategies}

    def draw(T):
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ax.set_xlim(-0.55, len(policies) - 0.45)
        ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.12)
        slot = 0.84 / len(strategies)
        th = min(slot * 0.86, _max_thickness(ax, "x"))  # 寬度上限 24px，並留 2px 以上的間隙
        for j, s in enumerate(strategies):
            xs = np.arange(len(policies)) - 0.42 + slot * (j + 0.5)
            _rounded_bars(ax, xs, vals[s], th, [color(s, T)] * len(xs))
            ax.plot([], [], color=color(s, T), lw=6, solid_capstyle="butt", label=STRATEGY_LABEL[s])
        # 只標每組最低的那一根（選擇性標註，不在每根長條上都寫數字）
        for i in range(len(policies)):
            j_min = int(np.argmin([vals[s][i] for s in strategies]))
            x = i - 0.42 + slot * (j_min + 0.5)
            ax.text(x, vals[strategies[j_min]][i] + ax.get_ylim()[1] * 0.012, f"{vals[strategies[j_min]][i]:.3f}",
                    ha="center", va="bottom", fontsize=8, color=T["ink2"])
        ax.set_xticks(range(len(policies)), names)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("expected cost per applicant (lower is better)")
        ax.set_title(title, pad=30)
        _legend_top(ax, ncol=5)
        return fig

    render(draw, path)


def ir_small_multiples(ir_scores: pd.DataFrame, metrics: list[str], title: str, path: Path) -> None:
    """受控不平衡實驗：列 = 指標、欄 = 模型；x 軸 = 不平衡比 (log 尺度)；陰影 = ± 1 標準差。"""
    models = [m for m in MODEL_DISPLAY if m in ir_scores.model.unique()]
    irs = sorted(ir_scores.ir.unique())
    labels = {"pr_auc_lift": "PR-AUC ÷ prevalence", "f1": "F1 at threshold 0.5",
              "f1_best": "best F1 over thresholds"}

    def draw(T):
        fig, axes = plt.subplots(len(metrics), len(models), figsize=(4.0 * len(models), 2.7 * len(metrics)),
                                 sharex=True, squeeze=False)
        for i, metric in enumerate(metrics):
            for j, model in enumerate(models):
                ax = axes[i][j]
                sub = ir_scores[ir_scores.model == model]
                for s in STRATEGY_ORDER:
                    g = sub[sub.strategy == s].groupby("ir")[metric].agg(["mean", "std"])
                    if g.empty:
                        continue
                    c = color(s, T)
                    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=c,
                                    alpha=0.10, lw=0)
                    ax.plot(g.index, g["mean"], color=c, label=STRATEGY_LABEL[s],
                            marker="o", ms=MARKER, mfc=c, mec=T["surface"], mew=RING)
                ax.set_xscale("log")
                ax.set_xticks(irs, [f"{r:g}" for r in irs])
                ax.minorticks_off()
                if i == 0:
                    ax.set_title(MODEL_DISPLAY[model], fontsize=10)
                if j == 0:
                    ax.set_ylabel(labels.get(metric, metric))
                if i == len(metrics) - 1:
                    ax.set_xlabel("imbalance ratio (majority : minority)")
        handles, lab = axes[0][0].get_legend_handles_labels()
        fig.suptitle(title, x=0.01, y=0.995, ha="left", fontweight="semibold", fontsize=11, color=T["ink"])
        leg = fig.legend(handles, lab, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=len(lab),
                         handlelength=2.0)
        for h in leg.legend_handles:  # 圖例的點不加外框，避免線看起來像虛線
            h.set_markeredgecolor(h.get_markerfacecolor())
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    render(draw, path)


def summary_heatmap(cells: pd.DataFrame, title: str, path: Path) -> None:
    """
    跨資料集總覽：列 = 資料集（依不平衡比排序），欄 = 策略。
    顏色 = 各模型「策略 − none」平均差異（藍 = 較好、灰 = 差不多、紅 = 較差）；
    格內文字 = 平均差異，以及「顯著較好 / 無差異 / 顯著較差」的模型數。
    """
    rows = cells.drop_duplicates("dataset").sort_values("ir")
    datasets = rows.dataset.tolist()
    ylabels = [f"{d}\n{m} · IR {ir:.0f}" for d, m, ir in zip(rows.display, rows.metric, rows.ir, strict=True)]
    strategies = [s for s in STRATEGY_ORDER if s in cells.strategy.unique()]
    grid = cells.pivot(index="dataset", columns="strategy", values="mean_diff").reindex(
        index=datasets, columns=strategies)
    lim = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0

    def draw(T):
        cmap = LinearSegmentedColormap.from_list("div", list(T["diverging"]))
        norm = TwoSlopeNorm(0, -lim, lim)
        fig, ax = plt.subplots(figsize=(1.75 * len(strategies) + 2.6, 0.72 * len(datasets) + 1.4))
        # 每一格各畫一個方塊，格與格之間留 2px 背景色間隙
        for i in range(len(datasets)):
            for j in range(len(strategies)):
                v = grid.iloc[i, j]
                fill = cmap(norm(v)) if np.isfinite(v) else T["surface"]
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fill,
                                           edgecolor=T["surface"], lw=2 * RING))
                r = cells[(cells.dataset == datasets[i]) & (cells.strategy == strategies[j])]
                if r.empty:
                    continue
                r = r.iloc[0]
                # 格內文字顏色依底色亮度選擇：深底用白字、淺底用深色字
                lum = np.dot(to_rgb(fill), [0.2126, 0.7152, 0.0722])
                tc = "#ffffff" if lum < 0.45 else "#0b0b0b"
                ax.text(j, i - 0.12, f"{r.mean_diff:+.3f}", ha="center", va="center", fontsize=9.5,
                        fontweight="semibold", color=tc)
                ax.text(j, i + 0.2, f"{r.better:.0f} / {r.tie:.0f} / {r.worse:.0f}", ha="center",
                        va="center", fontsize=8, color=tc)
        ax.set_xlim(-0.5, len(strategies) - 0.5)
        ax.set_ylim(len(datasets) - 0.5, -0.5)
        ax.set_xticks(range(len(strategies)), [STRATEGY_LABEL[s] for s in strategies])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(datasets)), ylabels, fontsize=8.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, pad=28)
        # 註解貼著表格下緣（用座標軸座標，不受圖片高度影響）
        ax.text(-0.5, len(datasets) - 0.5 + 0.35,
                "cell: mean Δ of the primary metric vs. no handling (averaged over models)   ·   "
                "models significantly better / no sig. diff / significantly worse",
                ha="left", va="top", fontsize=8, color=T["muted"])
        return fig

    render(draw, path)
