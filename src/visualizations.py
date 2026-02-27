"""
visualizations.py
-----------------
Publication-quality figures for the psychometric audit report.

All figures are saved to reports/figures/ as high-DPI PNG files suitable
for embedding into README.md or a PDF report.

Figures produced:
  Fig 1 – Score Distribution (Human vs AI), by group  [violin + swarm]
  Fig 2 – QWK Confusion Matrix  [annotated heatmap]
  Fig 3 – DIF Visualization  [score gap across ability strata]
  Fig 4 – Word-Count Bias  [scatter with regression lines, by group]
  Fig 5 – Residual Distribution by Group  [KDE + rug plot]
  Fig 6 – Pillar Summary Dashboard  [2×2 summary card]

Author: Sai Acharya (Independent Psychometric Audit, 2026)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")   # headless / no display required

# ──────────────── Style ────────────────────────────────────────────────────

PALETTE = {
    "Native":     "#4C9FE6",    # steel blue
    "Non-Native": "#F4845F",    # warm coral
    "accent":     "#A78BFA",    # violet
    "bg":         "#0F1117",    # near-black background
    "surface":    "#1C1E2B",    # card surface
    "text":       "#E8EAF0",    # primary text
    "subtext":    "#8B92A5",    # secondary text
    "grid":       "#2A2D3E",    # grid lines
}

SCORE_LABELS = [1, 2, 3, 4, 5, 6]

FIGDIR = Path("reports/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def _apply_base_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, linestyle="--", alpha=0.6)
    if title:   ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=10)


# ──────────────── Figure 1: Score Distributions ────────────────────────────

def fig_score_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Score Distributions: Human Rater vs AI Scorer",
        color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, (col, label) in zip(axes, [("human_score", "Human Rater Score"), ("ai_score", "AI Holistic Score")]):
        sns.violinplot(
            data=df, x="group", y=col, palette=PALETTE,
            inner="box", linewidth=1.2, ax=ax,
            order=["Native", "Non-Native"],
        )
        _apply_base_style(ax, title=label, xlabel="Demographic Group", ylabel="Score (1–6)")
        ax.set_ylim(0.5, 6.5)

    plt.tight_layout()
    out = FIGDIR / "fig1_score_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ──────────────── Figure 2: QWK Confusion Matrix ──────────────────────────

def fig_qwk_matrix(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Human–AI Score Agreement Matrix (Quadratic Weighted Kappa)",
        color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.02,
    )

    groups = [("All Groups", df), ("Non-Native Only", df[df["group"] == "Non-Native"])]
    for ax, (title, sub) in zip(axes, groups):
        cm = confusion_matrix(sub["human_score"], sub["ai_score"], labels=SCORE_LABELS)
        # Normalise by row (true label)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm, annot=cm, fmt="d",
            cmap="Blues", ax=ax,
            xticklabels=SCORE_LABELS, yticklabels=SCORE_LABELS,
            linewidths=0.4, linecolor=PALETTE["grid"],
            annot_kws={"size": 9, "color": "white"},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_facecolor(PALETTE["surface"])
        ax.set_xlabel("AI Score", color=PALETTE["text"])
        ax.set_ylabel("Human Score", color=PALETTE["text"])
        ax.set_title(title, color=PALETTE["text"], fontsize=11, fontweight="bold")
        ax.tick_params(colors=PALETTE["subtext"])

    plt.tight_layout()
    out = FIGDIR / "fig2_qwk_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ──────────────── Figure 3: DIF – Score Gap Across Strata ─────────────────

def fig_dif_visualization(df: pd.DataFrame) -> Path:
    """Show mean AI score by human-score stratum, separated by group."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])

    for group, color in [("Native", PALETTE["Native"]), ("Non-Native", PALETTE["Non-Native"])]:
        sub = df[df["group"] == group]
        means = sub.groupby("human_score")["ai_score"].mean()
        std   = sub.groupby("human_score")["ai_score"].sem()
        ax.plot(means.index, means.values, marker="o", color=color, label=group, linewidth=2)
        ax.fill_between(means.index, means - std, means + std, color=color, alpha=0.15)

    # Reference line: perfect agreement
    ax.plot([1, 6], [1, 6], "w--", linewidth=1, alpha=0.4, label="Perfect Agreement")

    _apply_base_style(
        ax,
        title="DIF Analysis: AI Score by Ability Stratum (Mantel-Haenszel Matching)",
        xlabel="Human Score (Ability Proxy)",
        ylabel="Mean AI Score",
    )
    ax.set_xlim(0.8, 6.2)
    ax.set_ylim(0.8, 6.5)
    ax.set_xticks(SCORE_LABELS)
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"], fontsize=10)

    # Annotate gap at score 4
    native_mean_4   = df[(df["group"] == "Native")     & (df["human_score"] == 4)]["ai_score"].mean()
    nonnative_mean_4 = df[(df["group"] == "Non-Native")  & (df["human_score"] == 4)]["ai_score"].mean()
    gap = native_mean_4 - nonnative_mean_4
    ax.annotate(
        f"DIF Gap ≈ {gap:.2f} pts at score=4",
        xy=(4, (native_mean_4 + nonnative_mean_4) / 2),
        xytext=(4.4, nonnative_mean_4 - 0.3),
        arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]),
        color=PALETTE["accent"], fontsize=9,
    )

    plt.tight_layout()
    out = FIGDIR / "fig3_dif_visualization.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ──────────────── Figure 4: Word-Count Bias (CIV) ─────────────────────────

def fig_word_count_bias(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["bg"])

    # Left: scatter ai_score vs word_count by group
    ax = axes[0]
    for group, color in [("Native", PALETTE["Native"]), ("Non-Native", PALETTE["Non-Native"])]:
        sub = df[df["group"] == group]
        ax.scatter(sub["word_count"], sub["ai_score"], color=color, alpha=0.35, s=20, label=group)
        # regression line
        m, b = np.polyfit(sub["word_count"], sub["ai_score"], 1)
        xs = np.linspace(sub["word_count"].min(), sub["word_count"].max(), 200)
        ax.plot(xs, m * xs + b, color=color, linewidth=2)

    _apply_base_style(ax, title="AI Score vs Essay Length", xlabel="Word Count", ylabel="AI Score")
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    # Right: regression controlling for human score — residuals vs word_count
    ax = axes[1]
    from sklearn.linear_model import LinearRegression
    X   = df[["human_score"]].values
    y   = df["ai_score"].values
    reg = LinearRegression().fit(X, y)
    residuals = y - reg.predict(X)

    for group, color in [("Native", PALETTE["Native"]), ("Non-Native", PALETTE["Non-Native"])]:
        mask = df["group"] == group
        ax.scatter(df.loc[mask, "word_count"], residuals[mask], color=color, alpha=0.35, s=20, label=group)
        m, b = np.polyfit(df.loc[mask, "word_count"], residuals[mask], 1)
        xs = np.linspace(df.loc[mask, "word_count"].min(), df.loc[mask, "word_count"].max(), 200)
        ax.plot(xs, m * xs + b, color=color, linewidth=2)

    ax.axhline(0, color="white", linewidth=0.8, alpha=0.4)
    _apply_base_style(ax, title="Residual AI Score vs Essay Length\n(after controlling for Human Score)", xlabel="Word Count", ylabel="Residual (AI – predicted)")
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    plt.tight_layout()
    out = FIGDIR / "fig4_word_count_bias.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ──────────────── Figure 5: Residual KDE by Group ────────────────────────

def fig_residual_distribution(df: pd.DataFrame) -> Path:
    df = df.copy()
    df["residual"] = df["ai_score"] - df["human_score"]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])

    for group, color in [("Native", PALETTE["Native"]), ("Non-Native", PALETTE["Non-Native"])]:
        sub = df[df["group"] == group]["residual"]
        sns.kdeplot(sub, ax=ax, color=color, fill=True, alpha=0.35, linewidth=2, label=group)
        ax.axvline(sub.mean(), color=color, linestyle="--", linewidth=1.5,
                   label=f"{group} mean={sub.mean():.2f}")

    ax.axvline(0, color="white", linewidth=1, alpha=0.5, linestyle=":")
    _apply_base_style(
        ax,
        title="Distribution of Score Residuals (AI − Human) by Demographic Group",
        xlabel="Score Residual (AI Score − Human Score)",
        ylabel="Density",
    )
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"], fontsize=10)

    plt.tight_layout()
    out = FIGDIR / "fig5_residual_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ──────────────── Figure 6: Dashboard Summary ───────────────────────────

def fig_summary_dashboard(results: dict) -> Path:
    """2×2 card layout with key metrics from all four pillars."""
    fig = plt.figure(figsize=(14, 8), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Psychometric Audit — At-a-Glance Dashboard",
        color=PALETTE["text"], fontsize=16, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30)

    p1 = results.get("pillar1_overall", {})
    p2 = results.get("pillar2a_mh_dif", {})
    p2b = results.get("pillar2b_olr_dif", {})
    p7 = results.get("pillar7_clearys_rule", {}).get("ELL", {}) or results.get("pillar7_clearys_rule", {}).get("SES", {}) or {}

    card_configs = [
        {
            "title": "PILLAR 1 · Agreement",
            "subtitle": "Human–AI Reliability",
            "metrics": [
                ("Quadratic Weighted Kappa", f"{p1['QWK']:.3f}"),
                ("Pearson r", f"{p1['Pearson_r']:.3f}"),
                ("RMSE", f"{p1['RMSE']:.3f}"),
                ("MAE", f"{p1['MAE']:.3f}"),
            ],
            "badge": _qwk_badge(p1["QWK"]),
            "badge_color": _qwk_color(p1["QWK"]),
        },
        {
            "title": "PILLAR 2 · Fairness (DIF)",
            "subtitle": "Mantel-Haenszel Analysis",
            "metrics": [
                ("MH Odds Ratio", f"{p2['MH_Odds_Ratio']:.3f}"),
                ("MH D-DIF (delta)", f"{p2['MH_D_DIF']:.3f}"),
                ("Chi-squared", f"{p2['MH_Chi2']:.3f}"),
                ("p-value", f"{p2['p_value']:.4f}"),
            ],
            "badge": p2["ETS_Class"],
            "badge_color": _ets_color(p2["ETS_Class"]),
        },
        {
            "title": "PILLAR 2b · Polytomous DIF",
            "subtitle": "Ordinal Logistic Regression",
            "metrics": [
                ("Uniform Bias Detected", "YES ⚠" if p2b.get("Uniform_DIF_Detected") else "No ✓"),
                ("Uniform p-value", f"{p2b.get('Uniform_p_value', 1.0):.4f}"),
                ("Non-Uniform Bias Detected", "YES ⚠" if p2b.get("NonUniform_DIF_Detected") else "No ✓"),
                ("Non-Uniform p-value", f"{p2b.get('NonUniform_p_value', 1.0):.4f}"),
            ] if p2b else [("Data", "Not Available")],
            "badge": "⚠ Variance Collapse" if p2b and p2b.get("NonUniform_DIF_Detected") else "✓ Uniform",
            "badge_color": "#F4845F" if p2b and p2b.get("NonUniform_DIF_Detected") else "#34D399",
        },
        {
            "title": "PILLAR 7 · Predictive Validity",
            "subtitle": "Cleary's Rule (1968)",
            "metrics": [
                ("Cleary's Rule Violated", "YES ⚠" if p7.get("Clearys_Rule_Violated") else "No ✓"),
                ("Intercept p-value", f"{p7.get('Intercept_Bias_p', 1.0):.4f}"),
                ("Slope p-value", f"{p7.get('Slope_Bias_p', 1.0):.4f}"),
                ("Residual SMD", f"{p7.get('SMD_Residuals', 0.0):.4f}"),
            ] if p7 else [("Data", "Not Available")],
            "badge": "⚠ Invariant Failed" if p7 and p7.get("Clearys_Rule_Violated") else "✓ Passed",
            "badge_color": "#F4845F" if p7 and p7.get("Clearys_Rule_Violated") else "#34D399",
        },
    ]

    for i, cfg in enumerate(card_configs):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PALETTE["surface"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Title
        ax.text(0.05, 0.92, cfg["title"], color=PALETTE["text"],
                fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.83, cfg["subtitle"], color=PALETTE["subtext"],
                fontsize=8.5, transform=ax.transAxes)

        # Separator line
        ax.axhline(0.78, color=PALETTE["grid"], linewidth=0.8, xmin=0.02, xmax=0.98)

        # Metrics
        for j, (label, value) in enumerate(cfg["metrics"]):
            y_pos = 0.68 - j * 0.16
            ax.text(0.05, y_pos, label, color=PALETTE["subtext"],
                    fontsize=8.5, transform=ax.transAxes)
            ax.text(0.95, y_pos, value, color=PALETTE["text"],
                    fontsize=9, fontweight="bold",
                    transform=ax.transAxes, ha="right")

        # Badge
        ax.text(0.95, 0.92, cfg["badge"], color=cfg["badge_color"],
                fontsize=9, fontweight="bold",
                transform=ax.transAxes, ha="right",
                bbox=dict(
                    facecolor=PALETTE["bg"], edgecolor=cfg["badge_color"],
                    boxstyle="round,pad=0.3", linewidth=1.2,
                ))

    out = FIGDIR / "fig6_summary_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# helpers for dashboard badges
def _qwk_badge(qwk: float) -> str:
    if qwk >= 0.80: return "Strong"
    if qwk >= 0.60: return "Moderate"
    return "Weak"

def _qwk_color(qwk: float) -> str:
    if qwk >= 0.80: return "#34D399"
    if qwk >= 0.60: return "#FBBF24"
    return "#F4845F"

def _ets_color(cls: str) -> str:
    if "A" in cls: return "#34D399"
    if "B" in cls: return "#FBBF24"
    return "#F4845F"

def _effect_color(d: float) -> str:
    if d >= 0.8: return "#F4845F"
    if d >= 0.5: return "#FBBF24"
    if d >= 0.2: return "#4C9FE6"
    return "#8B92A5"


def fig_clearys_rule_predictive_bias(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])
    
    sub = df.dropna(subset=["ai_score", "human_score", "group"]).copy()
    
    sns.regplot(
        data=sub[sub["group"] == "Native"],
        x="ai_score", y="human_score",
        scatter_kws={"s": 15, "alpha": 0.3},
        line_kws={"color": PALETTE["Native"], "lw": 2},
        label="Native Speakers", ax=ax, color=PALETTE["Native"]
    )
    
    sns.regplot(
        data=sub[sub["group"] == "Non-Native"],
        x="ai_score", y="human_score",
        scatter_kws={"s": 15, "alpha": 0.3},
        line_kws={"color": PALETTE["Non-Native"], "lw": 2},
        label="ELL / Non-Native", ax=ax, color=PALETTE["Non-Native"]
    )
    
    _apply_base_style(ax, title="Cleary's Rule: Predictive Bias Intercept & Slope", xlabel="AI Score", ylabel="Human Score")
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])
    
    plt.tight_layout()
    out = FIGDIR / "fig7_clearys_predictive_bias.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ──────────────── Orchestrator ────────────────────────────────────────────

def generate_all_figures(df: pd.DataFrame, results: dict) -> list[Path]:
    print("\n[Visualizations] Generating publication figures …")
    figs = []
    figs.append(fig_score_distributions(df))
    figs.append(fig_qwk_matrix(df))
    figs.append(fig_dif_visualization(df))
    figs.append(fig_word_count_bias(df))
    figs.append(fig_residual_distribution(df))
    figs.append(fig_summary_dashboard(results))
    figs.append(fig_clearys_rule_predictive_bias(df))
    print(f"[Visualizations] All {len(figs)} figures saved to {FIGDIR.resolve()}")
    return figs


if __name__ == "__main__":
    import json
    df      = pd.read_csv("data/scored_corpus.csv")
    results = json.load(open("data/analysis_results.json"))
    generate_all_figures(df, results)
