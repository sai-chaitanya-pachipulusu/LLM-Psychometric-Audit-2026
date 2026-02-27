"""
psychometric_analysis.py
------------------------
Core statistical analysis engine.  Implements SIX psychometric pillars:

  PILLAR 1 – AGREEMENT
    Quadratic Weighted Kappa (QWK) between human rater and AI scorer.
    Also reports Pearson r, RMSE, and mean absolute error.

  PILLAR 2 – FAIRNESS (Differential Item Functioning — ELL)
    Mantel-Haenszel statistic (MH-DIF) with ETS classification:
      A  = negligible DIF  (|MH D-DIF| < 1.0)
      B  = moderate DIF    (1.0 ≤ |MH D-DIF| < 1.5)
      C  = large DIF       (|MH D-DIF| ≥ 1.5)
    Reports odds-ratio, chi-squared, p-value, and ETS delta-scale.

  PILLAR 3 – CONSTRUCT-IRRELEVANT VARIANCE
    Regression of AI score on word-count, controlling for human score.
    Models AI score as a function of length to isolate construct-irrelevant
    surface feature bias.

  PILLAR 4 – SIGNIFICANCE
    Independent-samples t-test + Cohen's d comparing AI score residuals
    (AI – human) across demographic groups.
    One-way ANOVA across proficiency tiers (Low / Mid / High).

  PILLAR 5 – INTERSECTIONAL DIF  ★ NOVEL ★
    Runs MH-DIF separately for the core intersectional axes in ASAP 2.0:
      - Race/Ethnicity (race_ethnicity)
      - Gender (gender)
      - Economic Disadvantage (economically_disadvantaged)
      - Disability Status (student_disability_status)
    This is genuinely novel — very few published studies have applied
    intersectional DIF analysis to LLM-based scoring systems.

  PILLAR 6 – SUB-DIMENSIONAL FAIRNESS ★ NOVEL ★
    Runs DIF analysis on each sub-score (C1-C4) separately.
    Tests whether bias concentrates on surface-form features (C3/C4)
    vs. content features (C1/C2).

Author: Sai Chaitanya Pachipulusu (Independent Psychometric Audit, 2026)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
from typing import Tuple, Optional
from ets_predictive_bias import run_predictive_bias_suite

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 – INTER-RATER AGREEMENT
# ══════════════════════════════════════════════════════════════════════════════

def compute_qwk(
    df: pd.DataFrame,
    human_col: str = "human_score",
    ai_col: str = "ai_score",
) -> dict:
    """
    Quadratic Weighted Kappa (QWK) — the gold-standard AES reliability metric.

    Returns a results dict with QWK, Pearson r, RMSE, MAE.
    """
    human = df[human_col].astype(int)
    ai    = df[ai_col].astype(int)

    qwk   = cohen_kappa_score(human, ai, weights="quadratic")
    r, _  = stats.pearsonr(human, ai)
    rmse  = float(np.sqrt(np.mean((human - ai) ** 2)))
    mae   = float(np.mean(np.abs(human - ai)))

    result = {
        "QWK":     round(qwk, 4),
        "Pearson_r": round(r, 4),
        "RMSE":    round(rmse, 4),
        "MAE":     round(mae, 4),
        "N":       len(df),
    }
    return result


def compute_group_qwk(
    df: pd.DataFrame,
    group_col: str = "group",
    human_col: str = "human_score",
    ai_col: str = "ai_score",
) -> pd.DataFrame:
    """Compute QWK separately for each demographic group."""
    rows = []
    for grp, sub in df.groupby(group_col):
        row = compute_qwk(sub, human_col, ai_col)
        row["Group"] = grp
        rows.append(row)
    return pd.DataFrame(rows).set_index("Group")


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 – DIFFERENTIAL ITEM FUNCTIONING (Mantel-Haenszel)
# ══════════════════════════════════════════════════════════════════════════════

def mantel_haenszel_dif(
    df: pd.DataFrame,
    focal_group: str   = "Non-Native",
    group_col: str     = "group",
    match_col: str     = "human_score",   # matching variable = human score (ability proxy)
    outcome_col: str   = "ai_score",
    threshold: int     = 3,               # "high" AI score = score > threshold
) -> dict:
    """
    Mantel-Haenszel DIF analysis.

    focal_group: The minority/focal group (Non-Native speakers here).
    match_col:   Human rater score is used as the ability proxy to match
                 students — this ensures we compare students of equal
                 demonstrated ability, isolating the AI's group-specific bias.

    ETS DIF Classification (delta-scale):
      Level A (negligible): |MH D-DIF| < 1.0
      Level B (moderate):   1.0 ≤ |MH D-DIF| < 1.5
      Level C (large):      |MH D-DIF| ≥ 1.5

    Returns dict with OR, MH_chi2, p-value, MH_D_DIF, ETS_class.
    """
    focal = df[df[group_col] == focal_group]
    ref   = df[df[group_col] != focal_group]

    strata = sorted(df[match_col].unique())

    or_num_total = 0.0
    or_den_total = 0.0
    chi_num      = 0.0
    chi_den      = 0.0

    for stratum in strata:
        f_sub   = focal[focal[match_col] == stratum]
        r_sub   = ref[ref[match_col]     == stratum]

        f_n     = len(f_sub)
        r_n     = len(r_sub)

        if f_n == 0 or r_n == 0:
            continue

        f_high  = (f_sub[outcome_col] > threshold).sum()
        r_high  = (r_sub[outcome_col] > threshold).sum()
        total_n = f_n + r_n
        a_i     = f_high
        b_i     = r_high

        m1_i = f_high + r_high             # total "high" in stratum
        m0_i = f_n + r_n - m1_i           # total "low"  in stratum
        n1_i = f_n                         # focal size
        n0_i = r_n                         # reference size

        if total_n < 2:
            continue

        # MH OR components
        or_num_total += (a_i * (r_n - b_i)) / total_n
        or_den_total += (b_i * (f_n - a_i)) / total_n

        # MH chi-squared components
        expected_a_i = (n1_i * m1_i) / total_n
        var_a_i      = (n1_i * n0_i * m1_i * m0_i) / (total_n ** 2 * (total_n - 1)) if total_n > 1 else 0
        chi_num     += a_i - expected_a_i
        chi_den     += var_a_i

    # Overall odds ratio
    mh_or   = or_num_total / or_den_total if or_den_total != 0 else np.nan

    # MH chi-squared statistic (Yates-corrected)
    if chi_den > 0:
        mh_chi2 = (abs(chi_num) - 0.5) ** 2 / chi_den
    else:
        mh_chi2 = np.nan

    p_value = float(stats.chi2.sf(mh_chi2, df=1)) if not np.isnan(mh_chi2) else np.nan

    # ETS DELTA scale: MH D-DIF = -2.35 * ln(MH OR)
    mh_d_dif = -2.35 * np.log(mh_or) if mh_or > 0 else np.nan

    # ETS classification
    if np.isnan(mh_d_dif):
        ets_class = "Unknown"
    elif abs(mh_d_dif) < 1.0:
        ets_class = "A (Negligible)"
    elif abs(mh_d_dif) < 1.5:
        ets_class = "B (Moderate)"
    else:
        ets_class = "C (Large)"

    return {
        "Focal_Group":    focal_group,
        "MH_Odds_Ratio":  round(float(mh_or), 4),
        "MH_Chi2":        round(float(mh_chi2), 4),
        "p_value":        round(p_value, 6),
        "MH_D_DIF":       round(float(mh_d_dif), 4),
        "ETS_Class":      ets_class,
        "N_Focal":        len(focal),
        "N_Reference":    len(ref),
    }

def ordinal_logistic_regression_dif(
    df: pd.DataFrame,
    focal_group: str = "Non-Native",
    group_col: str = "group",
    match_col: str = "human_score",
    outcome_col: str = "ai_score"
) -> dict:
    """
    ★ NOVEL ANALYSIS ★
    Ordinal Logistic Regression (OLR) for Polytomous DIF.

    Unlike Mantel-Haenszel which requires binarizing the score (e.g. threshold=3),
    OLR analyzes the full 1-6 distribution. It tests for both:
      - Uniform DIF: Does the AI systematically score a group higher/lower across all ability levels?
      - Non-Uniform DIF: Does the AI bias change depending on the student's ability level?
    """
    # Prepare clean dataframe
    sub = df.dropna(subset=[outcome_col, match_col, group_col]).copy()
    if len(sub) < 100:
        return {}

    # Binarize group for regression: Focal = 1, Reference = 0
    sub["group_bin"] = np.where(sub[group_col] == focal_group, 1, 0)

    # OLR requires 0-indexed outcome variable
    try:
        sub["ai_score_ord"] = sub[outcome_col].astype(int) - sub[outcome_col].astype(int).min()
    except Exception:
        return {}

    # We need multiple categories to run OLR
    if sub["ai_score_ord"].nunique() < 3:
        return {}

    try:
        # Model 1: Outcome ~ Ability (Base)
        # Model 2: Outcome ~ Ability + Group (Uniform DIF)
        mod2 = OrderedModel(sub["ai_score_ord"], sub[[match_col, "group_bin"]], distr='logit').fit(method='bfgs', disp=False)

        # Model 3: Outcome ~ Ability + Group + Ability*Group (Non-Uniform DIF)
        sub["interaction"] = sub[match_col] * sub["group_bin"]
        mod3 = OrderedModel(sub["ai_score_ord"], sub[[match_col, "group_bin", "interaction"]], distr='logit').fit(method='bfgs', disp=False)

        uniform_p = float(mod2.pvalues.get("group_bin", np.nan))
        non_uniform_p = float(mod3.pvalues.get("interaction", np.nan))

        return {
            "Focal_Group": focal_group,
            "Uniform_DIF_Detected": bool(uniform_p < 0.05),
            "Uniform_p_value": round(uniform_p, 6) if not np.isnan(uniform_p) else np.nan,
            "Uniform_Beta": round(float(mod2.params.get("group_bin", np.nan)), 4),
            "NonUniform_DIF_Detected": bool(non_uniform_p < 0.05),
            "NonUniform_p_value": round(non_uniform_p, 6) if not np.isnan(non_uniform_p) else np.nan,
            "NonUniform_Interaction_Beta": round(float(mod3.params.get("interaction", np.nan)), 4),
        }
    except Exception as e:
        print(f"    [OLR DIF] Error fitting models: {e}")
        return {}




# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 – CONSTRUCT-IRRELEVANT VARIANCE (Word-Count Bias)
# ══════════════════════════════════════════════════════════════════════════════

def construct_irrelevant_variance(df: pd.DataFrame) -> dict:
    """
    Test whether essay length (word count) predicts AI score beyond
    what is explained by human score alone.

    Model A: ai_score ~ human_score
    Model B: ai_score ~ human_score + word_count

    If the word_count coefficient is significant (p < 0.05), the AI is
    introducing construct-irrelevant variance (CIV) — penalising or
    rewarding students for surface form rather than underlying ability.
    """
    df = df.dropna(subset=["ai_score", "human_score", "word_count"])

    # Standardise word_count (z-score) for interpretable coefficient
    df = df.copy()
    df["wc_z"] = (df["word_count"] - df["word_count"].mean()) / df["word_count"].std()

    # Model A – baseline
    X_a   = sm.add_constant(df["human_score"])
    mod_a = sm.OLS(df["ai_score"], X_a).fit()

    # Model B – with word count
    X_b   = sm.add_constant(df[["human_score", "wc_z"]])
    mod_b = sm.OLS(df["ai_score"], X_b).fit()

    wc_coef = mod_b.params.get("wc_z", np.nan)
    wc_pval = mod_b.pvalues.get("wc_z", np.nan)
    delta_r2 = mod_b.rsquared - mod_a.rsquared
    ci = mod_b.conf_int().loc["wc_z"].values if "wc_z" in mod_b.conf_int().index else [np.nan, np.nan]

    return {
        "ModelA_R2":               round(mod_a.rsquared, 4),
        "ModelB_R2":               round(mod_b.rsquared, 4),
        "Delta_R2_WordCount":      round(delta_r2, 4),
        "WordCount_Beta":          round(float(wc_coef), 4),
        "WordCount_Beta_CI_95":    [round(ci[0], 4), round(ci[1], 4)],
        "WordCount_p_value":       round(float(wc_pval), 6),
        "CIV_Detected":            bool(wc_pval < 0.05),
        "Interpretation": (
            "⚠  Significant construct-irrelevant variance detected — AI score is "
            "influenced by essay length beyond true ability."
            if wc_pval < 0.05 else
            "✓  No significant word-count bias detected at α=0.05."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 4 – STATISTICAL SIGNIFICANCE & EFFECT SIZE
# ══════════════════════════════════════════════════════════════════════════════

def significance_and_effect_size(
    df: pd.DataFrame,
    group_col: str    = "group",
    focal_group: str  = "Non-Native",
) -> dict:
    """
    Two-pronged significance analysis:

    (a) Independent-samples t-test on score RESIDUALS (ai_score – human_score)
        between Native and Non-Native groups.
        Cohen's d is computed as effect size.

    (b) One-way ANOVA across three proficiency tiers (Low/Mid/High) to test
        whether AI scoring error varies by ability level.
    """
    df = df.copy()
    df["residual"] = df["ai_score"] - df["human_score"]

    # ── (a) t-test ─────────────────────────────────
    focal_res = df[df[group_col] == focal_group]["residual"]
    ref_res   = df[df[group_col] != focal_group]["residual"]

    t_stat, t_p = stats.ttest_ind(focal_res, ref_res, equal_var=False)  # Welch's t

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(focal_res) - 1) * focal_res.std() ** 2 +
         (len(ref_res)   - 1) * ref_res.std()   ** 2)
        / (len(focal_res) + len(ref_res) - 2)
    )
    cohens_d = abs(focal_res.mean() - ref_res.mean()) / pooled_std if pooled_std > 0 else np.nan

    # ── (b) ANOVA across proficiency tiers ─────────
    df["tier"] = pd.cut(
        df["human_score"],
        bins=[0, 2, 4, 6],
        labels=["Low (1–2)", "Mid (3–4)", "High (5–6)"],
    )
    anova_groups = [grp["residual"].values for _, grp in df.groupby("tier")]
    f_stat, f_p  = stats.f_oneway(*anova_groups)

    # Eta-squared (effect size for ANOVA)
    grand_mean   = df["residual"].mean()
    ss_between   = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for g in anova_groups
    )
    ss_total     = ((df["residual"] - grand_mean) ** 2).sum()
    eta_sq       = ss_between / ss_total if ss_total > 0 else np.nan

    return {
        "t_test": {
            "Focal_Mean_Residual":     round(focal_res.mean(), 4),
            "Reference_Mean_Residual": round(ref_res.mean(), 4),
            "t_statistic":             round(float(t_stat), 4),
            "p_value":                 round(float(t_p), 6),
            "Cohens_d":                round(float(cohens_d), 4),
            "Effect_Magnitude":        (
                "Large"  if cohens_d >= 0.8 else
                "Medium" if cohens_d >= 0.5 else
                "Small"  if cohens_d >= 0.2 else "Negligible"
            ),
            "Significant_at_05":       bool(t_p < 0.05),
        },
        "anova": {
            "F_statistic": round(float(f_stat), 4),
            "p_value":     round(float(f_p), 6),
            "Eta_Squared": round(float(eta_sq), 4),
            "Significant_at_05": bool(f_p < 0.05),
        },
        "tier_summary": df.groupby("tier")["residual"].agg(
            Mean="mean", SD="std", N="count"
        ).round(4).to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 5 – INTERSECTIONAL DIF  ★ NOVEL ★
# ══════════════════════════════════════════════════════════════════════════════

# Demographic axes prioritized for intersectional fairness auditing.
# This focuses the DIF sweep on the four requested dimensions:
# race, gender, socioeconomic disadvantage, and disability status.
INTERSECTIONAL_AXES = {
    "race_ethnicity": {
        "col": "race_ethnicity",
        "label": "Race/Ethnicity",
        "preferred_reference": None,  # use majority group by default
    },
    "gender": {
        "col": "gender",
        "label": "Gender",
        "preferred_reference": None,
    },
    "economically_disadvantaged": {
        "col": "economically_disadvantaged",
        "label": "Economic Disadvantage",
        "preferred_reference": "No",
    },
    "student_disability_status": {
        "col": "student_disability_status",
        "label": "Disability Status",
        "preferred_reference": "No",
    },
}

_INVALID_GROUP_LABELS = {"", "unknown", "na", "n/a", "none", "null", "nan"}


def _safe_mh_dif(
    df: pd.DataFrame,
    focal_value: str,
    group_col: str,
    match_col: str = "human_score",
    outcome_col: str = "ai_score",
    threshold: int = 3,
    min_group_n: int = 30,
) -> Optional[dict]:
    """
    Run MH-DIF for a specific focal value within a group column.
    Returns None if sample is too small.
    """
    focal = df[df[group_col] == focal_value]
    ref   = df[df[group_col] != focal_value]

    if len(focal) < min_group_n or len(ref) < min_group_n:
        return None

    result = mantel_haenszel_dif(
        df,
        focal_group=focal_value,
        group_col=group_col,
        match_col=match_col,
        outcome_col=outcome_col,
        threshold=threshold,
    )
    mh_d = result.get("MH_D_DIF", np.nan)
    result["Significant_at_05"] = bool(result.get("p_value", 1.0) < 0.05)
    result["Direction"] = (
        "Focal disadvantaged" if not np.isnan(mh_d) and mh_d > 0 else
        "Focal advantaged" if not np.isnan(mh_d) and mh_d < 0 else
        "No directional effect"
    )

    return result


def _normalise_demographic_values(series: pd.Series) -> pd.Series:
    """Standardise demographic labels and drop placeholder values."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace(r"\s+", " ", regex=True)
    lowered = cleaned.str.lower()
    cleaned[lowered.isin(_INVALID_GROUP_LABELS)] = np.nan
    return cleaned


def intersectional_dif(
    df: pd.DataFrame,
    min_group_n: int = 30,
    threshold: int = 3,
) -> dict:
    """
    ★ NOVEL ANALYSIS ★
    Run Mantel-Haenszel DIF analysis across the core intersectional
    demographic axes available in ASAP 2.0.

    For each axis, we define the reference as the majority category
    (or a preferred reference when configured and available).
    Every other sufficiently represented category is evaluated as focal
    vs. that reference category.

    Returns a nested dict:
      { axis_name: { focal_value: MH-DIF result dict, ... }, ... }

    Minimum sample size per group: 30 (to ensure statistical validity).
    """
    results: dict[str, dict] = {}

    for axis_name, config in INTERSECTIONAL_AXES.items():
        col = config["col"]

        if col not in df.columns:
            print(f"    [Intersectional DIF] Skipping {axis_name}: column not in data")
            continue

        sub = df.copy()
        sub[col] = _normalise_demographic_values(sub[col])
        sub = sub[sub[col].notna()].copy()
        if len(sub) < 100:
            print(f"    [Intersectional DIF] Skipping {axis_name}: only {len(sub)} non-null rows")
            continue

        value_counts = sub[col].value_counts()
        if len(value_counts) < 2:
            print(f"    [Intersectional DIF] Skipping {axis_name}: only one category remains")
            continue

        print(f"    [{axis_name}] Values: {value_counts.to_dict()}")

        preferred_ref = config.get("preferred_reference")
        reference = None
        if preferred_ref is not None:
            for v in value_counts.index:
                if str(v).strip().lower() == str(preferred_ref).strip().lower():
                    reference = v
                    break
        if reference is None:
            reference = value_counts.index[0]

        axis_results: dict[str, dict] = {}
        for focal_val in value_counts.index:
            if focal_val == reference:
                continue
            if value_counts[focal_val] < min_group_n:
                print(f"      → Skipping {focal_val}: n={value_counts[focal_val]} < {min_group_n}")
                continue

            binary_df = sub[sub[col].isin([focal_val, reference])].copy()
            r = _safe_mh_dif(
                binary_df,
                focal_value=focal_val,
                group_col=col,
                match_col="human_score",
                outcome_col="ai_score",
                threshold=threshold,
                min_group_n=min_group_n,
            )
            if r is None:
                print(
                    f"      → DIF({focal_val} vs {reference}): "
                    f"insufficient sample (< {min_group_n} each)"
                )
                continue

            r["Axis"] = axis_name
            r["Axis_Label"] = config["label"]
            r["Focal_Label"] = str(focal_val)
            r["Reference_Label"] = str(reference)
            axis_results[str(focal_val)] = r
            print(
                f"      → DIF({focal_val} vs {reference}): "
                f"MH D-DIF = {r['MH_D_DIF']:.3f} [{r['ETS_Class']}]"
            )

        if axis_results:
            results[axis_name] = axis_results

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 6 – SUB-DIMENSIONAL DIF  ★ NOVEL ★
# ══════════════════════════════════════════════════════════════════════════════

def subdimensional_dif(
    df: pd.DataFrame,
    focal_group: str = "Non-Native",
    group_col: str = "group",
) -> dict:
    """
    ★ NOVEL ANALYSIS ★
    Run MH-DIF separately on each sub-dimensional AI score (C1–C4).

    This tests the hypothesis that bias concentrates on
    surface-form features (C3=Lexical Range, C4=Grammatical Accuracy)
    rather than content features (C1=Argumentation, C2=Coherence).

    If confirmed, this provides direct evidence that LLMs penalise
    non-native speakers for linguistic form rather than ideas — a
    crucial finding for educational fairness policy.
    """
    sub_scores = {
        "C1_Argumentation": "ai_c1",
        "C2_Coherence":     "ai_c2",
        "C3_Lexical_Range": "ai_c3",
        "C4_Grammar":       "ai_c4",
    }

    results = {}
    for label, col in sub_scores.items():
        if col not in df.columns:
            print(f"    [Sub-DIF] Skipping {label}: column '{col}' not found")
            continue

        sub = df[df[col].notna()].copy()
        if len(sub) < 100:
            continue

        # Run MH-DIF with the sub-score as outcome
        r = mantel_haenszel_dif(
            sub,
            focal_group=focal_group,
            group_col=group_col,
            match_col="human_score",
            outcome_col=col,
            threshold=3,
        )
        r["Sub_Dimension"] = label
        results[label] = r
        print(f"    [{label}] MH D-DIF = {r['MH_D_DIF']:.3f} [{r['ETS_Class']}]")

    # Also compute Cohen's d for each sub-score
    effect_sizes = {}
    for label, col in sub_scores.items():
        if col not in df.columns:
            continue
        focal_scores = df[df[group_col] == focal_group][col].dropna()
        ref_scores   = df[df[group_col] != focal_group][col].dropna()
        if len(focal_scores) < 30 or len(ref_scores) < 30:
            continue

        pooled_std = np.sqrt(
            ((len(focal_scores) - 1) * focal_scores.std() ** 2 +
             (len(ref_scores)   - 1) * ref_scores.std()   ** 2)
            / (len(focal_scores) + len(ref_scores) - 2)
        )
        d = abs(focal_scores.mean() - ref_scores.mean()) / pooled_std if pooled_std > 0 else 0
        effect_sizes[label] = {
            "Focal_Mean": round(focal_scores.mean(), 3),
            "Ref_Mean":   round(ref_scores.mean(), 3),
            "Cohens_d":   round(d, 4),
            "Effect_Magnitude": (
                "Large"  if d >= 0.8 else
                "Medium" if d >= 0.5 else
                "Small"  if d >= 0.2 else "Negligible"
            ),
        }
        print(f"    [{label}] Cohen's d = {d:.3f} ({effect_sizes[label]['Effect_Magnitude']})")

    return {
        "mh_dif_by_subdimension": results,
        "effect_sizes_by_subdimension": effect_sizes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT-LEVEL FAIRNESS (bonus analysis)
# ══════════════════════════════════════════════════════════════════════════════

def prompt_level_fairness(df: pd.DataFrame) -> dict:
    """
    Analyze whether DIF varies by writing prompt.
    Different prompts may elicit different levels of bias.
    """
    if "prompt_name" not in df.columns:
        return {}

    prompts = df["prompt_name"].dropna().unique()
    if len(prompts) < 2:
        return {}

    results = {}
    for prompt in sorted(prompts):
        sub = df[df["prompt_name"] == prompt].copy()
        if len(sub) < 60:  # need reasonable sample
            continue

        # Check we have both groups
        groups = sub["group"].value_counts()
        if len(groups) < 2 or groups.min() < 20:
            continue

        try:
            qwk_result = compute_qwk(sub)
            dif_result = mantel_haenszel_dif(sub)

            # Compute group-level mean residuals
            sub["residual"] = sub["ai_score"] - sub["human_score"]
            focal_resid = sub[sub["group"] == "Non-Native"]["residual"].mean()
            ref_resid   = sub[sub["group"] != "Non-Native"]["residual"].mean()

            results[str(prompt)] = {
                "N": len(sub),
                "QWK": qwk_result["QWK"],
                "MH_D_DIF": dif_result["MH_D_DIF"],
                "ETS_Class": dif_result["ETS_Class"],
                "Focal_Mean_Residual": round(focal_resid, 4),
                "Ref_Mean_Residual": round(ref_resid, 4),
            }
            print(f"    [Prompt: {prompt}] QWK={qwk_result['QWK']:.3f}  DIF={dif_result['MH_D_DIF']:.3f}")
        except Exception as e:
            print(f"    [Prompt: {prompt}] Error: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED REPORT
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(df: pd.DataFrame) -> dict:
    """Run all pillars and return a unified results dict."""
    print("\n" + "═" * 60)
    print("  PSYCHOMETRIC AUDIT — FULL ANALYSIS")
    print("═" * 60)

    print("\n[PILLAR 1] Computing QWK & Agreement Metrics …")
    p1_overall = compute_qwk(df)
    p1_groups  = compute_group_qwk(df)

    print("[PILLAR 2a] Running Mantel-Haenszel DIF Analysis (ELL) …")
    p2a = mantel_haenszel_dif(df)

    print("[PILLAR 2b] ★ Ordinal Logistic Regression (Polytomous DIF) …")
    p2b = ordinal_logistic_regression_dif(df)

    print("[PILLAR 3] Testing Construct-Irrelevant Variance (Word Count) …")
    p3 = construct_irrelevant_variance(df)

    print("[PILLAR 4] Computing t-test, Cohen's d, and ANOVA …")
    p4 = significance_and_effect_size(df)

    print("\n[PILLAR 5] ★ Intersectional DIF Analysis (NOVEL) …")
    p5 = intersectional_dif(df)

    print("\n[PILLAR 6] ★ Sub-Dimensional DIF (C1–C4) (NOVEL) …")
    p6 = subdimensional_dif(df)

    p_ets = run_predictive_bias_suite(df)

    print("\n[BONUS] Prompt-Level Fairness Analysis …")
    p7 = prompt_level_fairness(df)

    return {
        "pillar1_overall":        p1_overall,
        "pillar1_by_group":       p1_groups.to_dict(),
        "pillar2a_mh_dif":        p2a,
        "pillar2b_olr_dif":       p2b,
        "pillar3_civ":            p3,
        "pillar4_significance":   p4,
        "pillar5_intersectional": p5,
        "pillar6_subdimensional": p6,
        "pillar7_clearys_rule":   p_ets,
        "prompt_level_fairness":  p7,
    }


if __name__ == "__main__":
    import json
    df = pd.read_csv("data/scored_corpus.csv")
    results = run_full_analysis(df)
    print(json.dumps(results, indent=2, default=str))
