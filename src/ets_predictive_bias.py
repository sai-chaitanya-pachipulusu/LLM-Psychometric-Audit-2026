"""
ets_predictive_bias.py
----------------------
Implements ETS-specific predictive fairness tests:
1. Cleary's Rule for Predictive Bias (Regression Intercept & Slope bias)
2. Standardized Mean Difference (SMD) of Residuals

Cleary (1968) established that a test is biased if the criterion score
predicted from the common regression line is consistently too high or
too low for members of a subgroup. We regress Human Score onto AI Score
and test for significant interaction terms.

Author: Sai Chaitanya Pachipulusu (Independent Psychometric Audit, 2026)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

def compute_smd(focal_residuals, ref_residuals):
    """
    Computes Standardized Mean Difference (SMD) of residuals.
    ETS generally considers |SMD| > 0.15 to be practically significant.
    """
    if len(focal_residuals) < 30 or len(ref_residuals) < 30:
        return np.nan
        
    f_mean = np.mean(focal_residuals)
    r_mean = np.mean(ref_residuals)
    
    # Pooled standard deviation
    n_f = len(focal_residuals)
    n_r = len(ref_residuals)
    var_f = np.var(focal_residuals, ddof=1)
    var_r = np.var(ref_residuals, ddof=1)
    
    pooled_sd = np.sqrt(((n_f - 1) * var_f + (n_r - 1) * var_r) / (n_f + n_r - 2))
    
    if pooled_sd == 0:
        return np.nan
        
    return (f_mean - r_mean) / pooled_sd

def test_clearys_rule(df: pd.DataFrame, focal_group: str, group_col: str = "group") -> dict:
    """
    Tests Cleary's Rule for Predictive Bias.
    Model: Human_Score ~ AI_Score + Group + AI_Score*Group
    """
    sub = df.dropna(subset=["ai_score", "human_score", group_col]).copy()
    if len(sub) < 100:
        return {}
        
    sub["is_focal"] = np.where(sub[group_col] == focal_group, 1, 0)
    sub["interaction"] = sub["ai_score"] * sub["is_focal"]
    
    # OLS Regression
    X = sm.add_constant(sub[["ai_score", "is_focal", "interaction"]])
    y = sub["human_score"]
    
    try:
        model = sm.OLS(y, X).fit()
        
        # Intercept Bias (Group coefficient)
        intercept_bias_p = float(model.pvalues.get("is_focal", np.nan))
        intercept_bias_coef = float(model.params.get("is_focal", np.nan))
        
        # Slope Bias (Interaction coefficient)
        slope_bias_p = float(model.pvalues.get("interaction", np.nan))
        slope_bias_coef = float(model.params.get("interaction", np.nan))
        
        # SMD
        sub["residual"] = sub["ai_score"] - sub["human_score"]
        focal_res = sub[sub["is_focal"] == 1]["residual"]
        ref_res = sub[sub["is_focal"] == 0]["residual"]
        smd = compute_smd(focal_res, ref_res)
        
        return {
            "Focal_Group": focal_group,
            "Target_Column": group_col,
            "Intercept_Bias_Detected": bool(intercept_bias_p < 0.05),
            "Intercept_Bias_p": round(intercept_bias_p, 6) if not np.isnan(intercept_bias_p) else np.nan,
            "Intercept_Bias_Coef": round(intercept_bias_coef, 4) if not np.isnan(intercept_bias_coef) else np.nan,
            "Slope_Bias_Detected": bool(slope_bias_p < 0.05),
            "Slope_Bias_p": round(slope_bias_p, 6) if not np.isnan(slope_bias_p) else np.nan,
            "Slope_Bias_Coef": round(slope_bias_coef, 4) if not np.isnan(slope_bias_coef) else np.nan,
            "Clearys_Rule_Violated": bool(intercept_bias_p < 0.05 or slope_bias_p < 0.05),
            "SMD_Residuals": round(smd, 4) if not np.isnan(smd) else np.nan,
            "SMD_Exceeds_ETS_Threshold": bool(abs(smd) > 0.15) if not np.isnan(smd) else False
        }
    except Exception as e:
        print(f"    [Cleary's Rule] Error fitting model: {e}")
        return {}

def run_predictive_bias_suite(df: pd.DataFrame) -> dict:
    """Run Cleary's rule across major demographics."""
    print("\n[PILLAR 7] ★ Cleary's Rule & Predictive Bias (ETS Standard) …")
    
    results = {}
    
    # 1. Base ELL group
    r_ell = test_clearys_rule(df, "Non-Native", "group")
    if r_ell:
        results["ELL"] = r_ell
        print(f"    [ELL] Cleary's Rule Violated: {r_ell['Clearys_Rule_Violated']} | SMD: {r_ell['SMD_Residuals']:.3f}")
        
    # 2. Race/Ethnicity (Black vs White, Hispanic vs White as examples if available)
    if "race_ethnicity" in df.columns:
        for race in ["Hispanic/Latino", "Black/African American"]:
            # create binary subset
            sub_df = df[df["race_ethnicity"].isin([race, "White"])].copy()
            if len(sub_df) > 100:
                r_race = test_clearys_rule(sub_df, race, "race_ethnicity")
                if r_race:
                    results[f"Race_{race}"] = r_race
                    print(f"    [{race}] Cleary's Rule Violated: {r_race['Clearys_Rule_Violated']} | SMD: {r_race['SMD_Residuals']:.3f}")

    # 3. SES
    if "economically_disadvantaged" in df.columns:
        sub_df = df[df["economically_disadvantaged"].notna()].copy()
        if len(sub_df) > 100:
            r_ses = test_clearys_rule(sub_df, "Economically disadvantaged", "economically_disadvantaged")
            if r_ses:
                results["SES"] = r_ses
                print(f"    [SES] Cleary's Rule Violated: {r_ses['Clearys_Rule_Violated']} | SMD: {r_ses['SMD_Residuals']:.3f}")

    return results

if __name__ == "__main__":
    df = pd.read_csv("data/scored_corpus.csv")
    res = run_predictive_bias_suite(df)
    import json
    print(json.dumps(res, indent=2))
