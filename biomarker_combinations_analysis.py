"""
biomarker_combinations_analysis.py
===================================
Test combinations of biomarkers predicting delta (learning gain).

Approaches:
1. Multiple regression: groups of biomarkers together predicting delta
2. Pairwise interaction terms: biomarker_A * biomarker_B predicting delta
3. Composite scores: averaged biomarker groups vs delta
4. Mixed-effects with multiple predictors

All with FDR correction for multiple comparisons.
"""

import json
import os
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIRECTORY = os.path.join(SCRIPT_DIR, "participant_info")

positive_emotions = ["Happy", "Surprise"]
negative_emotions = ["Fear", "Sad", "Disgust", "Anger"]
all_emotions = positive_emotions + negative_emotions + ["Neutral"]

all_biomarkers = (
    all_emotions
    + ["positive", "negative", "non_neutral"]
    + [f"blinks_{b}" for b in ["short", "medium", "long"]]
    + [f"fixations_{d}_{disp}" for d in ["short", "medium", "long"]
       for disp in ["low_dispersion", "high_dispersion"]]
    + [f"pupil_diameter_{s}" for s in ["small", "medium", "large"]]
)


def calculate_percentages(metrics):
    """Calculate percentages, returning NaN for zero-total modalities."""
    pcts = {}
    emotions = metrics.get("emotions", {})
    total_emotions = sum(emotions.values()) if emotions else 0
    if total_emotions == 0:
        for e in all_emotions:
            pcts[e] = np.nan
        pcts["positive"] = pcts["negative"] = pcts["non_neutral"] = np.nan
    else:
        for e in all_emotions:
            pcts[e] = emotions.get(e, 0) / total_emotions * 100
        pcts["positive"] = sum(emotions.get(e, 0) for e in positive_emotions) / total_emotions * 100
        pcts["negative"] = sum(emotions.get(e, 0) for e in negative_emotions) / total_emotions * 100
        pcts["non_neutral"] = sum(emotions.get(e, 0) for e in all_emotions if e != "Neutral") / total_emotions * 100

    blinks = metrics.get("blinks", {})
    tb = sum(blinks.values()) if blinks else 0
    for b in ["short", "medium", "long"]:
        pcts[f"blinks_{b}"] = blinks.get(b, 0) / tb * 100 if tb > 0 else np.nan

    fixations = metrics.get("fixations", {})
    tf = sum(sum(d.values()) for d in fixations.values() if isinstance(d, dict)) if fixations else 0
    for dur in ["short", "medium", "long"]:
        for disp in ["low_dispersion", "high_dispersion"]:
            c = fixations.get(dur, {}).get(disp, 0)
            pcts[f"fixations_{dur}_{disp}"] = c / tf * 100 if tf > 0 else np.nan

    pupil = metrics.get("pupil_diameter", {})
    tp = sum(pupil.values()) if pupil else 0
    for s in ["small", "medium", "large"]:
        pcts[f"pupil_diameter_{s}"] = pupil.get(s, 0) / tp * 100 if tp > 0 else np.nan

    ecg = metrics.get("ecg_metrics", {})
    for k, v in ecg.items():
        pcts[f"ecg_{k}"] = v if v is not None else np.nan

    return pcts


def calculate_ecg_differences(video_ecg, break_ecg):
    differences = {}
    for key in video_ecg:
        if key.startswith('ecg_'):
            v, b = video_ecg.get(key), break_ecg.get(key)
            if v is not None and b is not None and not np.isnan(v) and not np.isnan(b):
                differences[f"diff_{key}"] = v - b
            else:
                differences[f"diff_{key}"] = np.nan
    return differences


def build_dataframe(directory):
    rows = []
    for jf in sorted(os.listdir(directory)):
        if not jf.endswith(".json"):
            continue
        with open(os.path.join(directory, jf)) as f:
            data = json.load(f)
        pid = data[0]["participant_id"]
        for entry in data:
            vp = calculate_percentages(entry)
            bp = calculate_percentages(entry.get("break_metrics", {}))
            change = {}
            for bm in all_biomarkers:
                if not bm.startswith("ecg_"):
                    a, b = vp.get(bm), bp.get(bm)
                    change[bm] = (a - b) if (a is not None and b is not None and
                                              not np.isnan(a) and not np.isnan(b)) else np.nan
            ecg_diffs = calculate_ecg_differences(vp, bp)
            rows.append({
                "participant": str(pid),
                "video": entry.get("video", ""),
                "difficulty": entry.get("difficulty", "unknown"),
                "delta": entry["delta"],
                **change, **ecg_diffs,
            })
    return pd.DataFrame(rows)


def safe_pearsonr(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    if len(xv) < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan, np.nan
    return pearsonr(xv, yv)


# ---------------------------------------------------------------------------
# Analysis 1: Composite biomarker scores vs delta
# ---------------------------------------------------------------------------

def test_composite_scores(df, metric_cols):
    """Create composite scores by averaging biomarker groups, then correlate with delta."""
    groups = {
        "emotion_composite": [c for c in metric_cols if any(e in c.lower() for e in
                              ['happy', 'sad', 'fear', 'disgust', 'angry', 'surprise',
                               'neutral', 'positive', 'negative', 'non_neutral'])
                              and not c.startswith('diff_ecg_')],
        "blink_composite": [c for c in metric_cols if 'blink' in c],
        "fixation_composite": [c for c in metric_cols if 'fixation' in c],
        "pupil_composite": [c for c in metric_cols if 'pupil' in c],
        "ecg_composite": [c for c in metric_cols if c.startswith('diff_ecg_')],
        "eye_tracking_composite": [c for c in metric_cols if any(e in c for e in
                                   ['blink', 'fixation', 'pupil'])],
        "all_biomarkers_composite": metric_cols,
    }

    results = []
    for group_name, cols in groups.items():
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue

        # Z-score normalize each feature, then average
        z_scores = df[valid_cols].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
        composite = z_scores.mean(axis=1)

        r, p = safe_pearsonr(composite.values, df["delta"].values)
        n_obs = int((~composite.isna() & ~df["delta"].isna()).sum())

        results.append({
            "composite": group_name,
            "n_features": len(valid_cols),
            "n_obs": n_obs,
            "pearson_r": round(r, 4) if not np.isnan(r) else None,
            "p_value": round(p, 6) if not np.isnan(p) else None,
        })

    res_df = pd.DataFrame(results)
    if not res_df.empty and res_df["p_value"].notna().any():
        pvals = res_df["p_value"].fillna(1.0).values
        _, adj, _, _ = multipletests(pvals, method="fdr_bh")
        res_df["p_value_fdr"] = [round(float(a), 6) for a in adj]

    return res_df


# ---------------------------------------------------------------------------
# Analysis 2: Per-participant composite scores
# ---------------------------------------------------------------------------

def test_composite_per_participant(df, metric_cols):
    """Composite scores per participant."""
    groups = {
        "emotion": [c for c in metric_cols if any(e in c.lower() for e in
                    ['happy', 'sad', 'fear', 'disgust', 'angry', 'surprise',
                     'neutral', 'positive', 'negative', 'non_neutral'])
                    and not c.startswith('diff_ecg_')],
        "eye_tracking": [c for c in metric_cols if any(e in c for e in
                         ['blink', 'fixation', 'pupil'])],
        "ecg": [c for c in metric_cols if c.startswith('diff_ecg_')],
    }

    results = []
    for pid in sorted(df["participant"].unique()):
        pdf = df[df["participant"] == pid]
        for group_name, cols in groups.items():
            valid_cols = [c for c in cols if c in pdf.columns]
            if not valid_cols:
                continue
            z_scores = pdf[valid_cols].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
            composite = z_scores.mean(axis=1)
            r, p = safe_pearsonr(composite.values, pdf["delta"].values)
            results.append({
                "participant": pid,
                "composite": group_name,
                "n_features": len(valid_cols),
                "n_obs": len(pdf),
                "pearson_r": round(r, 4) if not np.isnan(r) else None,
                "p_value": round(p, 6) if not np.isnan(p) else None,
            })

    res_df = pd.DataFrame(results)
    if not res_df.empty and res_df["p_value"].notna().any():
        pvals = res_df["p_value"].fillna(1.0).values
        _, adj, _, _ = multipletests(pvals, method="fdr_bh")
        res_df["p_value_fdr"] = [round(float(a), 6) for a in adj]

    return res_df


# ---------------------------------------------------------------------------
# Analysis 3: Pairwise interaction terms (mixed-effects)
# ---------------------------------------------------------------------------

def test_pairwise_interactions(df, metric_cols):
    """Test biomarker_A * biomarker_B interaction predicting delta via mixed-effects."""
    # Select top features by absolute mixed-effects coefficient to limit combinations
    # First, rank features by their individual effect size
    individual_effects = []
    for feat in metric_cols:
        mdf = df[["participant", "difficulty", "delta", feat]].dropna()
        if len(mdf) < 6 or mdf["participant"].nunique() < 2:
            continue
        mdf2 = mdf.rename(columns={feat: "fv"})
        try:
            model = smf.mixedlm("delta ~ fv + C(difficulty)", mdf2, groups=mdf2["participant"])
            fit = model.fit(reml=False)
            coef = fit.params.get("fv", 0)
            pval = fit.pvalues.get("fv", 1)
            individual_effects.append((feat, abs(coef), pval))
        except:
            pass

    # Sort by absolute effect size, take top 8
    individual_effects.sort(key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in individual_effects[:8]]

    print(f"  Testing pairwise interactions among top {len(top_features)} features:")
    for f, c, p in individual_effects[:8]:
        print(f"    {f}: |coef|={c:.4f}, p={p:.4f}")

    results = []
    pairs = list(itertools.combinations(top_features, 2))

    for feat_a, feat_b in pairs:
        mdf = df[["participant", "difficulty", "delta", feat_a, feat_b]].dropna()
        if len(mdf) < 8 or mdf["participant"].nunique() < 2:
            continue

        mdf = mdf.rename(columns={feat_a: "A", feat_b: "B"})
        try:
            # Model with interaction
            model = smf.mixedlm("delta ~ A * B + C(difficulty)", mdf, groups=mdf["participant"])
            fit = model.fit(reml=False)

            # Get interaction term
            interaction_key = "A:B"
            coef_int = fit.params.get(interaction_key)
            p_int = fit.pvalues.get(interaction_key)

            # Also get individual effects in the combined model
            coef_a = fit.params.get("A")
            p_a = fit.pvalues.get("A")
            coef_b = fit.params.get("B")
            p_b = fit.pvalues.get("B")

            results.append({
                "feature_A": feat_a,
                "feature_B": feat_b,
                "n_obs": len(mdf),
                "coef_A": round(float(coef_a), 6) if coef_a is not None else None,
                "p_A": round(float(p_a), 6) if p_a is not None else None,
                "coef_B": round(float(coef_b), 6) if coef_b is not None else None,
                "p_B": round(float(p_b), 6) if p_b is not None else None,
                "coef_interaction": round(float(coef_int), 6) if coef_int is not None else None,
                "p_interaction": round(float(p_int), 6) if p_int is not None else None,
            })
        except Exception as exc:
            pass

    res_df = pd.DataFrame(results)
    if not res_df.empty and res_df["p_interaction"].notna().any():
        pvals = res_df["p_interaction"].fillna(1.0).values
        _, adj, _, _ = multipletests(pvals, method="fdr_bh")
        res_df["p_interaction_fdr"] = [round(float(a), 6) for a in adj]

    return res_df


# ---------------------------------------------------------------------------
# Analysis 4: Multi-predictor mixed-effects (grouped)
# ---------------------------------------------------------------------------

def test_multipredictor_models(df, metric_cols):
    """Test grouped multi-predictor mixed-effects models."""
    groups = {
        "emotion_model": [c for c in metric_cols if any(e in c.lower() for e in
                          ['positive', 'negative', 'non_neutral', 'neutral'])
                          and not c.startswith('diff_ecg_')
                          and c in ['positive', 'negative', 'non_neutral', 'Neutral']],
        "blink_model": [c for c in metric_cols if 'blink' in c],
        "fixation_low_disp_model": [c for c in metric_cols if 'fixation' in c and 'low' in c],
        "pupil_model": [c for c in metric_cols if 'pupil' in c],
        "ecg_model": [c for c in metric_cols if c.startswith('diff_ecg_')],
        "cross_modal_top": [],  # Will be filled with top features from different modalities
    }

    # Build cross-modal model from top feature per modality
    modality_top = {}
    for feat in metric_cols:
        mdf = df[["participant", "difficulty", "delta", feat]].dropna()
        if len(mdf) < 6 or mdf["participant"].nunique() < 2:
            continue
        mdf2 = mdf.rename(columns={feat: "fv"})
        try:
            model = smf.mixedlm("delta ~ fv + C(difficulty)", mdf2, groups=mdf2["participant"])
            fit = model.fit(reml=False)
            pval = fit.pvalues.get("fv", 1)
            if 'blink' in feat:
                mod = 'blink'
            elif 'fixation' in feat:
                mod = 'fixation'
            elif 'pupil' in feat:
                mod = 'pupil'
            elif feat.startswith('diff_ecg_'):
                mod = 'ecg'
            else:
                mod = 'emotion'
            if mod not in modality_top or pval < modality_top[mod][1]:
                modality_top[mod] = (feat, pval)
        except:
            pass

    groups["cross_modal_top"] = [v[0] for v in modality_top.values()]

    results = []
    for model_name, features in groups.items():
        features = [f for f in features if f in df.columns]
        if len(features) < 2:
            continue

        mdf = df[["participant", "difficulty", "delta"] + features].dropna()
        if len(mdf) < 8 or mdf["participant"].nunique() < 2:
            continue

        # Rename features for formula
        rename_map = {f: f"x{i}" for i, f in enumerate(features)}
        mdf = mdf.rename(columns=rename_map)

        predictors = " + ".join(rename_map.values())
        formula = f"delta ~ {predictors} + C(difficulty)"

        try:
            model = smf.mixedlm(formula, mdf, groups=mdf["participant"])
            fit = model.fit(reml=False)

            # Model-level stats
            ll = fit.llf
            aic = -2 * ll + 2 * len(fit.params)

            for feat, renamed in rename_map.items():
                coef = fit.params.get(renamed)
                pval = fit.pvalues.get(renamed)
                ci = fit.conf_int().loc[renamed].tolist() if renamed in fit.conf_int().index else [None, None]

                results.append({
                    "model": model_name,
                    "feature": feat,
                    "n_obs": len(mdf),
                    "n_predictors": len(features),
                    "coef": round(float(coef), 6) if coef is not None else None,
                    "ci_low": round(float(ci[0]), 6) if ci[0] is not None else None,
                    "ci_high": round(float(ci[1]), 6) if ci[1] is not None else None,
                    "p_value": round(float(pval), 6) if pval is not None else None,
                    "model_aic": round(aic, 2),
                })
        except Exception as exc:
            print(f"  Multi-predictor model '{model_name}' failed: {exc}")

    res_df = pd.DataFrame(results)
    if not res_df.empty and res_df["p_value"].notna().any():
        pvals = res_df["p_value"].fillna(1.0).values
        _, adj, _, _ = multipletests(pvals, method="fdr_bh")
        res_df["p_value_fdr"] = [round(float(a), 6) for a in adj]

    return res_df


# ---------------------------------------------------------------------------
# Analysis 5: All pairwise biomarker correlations with delta (pooled)
# ---------------------------------------------------------------------------

def test_all_pairwise_pooled(df, metric_cols):
    """
    For every pair of biomarkers, create a combined score (average of z-scores)
    and test its correlation with delta across all observations pooled.
    Only test the top 10 individual features to keep combinations manageable.
    """
    # Rank features by absolute pooled correlation
    ranked = []
    for feat in metric_cols:
        r, p = safe_pearsonr(df[feat].values, df["delta"].values)
        if not np.isnan(r):
            ranked.append((feat, abs(r), p))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_feats = [f[0] for f in ranked[:10]]

    print(f"\n  Top 10 individual features (pooled across all participants):")
    for f, r, p in ranked[:10]:
        print(f"    {f}: |r|={r:.4f}, p={p:.4f}")

    results = []
    for fa, fb in itertools.combinations(top_feats, 2):
        # Combined z-score
        za = (df[fa] - df[fa].mean()) / df[fa].std() if df[fa].std() > 0 else df[fa]
        zb = (df[fb] - df[fb].mean()) / df[fb].std() if df[fb].std() > 0 else df[fb]
        combined = (za + zb) / 2

        r, p = safe_pearsonr(combined.values, df["delta"].values)
        n_obs = int((~combined.isna() & ~df["delta"].isna()).sum())

        results.append({
            "feature_A": fa,
            "feature_B": fb,
            "n_obs": n_obs,
            "combined_r": round(r, 4) if not np.isnan(r) else None,
            "combined_p": round(p, 6) if not np.isnan(p) else None,
        })

    res_df = pd.DataFrame(results)
    if not res_df.empty and res_df["combined_p"].notna().any():
        pvals = res_df["combined_p"].fillna(1.0).values
        _, adj, _, _ = multipletests(pvals, method="fdr_bh")
        res_df["combined_p_fdr"] = [round(float(a), 6) for a in adj]

    return res_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BIOMARKER COMBINATIONS ANALYSIS")
    print("=" * 70)

    df = build_dataframe(DIRECTORY)
    metric_cols = [c for c in df.columns if c not in ["participant", "video", "difficulty", "delta"]]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\nLoaded {len(df)} observations, {df['participant'].nunique()} participants, {len(metric_cols)} biomarkers\n")

    output_dir = os.path.join(SCRIPT_DIR, "results-video-break")
    os.makedirs(output_dir, exist_ok=True)

    # --- Analysis 1: Composite scores (pooled) ---
    print("=" * 50)
    print("ANALYSIS 1: COMPOSITE BIOMARKER SCORES (POOLED)")
    print("=" * 50)
    composite_df = test_composite_scores(df, metric_cols)
    composite_df.to_csv(os.path.join(output_dir, "composite_scores_pooled.csv"), index=False)
    print(composite_df.to_string(index=False))

    # --- Analysis 2: Composite scores (per participant) ---
    print("\n" + "=" * 50)
    print("ANALYSIS 2: COMPOSITE SCORES (PER PARTICIPANT)")
    print("=" * 50)
    composite_part_df = test_composite_per_participant(df, metric_cols)
    composite_part_df.to_csv(os.path.join(output_dir, "composite_scores_per_participant.csv"), index=False)
    print(composite_part_df.to_string(index=False))

    # --- Analysis 3: Pairwise interaction terms ---
    print("\n" + "=" * 50)
    print("ANALYSIS 3: PAIRWISE INTERACTION TERMS (MIXED-EFFECTS)")
    print("=" * 50)
    interaction_df = test_pairwise_interactions(df, metric_cols)
    interaction_df.to_csv(os.path.join(output_dir, "pairwise_interactions.csv"), index=False)
    if not interaction_df.empty:
        print(interaction_df.to_string(index=False))
    else:
        print("  No valid interaction models produced.")

    # --- Analysis 4: Multi-predictor mixed-effects models ---
    print("\n" + "=" * 50)
    print("ANALYSIS 4: MULTI-PREDICTOR MIXED-EFFECTS MODELS")
    print("=" * 50)
    multi_df = test_multipredictor_models(df, metric_cols)
    multi_df.to_csv(os.path.join(output_dir, "multipredictor_models.csv"), index=False)
    if not multi_df.empty:
        print(multi_df.to_string(index=False))
    else:
        print("  No valid multi-predictor models produced.")

    # --- Analysis 5: All pairwise combined scores (pooled) ---
    print("\n" + "=" * 50)
    print("ANALYSIS 5: PAIRWISE COMBINED SCORES (POOLED)")
    print("=" * 50)
    pairwise_df = test_all_pairwise_pooled(df, metric_cols)
    pairwise_df.to_csv(os.path.join(output_dir, "pairwise_combined_scores.csv"), index=False)
    if not pairwise_df.empty:
        # Show top 10 by raw p-value
        top = pairwise_df.sort_values("combined_p").head(15)
        print(top.to_string(index=False))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY OF SIGNIFICANT FINDINGS (FDR q < 0.05)")
    print("=" * 70)

    any_sig = False

    print("\nComposite scores (pooled):")
    if "p_value_fdr" in composite_df.columns:
        sig = composite_df[composite_df["p_value_fdr"].notna() & (composite_df["p_value_fdr"] < 0.05)]
        if not sig.empty:
            print(sig.to_string(index=False))
            any_sig = True
        else:
            print("  None significant.")

    print("\nComposite scores (per participant):")
    if "p_value_fdr" in composite_part_df.columns:
        sig = composite_part_df[composite_part_df["p_value_fdr"].notna() & (composite_part_df["p_value_fdr"] < 0.05)]
        if not sig.empty:
            print(sig.to_string(index=False))
            any_sig = True
        else:
            print("  None significant.")

    print("\nPairwise interactions:")
    if not interaction_df.empty and "p_interaction_fdr" in interaction_df.columns:
        sig = interaction_df[interaction_df["p_interaction_fdr"].notna() & (interaction_df["p_interaction_fdr"] < 0.05)]
        if not sig.empty:
            print(sig.to_string(index=False))
            any_sig = True
        else:
            print("  None significant.")

    print("\nMulti-predictor models:")
    if not multi_df.empty and "p_value_fdr" in multi_df.columns:
        sig = multi_df[multi_df["p_value_fdr"].notna() & (multi_df["p_value_fdr"] < 0.05)]
        if not sig.empty:
            print(sig.to_string(index=False))
            any_sig = True
        else:
            print("  None significant.")

    print("\nPairwise combined scores (pooled):")
    if not pairwise_df.empty and "combined_p_fdr" in pairwise_df.columns:
        sig = pairwise_df[pairwise_df["combined_p_fdr"].notna() & (pairwise_df["combined_p_fdr"] < 0.05)]
        if not sig.empty:
            print(sig.to_string(index=False))
            any_sig = True
        else:
            print("  None significant.")

    if not any_sig:
        print("\n>>> NO BIOMARKER COMBINATIONS REACHED SIGNIFICANCE AFTER FDR CORRECTION <<<")

    # Also report uncorrected promising results
    print("\n" + "=" * 70)
    print("PROMISING RESULTS (raw p < 0.05, NOT corrected)")
    print("=" * 70)

    if not interaction_df.empty:
        raw_sig = interaction_df[interaction_df["p_interaction"].notna() & (interaction_df["p_interaction"] < 0.05)]
        if not raw_sig.empty:
            print("\nPairwise interactions (raw p < 0.05):")
            print(raw_sig.to_string(index=False))

    if not multi_df.empty:
        raw_sig = multi_df[multi_df["p_value"].notna() & (multi_df["p_value"] < 0.05)]
        if not raw_sig.empty:
            print("\nMulti-predictor terms (raw p < 0.05):")
            print(raw_sig.to_string(index=False))

    if not pairwise_df.empty:
        raw_sig = pairwise_df[pairwise_df["combined_p"].notna() & (pairwise_df["combined_p"] < 0.05)]
        if not raw_sig.empty:
            print("\nPairwise combined scores (raw p < 0.05):")
            print(raw_sig.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"All CSV outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
