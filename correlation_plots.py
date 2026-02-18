"""
correlation_plots.py
====================
Primary analysis script for the biomarker-learning correlation study.

Addresses two reviewer concerns:

Concern 1 -- Statistical rigour:
  * Pearson correlations with 95% bootstrap confidence intervals
  * P-values with FDR (Benjamini-Hochberg) correction for multiple comparisons
  * Linear mixed-effects models (participant as random intercept)
  * Permutation tests for robustness (5000 permutations)

Concern 2 -- Missing data handling:
  * Detect truly missing data (zero totals ≠ "no emotion/fixation")
  * Report missingness rates and reasons per participant per modality
  * Within-participant median imputation
  * Sensitivity analysis comparing imputed vs non-imputed results

Outputs are saved to  results-video-break/  (plots and CSVs).
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIRECTORY = os.path.join(SCRIPT_DIR, "participant_info")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results-video-break")

N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000
RANDOM_SEED = 42

# Define emotion categories for grouping
positive_emotions = ["Happy", "Surprise"]
negative_emotions = ["Fear", "Sad", "Disgust", "Anger"]
all_emotions = positive_emotions + negative_emotions + ["Neutral"]

# Define all possible biomarkers
all_biomarkers = (
    all_emotions
    + ["positive", "negative", "non_neutral"]
    + [f"blinks_{blink}" for blink in ["short", "medium", "long"]]
    + [
        f"fixations_{duration}_{dispersion}"
        for duration in ["short", "medium", "long"]
        for dispersion in ["low_dispersion", "high_dispersion"]
    ]
    + [f"pupil_diameter_{size}" for size in ["small", "medium", "large"]]
)

ECG_DIFF_KEYS = [f"diff_ecg_{m}" for m in ["heart_rate", "sdnn", "rmssd", "pnn50"]]

ALL_METRIC_KEYS = [b for b in all_biomarkers if not b.startswith("ecg_")] + ECG_DIFF_KEYS


def get_display_name(metric):
    """Convert metric names to display format."""
    if metric.startswith('diff_ecg_'):
        metric_name = metric.replace('diff_ecg_', '')
        names = {'heart_rate': 'Heart Rate Change', 'sdnn': 'HRV Change',
                 'rmssd': 'RMSSD Change', 'pnn50': 'PNN50 Change'}
        return names.get(metric_name, metric_name.title())
    if metric == 'positive':
        return 'Positive Emotions'
    elif metric == 'negative':
        return 'Negative Emotions'
    elif metric == 'non_neutral':
        return 'Non-Neutral Emotions'
    name = metric.replace('_', ' ').title()
    name = name.replace('Ecg ', 'ECG ').replace('Pct ', 'Percentage ').replace('Disp ', 'Dispersion ')
    return name


# ---------------------------------------------------------------------------
# Missing-data-aware percentage calculation (Concern 2)
# ---------------------------------------------------------------------------

def calculate_percentages(metrics):
    """
    Calculate the percentage of each biomarker within the given metrics.

    Returns NaN (not 0) when the total count for a modality is zero,
    because zero total means the sensor produced no data (tracker loss,
    blink occlusion, etc.), not that the participant showed no response.
    """
    percentages = {}
    missing_flags = {}

    # --- Emotions ---
    emotions = metrics.get("emotions", {})
    total_emotions = sum(emotions.values()) if emotions else 0
    if total_emotions == 0:
        missing_flags["emotions"] = "no_samples" if not emotions else "zero_total_counts"
        for emotion in all_emotions:
            percentages[emotion] = np.nan
        percentages["positive"] = np.nan
        percentages["negative"] = np.nan
        percentages["non_neutral"] = np.nan
    else:
        for emotion in all_emotions:
            percentages[emotion] = emotions.get(emotion, 0) / total_emotions * 100
        positive_sum = sum(emotions.get(emo, 0) for emo in positive_emotions)
        negative_sum = sum(emotions.get(emo, 0) for emo in negative_emotions)
        non_neutral_sum = sum(emotions.get(emo, 0) for emo in all_emotions if emo != "Neutral")
        percentages["positive"] = positive_sum / total_emotions * 100
        percentages["negative"] = negative_sum / total_emotions * 100
        percentages["non_neutral"] = non_neutral_sum / total_emotions * 100

    # --- Blinks ---
    blinks = metrics.get("blinks", {})
    total_blinks = sum(blinks.values()) if blinks else 0
    if total_blinks == 0:
        missing_flags["blinks"] = "no_samples" if not blinks else "zero_blinks_possible_tracker_loss"
        for blink in ["short", "medium", "long"]:
            percentages[f"blinks_{blink}"] = np.nan
    else:
        for blink in ["short", "medium", "long"]:
            percentages[f"blinks_{blink}"] = blinks.get(blink, 0) / total_blinks * 100

    # --- Fixations ---
    fixations = metrics.get("fixations", {})
    total_fixations = (
        sum(sum(d.values()) for d in fixations.values() if isinstance(d, dict))
        if fixations else 0
    )
    if total_fixations == 0:
        missing_flags["fixations"] = "no_samples" if not fixations else "zero_fixations_possible_tracker_loss"
        for duration in ["short", "medium", "long"]:
            for dispersion in ["low_dispersion", "high_dispersion"]:
                percentages[f"fixations_{duration}_{dispersion}"] = np.nan
    else:
        for duration in ["short", "medium", "long"]:
            for dispersion in ["low_dispersion", "high_dispersion"]:
                count = fixations.get(duration, {}).get(dispersion, 0)
                percentages[f"fixations_{duration}_{dispersion}"] = count / total_fixations * 100

    # --- Pupil diameter ---
    pupil_diameter = metrics.get("pupil_diameter", {})
    total_pupil = sum(pupil_diameter.values()) if pupil_diameter else 0
    if total_pupil == 0:
        missing_flags["pupil"] = "no_samples" if not pupil_diameter else "zero_readings_possible_tracker_loss"
        for size in ["small", "medium", "large"]:
            percentages[f"pupil_diameter_{size}"] = np.nan
    else:
        for size in ["small", "medium", "large"]:
            percentages[f"pupil_diameter_{size}"] = pupil_diameter.get(size, 0) / total_pupil * 100

    # --- ECG metrics (store raw values, not percentages) ---
    ecg_metrics = metrics.get("ecg_metrics", {})
    if not ecg_metrics:
        missing_flags["ecg"] = "no_ecg_data"
    for key, value in ecg_metrics.items():
        percentages[f"ecg_{key}"] = value if value is not None else np.nan

    return percentages, missing_flags


def calculate_ecg_differences(video_ecg, break_ecg):
    """Calculate the difference between video and break ECG metrics."""
    differences = {}
    for key in video_ecg:
        if key.startswith('ecg_'):
            v = video_ecg.get(key)
            b = break_ecg.get(key)
            if v is not None and b is not None and not np.isnan(v) and not np.isnan(b):
                differences[f"diff_{key}"] = v - b
            else:
                differences[f"diff_{key}"] = np.nan
    return differences


# ---------------------------------------------------------------------------
# Statistical helpers (Concern 1)
# ---------------------------------------------------------------------------

def safe_pearsonr(x, y):
    """Return (r, p) or (NaN, NaN) on failure."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    if len(xv) < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan, np.nan
    return pearsonr(xv, yv)


def bootstrap_ci(x, y, n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    """Bootstrap 95% CI for Pearson r."""
    rng = np.random.RandomState(seed)
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan, np.nan, np.nan

    boot_rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        xb, yb = xv[idx], yv[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            boot_rs[b] = np.nan
        else:
            boot_rs[b] = np.corrcoef(xb, yb)[0, 1]

    boot_rs = boot_rs[~np.isnan(boot_rs)]
    if len(boot_rs) < 100:
        return np.nan, np.nan, np.nan

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_rs, alpha * 100)
    hi = np.percentile(boot_rs, (1 - alpha) * 100)
    r_obs = np.corrcoef(xv, yv)[0, 1]
    return r_obs, lo, hi


def permutation_test(x, y, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """Two-sided permutation test for Pearson r."""
    rng = np.random.RandomState(seed)
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan, np.nan

    r_obs = np.corrcoef(xv, yv)[0, 1]
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(yv)
        if np.std(yp) == 0:
            continue
        r_perm = np.corrcoef(xv, yp)[0, 1]
        if abs(r_perm) >= abs(r_obs):
            count += 1
    return r_obs, count / n_perm


# ---------------------------------------------------------------------------
# Build tidy DataFrame from JSON files (with missingness tracking)
# ---------------------------------------------------------------------------

def build_dataframe(directory):
    """
    Load all participant JSON files, compute change percentages
    (video - break), and build a tidy DataFrame with one row per
    participant-video.  Returns (df, missingness_records).
    """
    rows = []
    missingness_records = []

    for json_file in sorted(os.listdir(directory)):
        if not json_file.endswith(".json"):
            continue
        with open(os.path.join(directory, json_file)) as f:
            data = json.load(f)

        participant_id = data[0]["participant_id"]

        for entry in data:
            video_pcts, video_missing = calculate_percentages(entry)
            break_pcts, break_missing = calculate_percentages(entry.get("break_metrics", {}))

            # Change percentages (video minus break) for non-ECG biomarkers
            change = {}
            for biomarker in all_biomarkers:
                if not biomarker.startswith("ecg_"):
                    vp = video_pcts.get(biomarker)
                    bp = break_pcts.get(biomarker)
                    if vp is not None and bp is not None and not np.isnan(vp) and not np.isnan(bp):
                        change[biomarker] = vp - bp
                    else:
                        change[biomarker] = np.nan

            # ECG differences
            ecg_diffs = calculate_ecg_differences(video_pcts, break_pcts)

            row = {
                "participant": str(participant_id),
                "video": entry.get("video", ""),
                "difficulty": entry.get("difficulty", "unknown"),
                "delta": entry["delta"],
                **change,
                **ecg_diffs,
            }
            rows.append(row)

            # Track missingness reasons per video
            combined_missing = {}
            for modality in ["emotions", "blinks", "fixations", "pupil", "ecg"]:
                reasons = []
                if modality in video_missing:
                    reasons.append(f"video:{video_missing[modality]}")
                if modality in break_missing:
                    reasons.append(f"break:{break_missing[modality]}")
                if reasons:
                    combined_missing[modality] = "; ".join(reasons)

            missingness_records.append({
                "participant": str(participant_id),
                "video": entry.get("video", ""),
                **{f"missing_{k}_reason": v for k, v in combined_missing.items()},
            })

    df = pd.DataFrame(rows)
    miss_df = pd.DataFrame(missingness_records)
    return df, miss_df


# ---------------------------------------------------------------------------
# Missingness reporting (Concern 2)
# ---------------------------------------------------------------------------

def report_missingness(df, miss_df, output_dir):
    """Produce missingness CSV reports."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-feature missing rates
    metric_cols = [c for c in df.columns if c not in ["participant", "video", "difficulty", "delta"]]
    feat_rows = []
    for col in metric_cols:
        n_total = len(df)
        n_miss = int(df[col].isna().sum())
        feat_rows.append({
            "feature": col,
            "total_observations": n_total,
            "missing_count": n_miss,
            "missing_rate_pct": round(n_miss / n_total * 100, 2) if n_total > 0 else 0,
        })
    feat_df = pd.DataFrame(feat_rows).sort_values("missing_rate_pct", ascending=False)
    feat_df.to_csv(os.path.join(output_dir, "missingness_rates.csv"), index=False)

    # Per-participant per-modality
    reason_cols = [c for c in miss_df.columns if c.startswith("missing_") and c.endswith("_reason")]
    participant_miss = []
    for pid in sorted(df["participant"].unique()):
        pdf = df[df["participant"] == pid]
        pmdf = miss_df[miss_df["participant"] == pid]
        n_videos = len(pdf)
        for reason_col in reason_cols:
            modality = reason_col.replace("missing_", "").replace("_reason", "")
            if reason_col in pmdf.columns:
                n_miss = int(pmdf[reason_col].notna().sum())
                reasons = pmdf[reason_col].dropna().value_counts().to_dict()
            else:
                n_miss = 0
                reasons = {}
            participant_miss.append({
                "participant": pid,
                "modality": modality,
                "total_videos": n_videos,
                "missing_videos": n_miss,
                "missing_rate_pct": round(n_miss / n_videos * 100, 2) if n_videos > 0 else 0,
                "reasons": str(reasons) if reasons else "none",
            })
    pm_df = pd.DataFrame(participant_miss)
    pm_df.to_csv(os.path.join(output_dir, "missingness_per_participant.csv"), index=False)

    # Aggregate reason counts
    reason_rows = []
    for reason_col in reason_cols:
        if reason_col not in miss_df.columns:
            continue
        modality = reason_col.replace("missing_", "").replace("_reason", "")
        for reason, count in miss_df[reason_col].value_counts(dropna=True).items():
            reason_rows.append({"modality": modality, "reason": reason, "count": int(count)})
    reason_df = pd.DataFrame(reason_rows)
    reason_df.to_csv(os.path.join(output_dir, "missingness_reasons.csv"), index=False)

    print("\n=== MISSINGNESS REPORT ===")
    print(feat_df.to_string(index=False))
    if not pm_df.empty:
        print("\nPer-participant breakdown:")
        print(pm_df.to_string(index=False))

    return feat_df, pm_df


def impute_within_participant_median(df, feature_cols):
    """Impute missing values using the participant's own median for each feature."""
    df_imputed = df.copy()
    for feat in feature_cols:
        if feat not in df_imputed.columns:
            continue
        df_imputed[feat] = df_imputed.groupby("participant")[feat].transform(
            lambda s: s.fillna(s.median())
        )
    return df_imputed


# ---------------------------------------------------------------------------
# Per-participant correlations with CIs + FDR (Concern 1a)
# ---------------------------------------------------------------------------

def per_participant_correlations(df, metric_cols, output_dir):
    """
    For each participant, compute Pearson r, 95% bootstrap CI, raw p-value,
    and FDR-corrected p-value for every biomarker vs delta.
    """
    all_rows = []
    participants = sorted(df["participant"].unique())

    for pid in participants:
        pdf = df[df["participant"] == pid]
        delta = pdf["delta"].values.astype(float)
        raw_pvals = []
        row_indices = []

        for feat in metric_cols:
            if feat not in pdf.columns:
                continue
            x = pdf[feat].values.astype(float)
            n_obs = int(np.sum(~(np.isnan(x) | np.isnan(delta))))

            r, p = safe_pearsonr(x, delta)
            _, ci_lo, ci_hi = bootstrap_ci(x, delta)

            row = {
                "participant": pid,
                "feature": feat,
                "n_obs": n_obs,
                "pearson_r": round(r, 4) if not np.isnan(r) else None,
                "ci_95_low": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
                "ci_95_high": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
                "p_value_raw": round(p, 6) if not np.isnan(p) else None,
            }
            all_rows.append(row)

            if not np.isnan(p):
                raw_pvals.append(p)
                row_indices.append(len(all_rows) - 1)

        # FDR correction within this participant
        if raw_pvals:
            _, adj, _, _ = multipletests(raw_pvals, method="fdr_bh")
            for idx, adj_p in zip(row_indices, adj):
                all_rows[idx]["p_value_fdr"] = round(float(adj_p), 6)

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(os.path.join(output_dir, "correlations_with_ci.csv"), index=False)
    print("\n=== PER-PARTICIPANT CORRELATIONS WITH 95% BOOTSTRAP CIs ===")
    if not result_df.empty:
        sig = result_df[result_df.get("p_value_fdr", pd.Series(dtype=float)).notna()
                        & (result_df.get("p_value_fdr", pd.Series(dtype=float)) < 0.05)]
        print(f"Total tests: {len(result_df)}, Significant (FDR q<0.05): {len(sig)}")
        if not sig.empty:
            print(sig.to_string(index=False))
    return result_df


# ---------------------------------------------------------------------------
# Mixed-effects models (Concern 1b)
# ---------------------------------------------------------------------------

def run_mixed_effects(df, metric_cols, label, output_dir):
    """
    Fit delta ~ feature + C(difficulty) with participant random intercept.
    """
    results = []
    for feat in metric_cols:
        if feat not in df.columns:
            continue
        mdf = df[["participant", "difficulty", "delta", feat]].dropna()
        if len(mdf) < 6 or mdf["participant"].nunique() < 2:
            results.append({"feature": feat, "n_obs": len(mdf),
                            "n_participants": int(mdf["participant"].nunique()),
                            "coef": None, "ci_low": None, "ci_high": None,
                            "p_value": None})
            continue

        mdf = mdf.rename(columns={feat: "feature_value"})
        try:
            model = smf.mixedlm(
                "delta ~ feature_value + C(difficulty)",
                mdf,
                groups=mdf["participant"],
            )
            fit = model.fit(reml=False)
            coef = fit.params.get("feature_value")
            pval = fit.pvalues.get("feature_value")
            ci = fit.conf_int().loc["feature_value"].tolist()
            results.append({
                "feature": feat,
                "n_obs": len(mdf),
                "n_participants": int(mdf["participant"].nunique()),
                "coef": round(float(coef), 6) if coef is not None else None,
                "ci_low": round(float(ci[0]), 6) if ci else None,
                "ci_high": round(float(ci[1]), 6) if ci else None,
                "p_value": round(float(pval), 6) if pval is not None else None,
            })
        except Exception as exc:
            print(f"  Mixed-effects failed for {feat}: {exc}")
            results.append({"feature": feat, "n_obs": len(mdf),
                            "n_participants": int(mdf["participant"].nunique()),
                            "coef": None, "ci_low": None, "ci_high": None,
                            "p_value": None})

    res_df = pd.DataFrame(results)
    valid_mask = res_df["p_value"].notna()
    if valid_mask.any():
        pvals = res_df.loc[valid_mask, "p_value"].values
        _, adj, _, _ = multipletests(pvals, method="fdr_bh")
        res_df.loc[valid_mask, "p_value_fdr"] = [round(float(a), 6) for a in adj]
    else:
        res_df["p_value_fdr"] = None

    fname = f"mixed_effects_{label}.csv"
    res_df.to_csv(os.path.join(output_dir, fname), index=False)
    print(f"\n=== MIXED-EFFECTS MODEL ({label}) ===")
    if not res_df.empty:
        sig = res_df[res_df["p_value_fdr"].notna() & (res_df["p_value_fdr"] < 0.05)]
        print(f"Total biomarkers: {len(res_df)}, Significant (FDR q<0.05): {len(sig)}")
        if not sig.empty:
            print(sig.to_string(index=False))
    return res_df


# ---------------------------------------------------------------------------
# Permutation tests (Concern 1c)
# ---------------------------------------------------------------------------

def run_permutation_tests(df, metric_cols, output_dir):
    """Run permutation tests per participant for each biomarker vs delta."""
    rows = []
    participants = sorted(df["participant"].unique())

    for pid in participants:
        pdf = df[df["participant"] == pid]
        delta = pdf["delta"].values.astype(float)
        raw_pvals = []
        row_indices = []

        for feat in metric_cols:
            if feat not in pdf.columns:
                continue
            x = pdf[feat].values.astype(float)
            r_obs, p_perm = permutation_test(x, delta)
            row = {
                "participant": pid,
                "feature": feat,
                "n_obs": int(np.sum(~(np.isnan(x) | np.isnan(delta)))),
                "pearson_r": round(r_obs, 4) if not np.isnan(r_obs) else None,
                "permutation_p": round(p_perm, 6) if not np.isnan(p_perm) else None,
            }
            rows.append(row)
            if not np.isnan(p_perm):
                raw_pvals.append(p_perm)
                row_indices.append(len(rows) - 1)

        # FDR correction
        if raw_pvals:
            _, adj, _, _ = multipletests(raw_pvals, method="fdr_bh")
            for idx, adj_p in zip(row_indices, adj):
                rows[idx]["permutation_p_fdr"] = round(float(adj_p), 6)

    perm_df = pd.DataFrame(rows)
    perm_df.to_csv(os.path.join(output_dir, "permutation_tests.csv"), index=False)
    print("\n=== PERMUTATION TESTS ===")
    if not perm_df.empty:
        sig = perm_df[perm_df.get("permutation_p_fdr", pd.Series(dtype=float)).notna()
                      & (perm_df.get("permutation_p_fdr", pd.Series(dtype=float)) < 0.05)]
        print(f"Total tests: {len(perm_df)}, Significant (FDR q<0.05): {len(sig)}")
    return perm_df


# ---------------------------------------------------------------------------
# Sensitivity analysis (Concern 2 continued)
# ---------------------------------------------------------------------------

def sensitivity_analysis(df_raw, df_imputed, metric_cols, output_dir):
    """
    Compare mixed-effects results with and without imputation.
    """
    me_raw = run_mixed_effects(df_raw, metric_cols, "no_imputation", output_dir)
    me_imp = run_mixed_effects(df_imputed, metric_cols, "with_imputation", output_dir)

    merged = me_raw.merge(me_imp, on="feature", suffixes=("_raw", "_imputed"))
    merged["direction_changed"] = (
        np.sign(merged["coef_raw"].fillna(0)) != np.sign(merged["coef_imputed"].fillna(0))
    )
    merged["coef_diff"] = (merged["coef_imputed"].fillna(0) - merged["coef_raw"].fillna(0)).round(6)

    merged.to_csv(os.path.join(output_dir, "sensitivity_analysis.csv"), index=False)
    print("\n=== SENSITIVITY ANALYSIS ===")
    n_changed = int(merged["direction_changed"].sum())
    print(f"Direction changed for {n_changed}/{len(merged)} biomarkers after imputation.")
    return merged


# ---------------------------------------------------------------------------
# Plotting (updated with CIs and significance markers)
# ---------------------------------------------------------------------------

def plot_participant_correlations(df, participant_id, metric_cols, corr_df, output_dir):
    """Create bar plot for a single participant with significance markers."""
    pid_corr = corr_df[corr_df["participant"] == str(participant_id)].copy()
    if pid_corr.empty:
        return

    # Sort by absolute correlation
    pid_corr["abs_r"] = pid_corr["pearson_r"].abs()
    pid_corr = pid_corr.sort_values("abs_r", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(pid_corr) * 0.4)))

    colors = plt.cm.RdBu((pid_corr["pearson_r"].fillna(0).values + 1) / 2)
    bars = ax.barh(range(len(pid_corr)), pid_corr["pearson_r"].fillna(0).values, color=colors)

    # Add error bars from bootstrap CIs
    ci_lo = pid_corr["ci_95_low"].values
    ci_hi = pid_corr["ci_95_high"].values
    r_vals = pid_corr["pearson_r"].fillna(0).values
    for i in range(len(pid_corr)):
        if not np.isnan(ci_lo[i]) and not np.isnan(ci_hi[i]):
            ax.plot([ci_lo[i], ci_hi[i]], [i, i], color='black', linewidth=1.5, alpha=0.6)

    # Mark significant results
    for i, (_, row) in enumerate(pid_corr.iterrows()):
        width = row["pearson_r"] if not pd.isna(row["pearson_r"]) else 0
        sig_marker = ""
        if pd.notna(row.get("p_value_fdr")) and row["p_value_fdr"] < 0.05:
            sig_marker = " **"
        elif pd.notna(row.get("p_value_raw")) and row["p_value_raw"] < 0.05:
            sig_marker = " *"
        ax.text(width + np.sign(width) * 0.02 if width != 0 else 0.02,
                i, f'{width:.2f}{sig_marker}',
                ha='left' if width >= 0 else 'right',
                va='center', fontsize=9, weight='bold')

    ax.set_yticks(range(len(pid_corr)))
    ax.set_yticklabels([get_display_name(f) for f in pid_corr["feature"].values], fontsize=10)
    ax.set_xlabel("Pearson r with Learning Gain (95% Bootstrap CI)", fontsize=12)
    ax.set_title(f"Participant {participant_id}: Biomarker-Learning Correlations\n"
                 f"(* p<0.05 raw, ** p<0.05 FDR-corrected)", fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"participant_{participant_id}_correlation.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_aggregate_heatmaps(df, metric_cols, corr_df, output_dir):
    """Create aggregate heatmaps grouped by metric type."""
    # Build a pivot of correlations
    participants = sorted(df["participant"].unique())
    pivot_data = {}
    for _, row in corr_df.iterrows():
        feat = row["feature"]
        pid = row["participant"]
        pivot_data.setdefault(feat, {})[pid] = row["pearson_r"]

    pivot_df = pd.DataFrame(pivot_data).T
    pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns))

    # Group metrics
    metric_groups = {
        'Emotion': [b for b in metric_cols if b in pivot_df.index and
                    any(e in b.lower() for e in ['happy', 'sad', 'fear', 'disgust', 'angry',
                                                  'surprise', 'neutral', 'positive', 'negative', 'non_neutral'])
                    and not b.startswith('diff_ecg_')],
        'Eye-Tracking': [b for b in metric_cols if b in pivot_df.index and
                         any(e in b for e in ['blink', 'fixation', 'pupil'])],
        'ECG': [b for b in metric_cols if b in pivot_df.index and b.startswith('diff_ecg_')],
    }

    for group_name, metrics in metric_groups.items():
        if not metrics:
            continue

        sub_df = pivot_df.loc[metrics]
        if sub_df.empty:
            continue

        plt.figure(figsize=(max(10, len(sub_df.columns) * 0.8), max(8, len(sub_df) * 0.6)))
        sns.heatmap(
            sub_df.astype(float),
            cmap="RdBu",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 8},
            cbar_kws={"label": "Pearson r (with 95% Bootstrap CIs reported in CSV)"},
            vmin=-1, vmax=1,
        )
        plt.xlabel("Participant ID", fontsize=12)
        plt.ylabel("Biomarker", fontsize=12)

        y_labels = [get_display_name(m) for m in metrics]
        plt.yticks(np.arange(len(y_labels)) + 0.5, y_labels, rotation=0, fontsize=10)
        plt.title(f"Aggregate {group_name} Correlations with Learning Gain\n"
                  f"(See correlations_with_ci.csv for CIs and p-values)", fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"aggregate_{group_name.lower()}_correlation_heatmap.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Paper-ready summary
# ---------------------------------------------------------------------------

def write_paper_summary(feat_miss, corr_df, me_raw, perm_df, sensitivity_df, output_dir):
    """Write a plain-text summary for copy-pasting into the paper."""
    path = os.path.join(output_dir, "paper_summary.txt")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("REVIEWER-REQUESTED STATISTICAL SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # --- Missingness ---
        f.write("SECTION: MISSING DATA (Reviewer Concern 2)\n")
        f.write("-" * 50 + "\n\n")
        f.write("Missingness rates per biomarker feature:\n")
        for _, row in feat_miss.iterrows():
            f.write(f"  {row['feature']}: {row['missing_count']}/{row['total_observations']} "
                    f"({row['missing_rate_pct']:.1f}%)\n")

        f.write("\nImputation: Within-participant median imputation was applied.\n")
        f.write("Sensitivity: Mixed-effects models were run with and without imputation.\n")
        if sensitivity_df is not None:
            n_changed = int(sensitivity_df["direction_changed"].sum())
            f.write(f"  Direction of effect changed for {n_changed}/{len(sensitivity_df)} biomarkers.\n\n")

        # --- Correlations ---
        f.write("SECTION: STATISTICAL ANALYSIS (Reviewer Concern 1)\n")
        f.write("-" * 50 + "\n\n")

        f.write("Per-participant Pearson correlations with 95% bootstrap CIs:\n")
        if corr_df is not None and not corr_df.empty:
            if "p_value_fdr" in corr_df.columns:
                sig = corr_df[corr_df["p_value_fdr"].notna() & (corr_df["p_value_fdr"] < 0.05)]
            else:
                sig = pd.DataFrame()
            f.write(f"  Total tests: {len(corr_df)}\n")
            f.write(f"  Significant after FDR (q<0.05): {len(sig)}\n")
            if not sig.empty:
                for _, row in sig.iterrows():
                    f.write(f"    P{row['participant']}, {row['feature']}: "
                            f"r={row['pearson_r']}, CI [{row['ci_95_low']}, {row['ci_95_high']}], "
                            f"p_raw={row['p_value_raw']}, p_FDR={row['p_value_fdr']}\n")

        f.write("\nMixed-effects models (delta ~ biomarker + difficulty, random intercept for participant):\n")
        if me_raw is not None and not me_raw.empty:
            if "p_value_fdr" in me_raw.columns:
                sig_me = me_raw[me_raw["p_value_fdr"].notna() & (me_raw["p_value_fdr"] < 0.05)]
            else:
                sig_me = pd.DataFrame()
            f.write(f"  Total biomarkers: {len(me_raw)}\n")
            f.write(f"  Significant after FDR (q<0.05): {len(sig_me)}\n")
            if not sig_me.empty:
                for _, row in sig_me.iterrows():
                    f.write(f"    {row['feature']}: coef={row['coef']}, "
                            f"CI [{row['ci_low']}, {row['ci_high']}], "
                            f"p={row['p_value']}, p_FDR={row['p_value_fdr']}\n")

        f.write(f"\nPermutation tests ({N_PERMUTATIONS} permutations per participant per biomarker):\n")
        if perm_df is not None and not perm_df.empty:
            if "permutation_p_fdr" in perm_df.columns:
                sig_perm = perm_df[perm_df["permutation_p_fdr"].notna() & (perm_df["permutation_p_fdr"] < 0.05)]
            else:
                sig_perm = pd.DataFrame()
            f.write(f"  Total tests: {len(perm_df)}\n")
            f.write(f"  Significant after FDR (q<0.05): {len(sig_perm)}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 70 + "\n")

    print(f"\nPaper summary written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading participant data from:", DIRECTORY)
    df, miss_df = build_dataframe(DIRECTORY)
    if df.empty:
        print("ERROR: No data loaded.")
        return

    metric_cols = [c for c in df.columns if c not in ["participant", "video", "difficulty", "delta"]]

    # Convert to numeric
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    participants = sorted(df["participant"].unique())
    print(f"Loaded {len(df)} video-level records for {len(participants)} participants.")
    print(f"Records per participant: {df.groupby('participant').size().to_dict()}")

    # ---- Concern 2: Missingness ----
    print("\n" + "=" * 60)
    print("CONCERN 2: MISSINGNESS REPORTING")
    print("=" * 60)
    feat_miss, pm_miss = report_missingness(df, miss_df, OUTPUT_DIR)

    # Imputation
    df_imputed = impute_within_participant_median(df, metric_cols)
    impute_summary = []
    for feat in metric_cols:
        n_before = int(df[feat].isna().sum())
        n_after = int(df_imputed[feat].isna().sum())
        impute_summary.append({"feature": feat, "missing_before": n_before,
                                "missing_after": n_after, "imputed": n_before - n_after})
    imp_df = pd.DataFrame(impute_summary)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "imputation_summary.csv"), index=False)
    print("\nImputation summary:")
    print(imp_df.to_string(index=False))

    # ---- Concern 1: Statistical rigour ----
    print("\n" + "=" * 60)
    print("CONCERN 1: STATISTICAL RIGOUR")
    print("=" * 60)

    # 1a. Per-participant correlations with bootstrap CIs
    print("\n--- Per-participant correlations with bootstrap CIs ---")
    corr_df = per_participant_correlations(df, metric_cols, OUTPUT_DIR)

    # 1b. Mixed-effects (raw data, no imputation)
    print("\n--- Mixed-effects models (no imputation) ---")
    me_raw = run_mixed_effects(df, metric_cols, "no_imputation", OUTPUT_DIR)

    # 1c. Permutation tests
    print("\n--- Permutation tests ---")
    perm_df = run_permutation_tests(df, metric_cols, OUTPUT_DIR)

    # ---- Sensitivity analysis ----
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)
    sensitivity_df = sensitivity_analysis(df, df_imputed, metric_cols, OUTPUT_DIR)

    # ---- Plots ----
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Per-participant correlation bar plots with CIs
    for pid in participants:
        plot_participant_correlations(df, pid, metric_cols, corr_df, OUTPUT_DIR)

    # Aggregate heatmaps
    plot_aggregate_heatmaps(df, metric_cols, corr_df, OUTPUT_DIR)

    # Per-participant CSV files (backwards-compatible)
    for pid in participants:
        pid_corr = corr_df[corr_df["participant"] == pid]
        if not pid_corr.empty:
            pid_corr.to_csv(os.path.join(OUTPUT_DIR, f"participant_{pid}_correlations.csv"), index=False)

    # ---- Paper summary ----
    write_paper_summary(feat_miss, corr_df, me_raw, perm_df, sensitivity_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print("=" * 60)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {fname} ({size:,} bytes)")


if __name__ == "__main__":
    main()
