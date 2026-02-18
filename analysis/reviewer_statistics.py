"""
reviewer_statistics.py
======================
Produces all statistical outputs requested by the reviewer:

Reviewer Concern 1 -- Statistical rigour
  * Pearson correlations with 95% bootstrap confidence intervals
  * P-values corrected for multiple comparisons (FDR Benjamini-Hochberg)
  * Linear mixed-effects models (participant as random intercept)
  * Permutation tests for robustness

Reviewer Concern 2 -- Missing data handling
  * Missingness rates per participant per modality
  * Missingness reasons (tracker loss, blink occlusion, no_samples, etc.)
  * Within-participant median imputation
  * Sensitivity analysis (imputed vs. non-imputed results)

Outputs are saved to  reviewer_outputs/  (created automatically).
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PARTICIPANT_INFO_DIR = os.path.join(PROJECT_ROOT, "participant_info")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reviewer_outputs")

EMOTION_FEATURES = [
    "emotion_diversity",
    "non_neutral_percent",
    "positive_percent",
    "negative_percent",
    "neutral_percent",
    "happy_percent",
    "sad_percent",
    "surprise_percent",
    "fear_percent",
    "disgust_percent",
    "angry_percent",
]

EYE_FEATURES = [
    "short_blink_percent",
    "medium_blink_percent",
    "long_blink_percent",
    "short_fixation_low_disp_percent",
    "short_fixation_high_disp_percent",
    "medium_fixation_low_disp_percent",
    "medium_fixation_high_disp_percent",
    "long_fixation_low_disp_percent",
    "long_fixation_high_disp_percent",
    "small_pupil_percent",
    "medium_pupil_percent",
    "large_pupil_percent",
]

ECG_FEATURES = [
    "ecg_heart_rate",
    "ecg_sdnn",
    "ecg_rmssd",
    "ecg_pnn50",
]

ALL_FEATURES = EMOTION_FEATURES + EYE_FEATURES + ECG_FEATURES

N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_participant_data(folder_path=None):
    """Load all participant JSON files from the specified folder."""
    if folder_path is None:
        folder_path = PARTICIPANT_INFO_DIR
    participants_data = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json") and not filename.startswith("."):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                participants_data.extend(json.load(f))
    return participants_data


def calculate_emotion_metrics(emotions_dict):
    """Calculate percentages for emotion metrics."""
    if not emotions_dict:
        return {k: None for k in EMOTION_FEATURES}

    total = sum(emotions_dict.values())
    if total == 0:
        return {k: 0.0 for k in EMOTION_FEATURES}

    neutral = emotions_dict.get("Neutral", 0)
    happy = emotions_dict.get("Happy", 0)
    sad = emotions_dict.get("Sad", 0)
    surprise = emotions_dict.get("Surprise", 0)
    fear = emotions_dict.get("Fear", 0)
    disgust = emotions_dict.get("Disgust", 0)
    angry = emotions_dict.get("Angry", 0)

    return {
        "emotion_diversity": len(emotions_dict),
        "non_neutral_percent": (total - neutral) / total * 100,
        "positive_percent": (happy + surprise) / total * 100,
        "negative_percent": (sad + fear + disgust + angry) / total * 100,
        "neutral_percent": neutral / total * 100,
        "happy_percent": happy / total * 100,
        "sad_percent": sad / total * 100,
        "surprise_percent": surprise / total * 100,
        "fear_percent": fear / total * 100,
        "disgust_percent": disgust / total * 100,
        "angry_percent": angry / total * 100,
    }


def calculate_eye_metrics(video_data):
    """Calculate percentages for eye-tracking metrics."""
    blinks = video_data.get("blinks", {})
    fixations = video_data.get("fixations", {})
    pupil_diameter = video_data.get("pupil_diameter", {})

    if not isinstance(blinks, dict):
        blinks = {}
    if not isinstance(fixations, dict):
        fixations = {}
    if not isinstance(pupil_diameter, dict):
        pupil_diameter = {}

    # Blinks
    sb, mb, lb = blinks.get("short", 0), blinks.get("medium", 0), blinks.get("long", 0)
    tb = sb + mb + lb
    if tb == 0:
        blink_pcts = {"short_blink_percent": None, "medium_blink_percent": None, "long_blink_percent": None}
    else:
        blink_pcts = {
            "short_blink_percent": sb / tb * 100,
            "medium_blink_percent": mb / tb * 100,
            "long_blink_percent": lb / tb * 100,
        }

    # Fixations
    fix_pcts = {}
    for dur in ["short", "medium", "long"]:
        low = fixations.get(dur, {}).get("low_dispersion", 0)
        high = fixations.get(dur, {}).get("high_dispersion", 0)
        total_dur = low + high
        if total_dur == 0:
            fix_pcts[f"{dur}_fixation_low_disp_percent"] = None
            fix_pcts[f"{dur}_fixation_high_disp_percent"] = None
        else:
            fix_pcts[f"{dur}_fixation_low_disp_percent"] = low / total_dur * 100
            fix_pcts[f"{dur}_fixation_high_disp_percent"] = high / total_dur * 100

    # Pupil
    sp, mp, lp = pupil_diameter.get("small", 0), pupil_diameter.get("medium", 0), pupil_diameter.get("large", 0)
    tp = sp + mp + lp
    if tp == 0:
        pupil_pcts = {"small_pupil_percent": None, "medium_pupil_percent": None, "large_pupil_percent": None}
    else:
        pupil_pcts = {
            "small_pupil_percent": sp / tp * 100,
            "medium_pupil_percent": mp / tp * 100,
            "large_pupil_percent": lp / tp * 100,
        }

    return {**blink_pcts, **fix_pcts, **pupil_pcts}


def calculate_ecg_metrics(video_data):
    """Extract ECG metrics from video data (top-level or under ecg_metrics)."""
    ecg = video_data.get("ecg_metrics", {})
    if not isinstance(ecg, dict) or not ecg:
        return {k: None for k in ECG_FEATURES}
    return {
        "ecg_heart_rate": ecg.get("heart_rate"),
        "ecg_sdnn": ecg.get("sdnn"),
        "ecg_rmssd": ecg.get("rmssd"),
        "ecg_pnn50": ecg.get("pnn50"),
    }


def extract_features(participants_data):
    """Build a tidy DataFrame with one row per participant-video."""
    rows = []
    for vd in participants_data:
        pid = vd.get("participant_id")
        delta = vd.get("delta")
        if pid is None or delta is None:
            continue

        emotions = vd.get("emotions", {})
        blinks = vd.get("blinks", {})
        fixations = vd.get("fixations", {})
        pupil_diameter = vd.get("pupil_diameter", {})

        quality = vd.get("quality", {})
        missing_reason = quality.get("missing_reason", {}) if isinstance(quality, dict) else {}

        em = calculate_emotion_metrics(emotions)
        ey = calculate_eye_metrics(vd)
        ecg = calculate_ecg_metrics(vd)

        # --- Detect effective missingness from the raw data ---
        # When quality field is absent, infer from the raw counts.
        # This catches cases like tracker loss or blink occlusion.

        # Emotions: empty or absent
        if not emotions:
            for k in EMOTION_FEATURES:
                em[k] = None
            missing_reason.setdefault("emotions", "no_samples")

        # Blinks: no data or all-zero counts
        if not isinstance(blinks, dict) or not blinks:
            for k in ["short_blink_percent", "medium_blink_percent", "long_blink_percent"]:
                ey[k] = None
            missing_reason.setdefault("blinks", "no_samples")
        elif sum(blinks.values()) == 0:
            # Zero blinks detected -- possible tracker loss
            for k in ["short_blink_percent", "medium_blink_percent", "long_blink_percent"]:
                ey[k] = None
            missing_reason.setdefault("blinks", "zero_blinks_detected_possible_tracker_loss")

        # Fixations: empty or absent
        if not isinstance(fixations, dict) or not fixations:
            for k in EYE_FEATURES:
                if "fixation" in k:
                    ey[k] = None
            missing_reason.setdefault("fixations", "no_samples")

        # Pupil: empty or absent or all zero
        if not isinstance(pupil_diameter, dict) or not pupil_diameter:
            for k in ["small_pupil_percent", "medium_pupil_percent", "large_pupil_percent"]:
                ey[k] = None
            missing_reason.setdefault("pupil", "no_samples")
        elif sum(pupil_diameter.values()) == 0:
            for k in ["small_pupil_percent", "medium_pupil_percent", "large_pupil_percent"]:
                ey[k] = None
            missing_reason.setdefault("pupil", "zero_readings_possible_tracker_loss")

        # ECG: missing if all None
        if all(v is None for v in ecg.values()):
            missing_reason.setdefault("ecg", "no_ecg_data")

        # Apply explicit quality flags (overrides data-inferred flags above)
        if missing_reason.get("emotions"):
            for k in EMOTION_FEATURES:
                em[k] = None
        if missing_reason.get("blinks"):
            for k in ["short_blink_percent", "medium_blink_percent", "long_blink_percent"]:
                ey[k] = None
        if missing_reason.get("fixations"):
            for k in EYE_FEATURES:
                if "fixation" in k:
                    ey[k] = None
        if missing_reason.get("pupil"):
            for k in ["small_pupil_percent", "medium_pupil_percent", "large_pupil_percent"]:
                ey[k] = None

        # Raw sample counts for missingness reporting
        sample_counts = {}
        if isinstance(quality, dict) and quality:
            sample_counts = {
                "emotion_samples": quality.get("emotion_samples", 0),
                "blink_samples": quality.get("blink_samples", 0),
                "fixation_samples": quality.get("fixation_samples", 0),
                "pupil_samples": quality.get("pupil_samples", 0),
            }
        else:
            # Infer sample counts from the raw data when quality field is absent
            sample_counts = {
                "emotion_samples": sum(emotions.values()) if emotions else 0,
                "blink_samples": sum(blinks.values()) if isinstance(blinks, dict) else 0,
                "fixation_samples": (
                    sum(
                        sum(d.values()) for d in fixations.values()
                        if isinstance(d, dict)
                    )
                    if isinstance(fixations, dict)
                    else 0
                ),
                "pupil_samples": sum(pupil_diameter.values()) if isinstance(pupil_diameter, dict) else 0,
            }

        rows.append(
            {
                "participant": pid,
                "video": vd.get("video", "").split("/")[-1],
                "difficulty": vd.get("difficulty"),
                "delta": delta,
                "missing_emotions_reason": missing_reason.get("emotions"),
                "missing_blinks_reason": missing_reason.get("blinks"),
                "missing_fixations_reason": missing_reason.get("fixations"),
                "missing_pupil_reason": missing_reason.get("pupil"),
                "missing_ecg_reason": missing_reason.get("ecg"),
                **sample_counts,
                **em,
                **ey,
                **ecg,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Concern 2: Missingness
# ---------------------------------------------------------------------------

def report_missingness(df, output_dir):
    """
    Produce two CSV files:
      1. missingness_rates.csv  -- per-feature missing rate and count
      2. missingness_reasons.csv -- per-modality reason breakdown

    Also prints a summary to stdout.
    """
    # --- Per-feature missing rates ---
    feat_rows = []
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            continue
        n_total = len(df)
        n_missing = int(df[feat].isna().sum())
        feat_rows.append(
            {
                "feature": feat,
                "total_observations": n_total,
                "missing_count": n_missing,
                "missing_rate_pct": round(n_missing / n_total * 100, 2) if n_total > 0 else 0,
            }
        )
    feat_df = pd.DataFrame(feat_rows).sort_values("missing_rate_pct", ascending=False)
    feat_df.to_csv(os.path.join(output_dir, "missingness_rates.csv"), index=False)

    # --- Per-participant per-modality ---
    participant_miss = []
    for pid in sorted(df["participant"].unique()):
        pdf = df[df["participant"] == pid]
        n_videos = len(pdf)
        for modality, reason_col, feat_cols in [
            ("emotions", "missing_emotions_reason", EMOTION_FEATURES),
            ("blinks", "missing_blinks_reason", ["short_blink_percent", "medium_blink_percent", "long_blink_percent"]),
            ("fixations", "missing_fixations_reason", [f for f in EYE_FEATURES if "fixation" in f]),
            ("pupil", "missing_pupil_reason", ["small_pupil_percent", "medium_pupil_percent", "large_pupil_percent"]),
            ("ecg", "missing_ecg_reason", ECG_FEATURES),
        ]:
            n_miss = int(pdf[feat_cols[0]].isna().sum()) if feat_cols[0] in pdf.columns else 0
            reasons = pdf[reason_col].dropna().value_counts().to_dict() if reason_col in pdf.columns else {}
            participant_miss.append(
                {
                    "participant": pid,
                    "modality": modality,
                    "total_videos": n_videos,
                    "missing_videos": n_miss,
                    "missing_rate_pct": round(n_miss / n_videos * 100, 2) if n_videos > 0 else 0,
                    "reasons": str(reasons) if reasons else "none",
                }
            )
    pm_df = pd.DataFrame(participant_miss)
    pm_df.to_csv(os.path.join(output_dir, "missingness_per_participant.csv"), index=False)

    # --- Aggregate reason counts ---
    reason_rows = []
    for reason_col in ["missing_emotions_reason", "missing_blinks_reason", "missing_fixations_reason", "missing_pupil_reason", "missing_ecg_reason"]:
        if reason_col not in df.columns:
            continue
        modality = reason_col.replace("missing_", "").replace("_reason", "")
        for reason, count in df[reason_col].value_counts(dropna=True).items():
            reason_rows.append({"modality": modality, "reason": reason, "count": int(count)})
    reason_df = pd.DataFrame(reason_rows)
    reason_df.to_csv(os.path.join(output_dir, "missingness_reasons.csv"), index=False)

    # Print summary
    print("\n=== MISSINGNESS REPORT ===")
    print(feat_df.to_string(index=False))
    print("\nPer-participant breakdown:")
    print(pm_df.to_string(index=False))
    if not reason_df.empty:
        print("\nMissingness reasons:")
        print(reason_df.to_string(index=False))

    return feat_df, pm_df, reason_df


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
# Concern 1: Statistical rigour -- helpers
# ---------------------------------------------------------------------------

def safe_pearsonr(x, y):
    """Return (r, p) or (NaN, NaN) on failure."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    if len(xv) < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan, np.nan
    return pearsonr(xv, yv)


def bootstrap_ci(x, y, n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    """Bootstrap 95 % CI for Pearson r."""
    rng = np.random.RandomState(seed)
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
    p_perm = count / n_perm
    return r_obs, p_perm


# ---------------------------------------------------------------------------
# Concern 1a: Per-participant correlations with CIs + corrected p-values
# ---------------------------------------------------------------------------

def per_participant_correlations(df, output_dir):
    """
    For each participant, compute Pearson r, 95 % bootstrap CI, raw p-value,
    and FDR-corrected p-value for every biomarker vs delta.
    """
    all_rows = []
    participants = sorted(df["participant"].unique())

    for pid in participants:
        pdf = df[df["participant"] == pid]
        delta = pdf["delta"].values.astype(float)
        raw_pvals = []
        row_indices = []

        for feat in ALL_FEATURES:
            if feat not in pdf.columns:
                continue
            x = pdf[feat].values.astype(float)

            r, p = safe_pearsonr(x, delta)
            _, ci_lo, ci_hi = bootstrap_ci(x, delta)

            row = {
                "participant": pid,
                "feature": feat,
                "n_obs": int(np.sum(~(np.isnan(x) | np.isnan(delta)))),
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
    print(result_df.to_string(index=False))
    return result_df


# ---------------------------------------------------------------------------
# Concern 1b: Mixed-effects models
# ---------------------------------------------------------------------------

def run_mixed_effects(df, label, output_dir):
    """
    Fit delta ~ feature + C(difficulty) with participant random intercept.
    Returns DataFrame with coefficient, CI, p-value, and FDR-adjusted p-value.
    """
    results = []
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            continue
        mdf = df[["participant", "difficulty", "delta", feat]].dropna()
        if len(mdf) < 6 or mdf["participant"].nunique() < 2:
            results.append({"feature": feat, "n_obs": len(mdf),
                            "n_participants": mdf["participant"].nunique(),
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

    # FDR correction
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
    print(res_df.to_string(index=False))
    return res_df


# ---------------------------------------------------------------------------
# Concern 1c: Permutation tests
# ---------------------------------------------------------------------------

def run_permutation_tests(df, output_dir):
    """Run permutation tests per participant for each biomarker vs delta."""
    rows = []
    participants = sorted(df["participant"].unique())

    for pid in participants:
        pdf = df[df["participant"] == pid]
        delta = pdf["delta"].values.astype(float)
        raw_pvals = []
        row_indices = []

        for feat in ALL_FEATURES:
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
    print(perm_df.to_string(index=False))
    return perm_df


# ---------------------------------------------------------------------------
# Concern 2 continued: Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(df_raw, df_imputed, output_dir):
    """
    Compare mixed-effects results with and without imputation.
    Flag biomarkers where the direction of the effect changed.
    """
    me_raw = run_mixed_effects(df_raw, "no_imputation", output_dir)
    me_imp = run_mixed_effects(df_imputed, "with_imputation", output_dir)

    merged = me_raw.merge(me_imp, on="feature", suffixes=("_raw", "_imputed"))
    merged["direction_changed"] = (
        np.sign(merged["coef_raw"].fillna(0)) != np.sign(merged["coef_imputed"].fillna(0))
    )
    merged["coef_diff"] = (merged["coef_imputed"].fillna(0) - merged["coef_raw"].fillna(0)).round(6)

    merged.to_csv(os.path.join(output_dir, "sensitivity_analysis.csv"), index=False)
    print("\n=== SENSITIVITY ANALYSIS (raw vs imputed) ===")
    print(merged[["feature", "coef_raw", "coef_imputed", "direction_changed", "coef_diff"]].to_string(index=False))
    return merged


# ---------------------------------------------------------------------------
# Paper-ready summary
# ---------------------------------------------------------------------------

def write_paper_summary(miss_feat, miss_part, corr_df, me_raw, perm_df, sensitivity_df, output_dir):
    """Write a plain-text summary with key numbers for copy-pasting into the paper."""
    path = os.path.join(output_dir, "paper_summary.txt")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("REVIEWER-REQUESTED STATISTICAL SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # --- Missingness ---
        f.write("SECTION: MISSING DATA (Reviewer Concern 2)\n")
        f.write("-" * 50 + "\n\n")
        f.write("Missingness rates per biomarker feature:\n")
        for _, row in miss_feat.iterrows():
            f.write(f"  {row['feature']}: {row['missing_count']}/{row['total_observations']} "
                    f"({row['missing_rate_pct']:.1f}%)\n")

        f.write("\nMissingness per participant per modality:\n")
        for _, row in miss_part.iterrows():
            f.write(f"  Participant {row['participant']}, {row['modality']}: "
                    f"{row['missing_videos']}/{row['total_videos']} videos missing "
                    f"({row['missing_rate_pct']:.1f}%) -- reasons: {row['reasons']}\n")

        f.write("\nImputation: Within-participant median imputation was applied.\n")
        f.write("Sensitivity: Mixed-effects models were run both with and without imputation.\n")
        n_changed = int(sensitivity_df["direction_changed"].sum()) if sensitivity_df is not None else 0
        n_total = len(sensitivity_df) if sensitivity_df is not None else 0
        f.write(f"  Direction of effect changed for {n_changed}/{n_total} biomarkers after imputation.\n\n")

        # --- Correlations ---
        f.write("SECTION: STATISTICAL ANALYSIS (Reviewer Concern 1)\n")
        f.write("-" * 50 + "\n\n")

        f.write("Per-participant Pearson correlations with 95% bootstrap CIs:\n")
        if corr_df is not None:
            sig_rows = corr_df[corr_df["p_value_fdr"].notna() & (corr_df["p_value_fdr"] < 0.05)]
            f.write(f"  Total correlation tests: {len(corr_df)}\n")
            f.write(f"  Significant after FDR correction (q < 0.05): {len(sig_rows)}\n")
            if not sig_rows.empty:
                f.write("  Significant correlations:\n")
                for _, row in sig_rows.iterrows():
                    f.write(f"    Participant {row['participant']}, {row['feature']}: "
                            f"r = {row['pearson_r']}, 95% CI [{row['ci_95_low']}, {row['ci_95_high']}], "
                            f"p_raw = {row['p_value_raw']}, p_FDR = {row['p_value_fdr']}\n")

        f.write("\nMixed-effects models (delta ~ biomarker + difficulty, random intercept for participant):\n")
        if me_raw is not None:
            sig_me = me_raw[me_raw["p_value_fdr"].notna() & (me_raw["p_value_fdr"] < 0.05)]
            f.write(f"  Total biomarkers tested: {len(me_raw)}\n")
            f.write(f"  Significant after FDR correction (q < 0.05): {len(sig_me)}\n")
            if not sig_me.empty:
                f.write("  Significant effects:\n")
                for _, row in sig_me.iterrows():
                    f.write(f"    {row['feature']}: coef = {row['coef']}, "
                            f"95% CI [{row['ci_low']}, {row['ci_high']}], "
                            f"p = {row['p_value']}, p_FDR = {row['p_value_fdr']}\n")

        f.write("\nPermutation tests (5000 permutations per participant per biomarker):\n")
        if perm_df is not None:
            sig_perm = perm_df[perm_df["permutation_p_fdr"].notna() & (perm_df["permutation_p_fdr"] < 0.05)]
            f.write(f"  Total tests: {len(perm_df)}\n")
            f.write(f"  Significant after FDR correction (q < 0.05): {len(sig_perm)}\n")
            if not sig_perm.empty:
                f.write("  Significant results:\n")
                for _, row in sig_perm.iterrows():
                    f.write(f"    Participant {row['participant']}, {row['feature']}: "
                            f"r = {row['pearson_r']}, perm_p = {row['permutation_p']}, "
                            f"perm_p_FDR = {row['permutation_p_fdr']}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 70 + "\n")

    print(f"\nPaper summary written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading participant data from:", PARTICIPANT_INFO_DIR)
    raw_data = load_participant_data()
    if not raw_data:
        print("ERROR: No participant data found. Check the path.")
        sys.exit(1)

    print(f"Loaded {len(raw_data)} video-level records.")
    df = extract_features(raw_data)
    # Convert feature columns to numeric (they may contain None -> NaN)
    for feat in ALL_FEATURES:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors="coerce")

    participants = sorted(df["participant"].unique())
    print(f"Participants: {participants}")
    print(f"Records per participant: {df.groupby('participant').size().to_dict()}")

    # ---- Concern 2: Missingness ----
    print("\n" + "=" * 60)
    print("CONCERN 2: MISSINGNESS REPORTING")
    print("=" * 60)
    miss_feat, miss_part, miss_reasons = report_missingness(df, OUTPUT_DIR)

    # Imputation
    df_imputed = impute_within_participant_median(df, ALL_FEATURES)
    impute_summary = []
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            continue
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

    # 1a. Bootstrap CIs + corrected p-values
    print("\n--- Per-participant correlations with bootstrap CIs ---")
    corr_df = per_participant_correlations(df, OUTPUT_DIR)

    # 1b. Mixed-effects (raw)
    print("\n--- Mixed-effects models (no imputation) ---")
    me_raw = run_mixed_effects(df, "no_imputation", OUTPUT_DIR)

    # 1c. Permutation tests
    print("\n--- Permutation tests ---")
    perm_df = run_permutation_tests(df, OUTPUT_DIR)

    # ---- Sensitivity analysis ----
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)
    sensitivity_df = sensitivity_analysis(df, df_imputed, OUTPUT_DIR)

    # ---- Paper summary ----
    write_paper_summary(miss_feat, miss_part, corr_df, me_raw, perm_df, sensitivity_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print("=" * 60)
    print("Files produced:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname} ({size:,} bytes)")


if __name__ == "__main__":
    main()
