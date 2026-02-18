"""
aggregate_biomarker_correlations.py
====================================
Aggregate biomarker correlation analysis across all participants.

Addresses reviewer concerns:

Concern 1 -- Statistical rigour:
  * Pearson correlations with 95% bootstrap confidence intervals
  * P-values with FDR (Benjamini-Hochberg) correction
  * Mixed-effects models (participant as random intercept)
  * Permutation tests for robustness

Concern 2 -- Missing data handling:
  * Detect truly missing data (zero totals → NaN, not 0)
  * Report missingness rates and reasons
  * Within-participant median imputation
  * Sensitivity analysis comparing imputed vs non-imputed results
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set styling for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000
RANDOM_SEED = 42


def load_participant_data(folder_path='participants'):
    """Load all participant JSON files from the specified folder."""
    participants_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and not filename.startswith('.'):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r') as file:
                    participant_name = filename.split('.')[0]
                    participants_data[participant_name] = json.load(file)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return participants_data


def calculate_emotion_metrics(emotions_dict):
    """
    Calculate various emotion metrics from the emotions dictionary.
    Returns None for all metrics when data is missing (not 0).
    """
    if not emotions_dict:
        return {
            'diversity': None,
            'non_neutral_percent': None,
            'positive_percent': None,
            'negative_percent': None,
            'percentages': {},
            'missing_reason': 'no_samples'
        }

    total_counts = sum(emotions_dict.values())

    if total_counts == 0:
        return {
            'diversity': None,
            'non_neutral_percent': None,
            'positive_percent': None,
            'negative_percent': None,
            'percentages': {},
            'missing_reason': 'zero_total_counts'
        }

    emotion_diversity = len(emotions_dict)
    neutral_count = emotions_dict.get('Neutral', 0)
    non_neutral_percent = (total_counts - neutral_count) / total_counts * 100

    positive_emotions = ['Happy', 'Surprise']
    negative_emotions = ['Sad', 'Angry', 'Fear', 'Disgust']

    positive_count = sum(emotions_dict.get(emotion, 0) for emotion in positive_emotions)
    negative_count = sum(emotions_dict.get(emotion, 0) for emotion in negative_emotions)

    positive_percent = positive_count / total_counts * 100
    negative_percent = negative_count / total_counts * 100

    emotion_percentages = {emotion: count / total_counts * 100
                           for emotion, count in emotions_dict.items()}

    return {
        'diversity': emotion_diversity,
        'non_neutral_percent': non_neutral_percent,
        'positive_percent': positive_percent,
        'negative_percent': negative_percent,
        'percentages': emotion_percentages,
        'missing_reason': None
    }


def calculate_eye_metrics(video_data):
    """
    Calculate eye-related metrics from the video data.
    Returns None (not 0) when sensor data is absent or total counts are zero.
    """
    blinks = video_data.get('blinks', {})
    fixations = video_data.get('fixations', {})
    pupil_diameter = video_data.get('pupil_diameter', {})

    result = {
        'total_blinks': None,
        'short_blinks': None,
        'medium_blinks': None,
        'long_blinks': None,
        'missing_blinks_reason': None,
        'missing_fixations_reason': None,
        'missing_pupil_reason': None,
    }

    # Blinks
    if blinks:
        total = sum(blinks.values())
        if total == 0:
            result['total_blinks'] = None
            result['short_blinks'] = None
            result['medium_blinks'] = None
            result['long_blinks'] = None
            result['missing_blinks_reason'] = 'zero_blinks_possible_tracker_loss'
        else:
            result['total_blinks'] = total
            result['short_blinks'] = blinks.get('short', 0)
            result['medium_blinks'] = blinks.get('medium', 0)
            result['long_blinks'] = blinks.get('long', 0)
    else:
        result['missing_blinks_reason'] = 'no_samples'

    # Fixations
    if fixations:
        any_data = False
        for duration in ['short', 'medium', 'long']:
            if duration in fixations:
                low_disp = fixations[duration].get('low_dispersion', 0)
                high_disp = fixations[duration].get('high_dispersion', 0)
                total = low_disp + high_disp
                if total > 0:
                    result[f'{duration}_total'] = total
                    result[f'{duration}_low_disp_ratio'] = low_disp / total
                    any_data = True
                else:
                    result[f'{duration}_total'] = None
                    result[f'{duration}_low_disp_ratio'] = None
            else:
                result[f'{duration}_total'] = None
                result[f'{duration}_low_disp_ratio'] = None
        if not any_data:
            result['missing_fixations_reason'] = 'zero_fixations_possible_tracker_loss'
    else:
        for duration in ['short', 'medium', 'long']:
            result[f'{duration}_total'] = None
            result[f'{duration}_low_disp_ratio'] = None
        result['missing_fixations_reason'] = 'no_samples'

    # Pupil diameter
    if pupil_diameter:
        total_pupil_readings = sum(pupil_diameter.values())
        if total_pupil_readings > 0:
            result['small_ratio'] = pupil_diameter.get('small', 0) / total_pupil_readings
            result['medium_ratio'] = pupil_diameter.get('medium', 0) / total_pupil_readings
            result['large_ratio'] = pupil_diameter.get('large', 0) / total_pupil_readings
        else:
            result['small_ratio'] = None
            result['medium_ratio'] = None
            result['large_ratio'] = None
            result['missing_pupil_reason'] = 'zero_readings_possible_tracker_loss'
    else:
        result['small_ratio'] = None
        result['medium_ratio'] = None
        result['large_ratio'] = None
        result['missing_pupil_reason'] = 'no_samples'

    return result


def extract_features(participants_data):
    """Extract features from all participants data for analysis."""
    all_features = []

    for participant_name, videos_data in participants_data.items():
        for video_data in videos_data:
            video_name = os.path.basename(video_data.get('video', 'unknown'))
            difficulty = video_data.get('difficulty', 'unknown')
            delta = video_data.get('delta')

            if delta is None:
                continue

            emotion_metrics = calculate_emotion_metrics(video_data.get('emotions', {}))
            eye_metrics = calculate_eye_metrics(video_data)

            missing_reason = emotion_metrics.pop('missing_reason', None)

            features = {
                'participant': participant_name,
                'video': video_name,
                'difficulty': difficulty,
                'delta': delta,
                'missing_emotions_reason': missing_reason,
                **emotion_metrics,
                **eye_metrics
            }

            all_features.append(features)

    return pd.DataFrame(all_features)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def safe_pearsonr(x, y):
    """Return (r, p) or (NaN, NaN) on failure."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    if len(xv) < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan, np.nan
    return pearsonr(xv, yv)


def safe_correlation(x, y):
    """Calculate correlation safely handling edge cases and missing data."""
    valid_data = [(x_val, y_val) for x_val, y_val in zip(x, y)
                  if x_val is not None and y_val is not None
                  and not (pd.isna(x_val) or pd.isna(y_val))]

    if not valid_data or len(valid_data) < 2:
        return np.nan, np.nan

    x_filtered, y_filtered = zip(*valid_data)
    x_array = np.array(x_filtered)
    y_array = np.array(y_filtered)

    if len(set(x_array)) <= 1 or len(set(y_array)) <= 1:
        return np.nan, np.nan

    try:
        return pearsonr(x_array, y_array)
    except Exception:
        return np.nan, np.nan


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
# Missingness reporting
# ---------------------------------------------------------------------------

def report_missingness(df, output_dir):
    """Report and save missingness rates."""
    numeric_cols = [c for c in df.columns
                    if c not in ['participant', 'video', 'difficulty', 'delta',
                                 'percentages', 'missing_emotions_reason',
                                 'missing_blinks_reason', 'missing_fixations_reason',
                                 'missing_pupil_reason']
                    and pd.api.types.is_numeric_dtype(df[c])]

    feat_rows = []
    for col in numeric_cols:
        n_total = len(df)
        n_miss = int(df[col].isna().sum())
        feat_rows.append({
            "feature": col,
            "total_observations": n_total,
            "missing_count": n_miss,
            "missing_rate_pct": round(n_miss / n_total * 100, 2) if n_total else 0,
        })
    feat_df = pd.DataFrame(feat_rows).sort_values("missing_rate_pct", ascending=False)
    feat_df.to_csv(os.path.join(output_dir, "missingness_rates.csv"), index=False)

    # Reason breakdown
    reason_cols = [c for c in df.columns if c.startswith("missing_") and c.endswith("_reason")]
    reason_rows = []
    for col in reason_cols:
        modality = col.replace("missing_", "").replace("_reason", "")
        for reason, count in df[col].value_counts(dropna=True).items():
            reason_rows.append({"modality": modality, "reason": reason, "count": int(count)})
    reason_df = pd.DataFrame(reason_rows)
    reason_df.to_csv(os.path.join(output_dir, "missingness_reasons.csv"), index=False)

    print("\n=== MISSINGNESS REPORT ===")
    print(feat_df.to_string(index=False))
    return feat_df


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
# Correlation analysis with CIs and p-values
# ---------------------------------------------------------------------------

def calculate_participant_correlations_with_ci(df):
    """
    Calculate correlations between features and delta for each participant,
    with bootstrap CIs and p-values.
    """
    participants = df['participant'].unique()
    numeric_features = [
        col for col in df.columns
        if col not in ['participant', 'video', 'difficulty', 'delta',
                       'percentages', 'missing_emotions_reason',
                       'missing_blinks_reason', 'missing_fixations_reason',
                       'missing_pupil_reason']
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    correlations = {}
    p_values = {}
    ci_data = []

    for participant in participants:
        participant_df = df[df['participant'] == participant]

        if len(participant_df) < 2:
            continue

        participant_corrs = {}
        participant_p_values = {}
        raw_pvals = []
        feat_indices = []

        for feature in numeric_features:
            valid_data = participant_df.dropna(subset=[feature])
            if len(valid_data) >= 2:
                x = valid_data[feature].values.astype(float)
                y = valid_data['delta'].values.astype(float)

                corr, p_val = safe_pearsonr(x, y)
                _, ci_lo, ci_hi = bootstrap_ci(x, y)

                participant_corrs[feature] = corr if not np.isnan(corr) else np.nan
                participant_p_values[feature] = p_val if not np.isnan(p_val) else np.nan

                ci_data.append({
                    "participant": participant,
                    "feature": feature,
                    "n_obs": len(valid_data),
                    "pearson_r": round(corr, 4) if not np.isnan(corr) else None,
                    "ci_95_low": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
                    "ci_95_high": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
                    "p_value_raw": round(p_val, 6) if not np.isnan(p_val) else None,
                })

                if not np.isnan(p_val):
                    raw_pvals.append(p_val)
                    feat_indices.append(len(ci_data) - 1)

        # FDR correction within participant
        if raw_pvals:
            _, adj, _, _ = multipletests(raw_pvals, method="fdr_bh")
            for idx, adj_p in zip(feat_indices, adj):
                ci_data[idx]["p_value_fdr"] = round(float(adj_p), 6)

        correlations[participant] = participant_corrs
        p_values[participant] = participant_p_values

    ci_df = pd.DataFrame(ci_data)
    return correlations, p_values, ci_df


def aggregate_correlations(correlations):
    """Aggregate correlations across participants."""
    all_correlations = defaultdict(list)
    abs_correlations = defaultdict(list)

    for participant, corrs in correlations.items():
        for feature, corr in corrs.items():
            if corr is not None and not np.isnan(corr):
                all_correlations[feature].append(corr)
                abs_correlations[feature].append(abs(corr))

    correlation_stats = {}
    for feature in all_correlations:
        if all_correlations[feature]:
            correlation_stats[feature] = {
                'mean': np.mean(all_correlations[feature]),
                'abs_mean': np.mean(abs_correlations[feature]),
                'median': np.median(all_correlations[feature]),
                'abs_median': np.median(abs_correlations[feature]),
                'std': np.std(all_correlations[feature]),
                'min': min(all_correlations[feature]),
                'max': max(all_correlations[feature]),
                'count': len(all_correlations[feature])
            }

    return correlation_stats


# ---------------------------------------------------------------------------
# Mixed-effects models
# ---------------------------------------------------------------------------

def run_mixed_effects(df, feature_cols, label, output_dir):
    """Fit delta ~ feature + C(difficulty) with participant random intercept."""
    results = []
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        mdf = df[["participant", "difficulty", "delta", feat]].dropna()
        if len(mdf) < 6 or mdf["participant"].nunique() < 2:
            results.append({"feature": feat, "n_obs": len(mdf),
                            "coef": None, "ci_low": None, "ci_high": None, "p_value": None})
            continue

        mdf = mdf.rename(columns={feat: "feature_value"})
        try:
            model = smf.mixedlm(
                "delta ~ feature_value + C(difficulty)", mdf,
                groups=mdf["participant"]
            )
            fit = model.fit(reml=False)
            coef = fit.params.get("feature_value")
            pval = fit.pvalues.get("feature_value")
            ci = fit.conf_int().loc["feature_value"].tolist()
            results.append({
                "feature": feat, "n_obs": len(mdf),
                "coef": round(float(coef), 6) if coef is not None else None,
                "ci_low": round(float(ci[0]), 6) if ci else None,
                "ci_high": round(float(ci[1]), 6) if ci else None,
                "p_value": round(float(pval), 6) if pval is not None else None,
            })
        except Exception as exc:
            print(f"  Mixed-effects failed for {feat}: {exc}")
            results.append({"feature": feat, "n_obs": len(mdf),
                            "coef": None, "ci_low": None, "ci_high": None, "p_value": None})

    res_df = pd.DataFrame(results)
    valid_mask = res_df["p_value"].notna()
    if valid_mask.any():
        _, adj, _, _ = multipletests(res_df.loc[valid_mask, "p_value"].values, method="fdr_bh")
        res_df.loc[valid_mask, "p_value_fdr"] = [round(float(a), 6) for a in adj]
    else:
        res_df["p_value_fdr"] = None

    res_df.to_csv(os.path.join(output_dir, f"mixed_effects_{label}.csv"), index=False)
    print(f"\n=== MIXED-EFFECTS ({label}) ===")
    sig = res_df[res_df["p_value_fdr"].notna() & (res_df["p_value_fdr"] < 0.05)]
    print(f"Tested: {len(res_df)}, Significant (FDR q<0.05): {len(sig)}")
    return res_df


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def get_display_feature_name(feature):
    """Convert feature names to proper display format for research plots."""
    display_name = feature.replace('_', ' ').replace('percentages.', '')

    replacements = {
        'non neutral percent': 'Non-Neutral Emotion Percentage',
        'positive percent': 'Positive Emotion Percentage',
        'negative percent': 'Negative Emotion Percentage',
        'diversity': 'Emotion Diversity (Count)',
        'total blinks': 'Total Blink Count',
        'short blinks': 'Short-Duration Blink Count',
        'medium blinks': 'Medium-Duration Blink Count',
        'long blinks': 'Long-Duration Blink Count',
        'short total': 'Total Short-Duration Fixations',
        'medium total': 'Total Medium-Duration Fixations',
        'long total': 'Total Long-Duration Fixations',
        'short low disp ratio': 'Low Dispersion Ratio for Short Fixations',
        'medium low disp ratio': 'Low Dispersion Ratio for Medium Fixations',
        'long low disp ratio': 'Low Dispersion Ratio for Long Fixations',
        'small ratio': 'Small Pupil Diameter Ratio',
        'medium ratio': 'Medium Pupil Diameter Ratio',
        'large ratio': 'Large Pupil Diameter Ratio'
    }

    if display_name.lower() in replacements:
        return replacements[display_name.lower()]

    if display_name.lower() in ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry']:
        return f'{display_name.capitalize()} Emotion Percentage'

    return ' '.join(word.capitalize() for word in display_name.split())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_aggregate_correlations(correlation_stats, title, filename, top_n=15, use_abs=True):
    """Plot aggregate correlations across participants with significance info."""
    if not correlation_stats:
        print(f"No valid correlation statistics for {filename}")
        return

    stats_df = pd.DataFrame(correlation_stats).T

    if len(stats_df) < top_n:
        top_n = len(stats_df)
        if top_n == 0:
            return

    if use_abs:
        stats_df = stats_df.sort_values('abs_mean', ascending=False)
    else:
        stats_df = stats_df.sort_values('mean', ascending=False)

    stats_df = stats_df.head(top_n)

    plt.figure(figsize=(12, 10))

    bars = plt.barh(
        y=stats_df.index,
        width=stats_df['mean'],
        xerr=stats_df['std'],
        alpha=0.8,
        color=[plt.cm.RdBu(0.5 * (x + 1)) for x in stats_df['mean']]
    )

    for i, bar in enumerate(bars):
        plt.text(
            0.01 if stats_df['mean'].iloc[i] < 0 else -0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{stats_df["mean"].iloc[i]:.2f} (n={int(stats_df["count"].iloc[i])})',
            va='center',
            ha='left' if stats_df['mean'].iloc[i] < 0 else 'right',
            color='black',
            fontweight='bold'
        )

    labels = [get_display_feature_name(label) for label in stats_df.index]
    plt.yticks(range(len(labels)), labels)

    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title(title + "\n(Error bars = SD across participants; see CSVs for CIs and p-values)",
              fontsize=14, fontweight='bold')
    plt.xlabel('Mean Pearson Correlation Coefficient with Learning Gain (Delta Score)', fontsize=14)
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_features_heatmap(correlations, emotion_features, eye_features, title, filename, top_n=5):
    """Plot a heatmap of top biomarker correlations across participants."""
    participants = list(correlations.keys())

    if not participants:
        return

    top_emotion_features = []
    for feature in emotion_features:
        feature_corrs = [abs(corrs.get(feature, 0)) for p, corrs in correlations.items()
                         if feature in corrs and corrs[feature] is not None and not np.isnan(corrs[feature])]
        if feature_corrs:
            top_emotion_features.append((feature, np.mean(feature_corrs)))

    top_eye_features = []
    for feature in eye_features:
        feature_corrs = [abs(corrs.get(feature, 0)) for p, corrs in correlations.items()
                         if feature in corrs and corrs[feature] is not None and not np.isnan(corrs[feature])]
        if feature_corrs:
            top_eye_features.append((feature, np.mean(feature_corrs)))

    top_emotion_features.sort(key=lambda x: x[1], reverse=True)
    top_emotion_features = [f[0] for f in top_emotion_features[:min(top_n, len(top_emotion_features))]]

    top_eye_features.sort(key=lambda x: x[1], reverse=True)
    top_eye_features = [f[0] for f in top_eye_features[:min(top_n, len(top_eye_features))]]

    all_top_features = top_emotion_features + top_eye_features
    if not all_top_features:
        return

    # Use NaN for missing data in heatmap (not 0)
    heatmap_data = []
    for participant in participants:
        row = []
        for feature in all_top_features:
            val = correlations[participant].get(feature)
            row.append(val if val is not None and not np.isnan(val) else np.nan)
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data, index=participants, columns=all_top_features)
    display_columns = [get_display_feature_name(col) for col in all_top_features]

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        heatmap_df.astype(float),
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Pearson r with Learning Gain (see CSVs for CIs)'},
        vmin=-1, vmax=1,
    )

    plt.title(title + "\n(See correlations_with_ci.csv for bootstrap CIs and FDR-corrected p-values)",
              fontsize=14, fontweight='bold')
    ax.set_xticklabels(display_columns, rotation=45, ha='right')
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Load data
    print("Loading participant data...")
    participants_data = load_participant_data()

    if not participants_data:
        print("No participant data found.")
        return

    print(f"Loaded data for {len(participants_data)} participants")

    # Extract features
    print("Extracting features...")
    features_df = extract_features(participants_data)

    # Process emotion percentages
    for i, row in features_df.iterrows():
        if isinstance(row.get('percentages'), dict):
            for emotion, percentage in row['percentages'].items():
                features_df.at[i, f'percentages.{emotion}'] = percentage

    if 'percentages' in features_df.columns:
        features_df = features_df.drop('percentages', axis=1)

    # Create output directory
    output_dir = 'aggregate_analysis'
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Identify numeric feature columns
    numeric_features = [
        col for col in features_df.columns
        if col not in ['participant', 'video', 'difficulty', 'delta',
                       'missing_emotions_reason', 'missing_blinks_reason',
                       'missing_fixations_reason', 'missing_pupil_reason']
        and pd.api.types.is_numeric_dtype(features_df[col])
    ]

    # --- Concern 2: Missingness ---
    print("\n=== MISSINGNESS REPORTING ===")
    report_missingness(features_df, '.')

    # Imputation
    df_imputed = impute_within_participant_median(features_df, numeric_features)

    # --- Concern 1: Statistical rigour ---
    print("\nCalculating participant correlations with bootstrap CIs...")
    participant_correlations, p_values, ci_df = calculate_participant_correlations_with_ci(features_df)
    ci_df.to_csv("correlations_with_ci.csv", index=False)

    # Aggregate correlations
    print("Aggregating correlations...")
    correlation_stats = aggregate_correlations(participant_correlations)

    # Mixed-effects models
    print("Running mixed-effects models...")
    me_raw = run_mixed_effects(features_df, numeric_features, "no_imputation", ".")
    me_imputed = run_mixed_effects(df_imputed, numeric_features, "with_imputation", ".")

    # Sensitivity analysis
    sensitivity = me_raw.merge(me_imputed, on="feature", suffixes=("_raw", "_imputed"))
    sensitivity["direction_changed"] = (
        np.sign(sensitivity["coef_raw"].fillna(0)) != np.sign(sensitivity["coef_imputed"].fillna(0))
    )
    sensitivity.to_csv("sensitivity_analysis.csv", index=False)
    n_changed = int(sensitivity["direction_changed"].sum())
    print(f"\nSensitivity: direction changed for {n_changed}/{len(sensitivity)} biomarkers after imputation.")

    # Permutation tests
    print("Running permutation tests...")
    perm_rows = []
    for participant in sorted(features_df['participant'].unique()):
        pdf = features_df[features_df['participant'] == participant]
        delta = pdf['delta'].values.astype(float)
        raw_pvals = []
        row_indices = []
        for feat in numeric_features:
            if feat not in pdf.columns:
                continue
            x = pdf[feat].values.astype(float)
            r_obs, p_perm = permutation_test(x, delta)
            perm_rows.append({
                "participant": participant, "feature": feat,
                "n_obs": int(np.sum(~(np.isnan(x) | np.isnan(delta)))),
                "pearson_r": round(r_obs, 4) if not np.isnan(r_obs) else None,
                "permutation_p": round(p_perm, 6) if not np.isnan(p_perm) else None,
            })
            if not np.isnan(p_perm):
                raw_pvals.append(p_perm)
                row_indices.append(len(perm_rows) - 1)

        if raw_pvals:
            _, adj, _, _ = multipletests(raw_pvals, method="fdr_bh")
            for idx, adj_p in zip(row_indices, adj):
                perm_rows[idx]["permutation_p_fdr"] = round(float(adj_p), 6)

    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv("permutation_tests.csv", index=False)
    print(f"Permutation tests complete: {len(perm_df)} tests")

    # Define feature groups
    emotion_features = [
        'diversity', 'non_neutral_percent', 'positive_percent', 'negative_percent'
    ]
    emotion_features.extend([col for col in features_df.columns if col.startswith('percentages.')])

    eye_features = [
        'total_blinks', 'short_blinks', 'medium_blinks', 'long_blinks',
        'short_total', 'medium_total', 'long_total',
        'short_low_disp_ratio', 'medium_low_disp_ratio', 'long_low_disp_ratio',
        'small_ratio', 'medium_ratio', 'large_ratio'
    ]

    emotion_features = [f for f in emotion_features if f in correlation_stats]
    eye_features = [f for f in eye_features if f in correlation_stats]

    # Plot aggregate correlations
    print("Creating aggregate correlation plots...")
    plot_aggregate_correlations(
        {k: v for k, v in correlation_stats.items() if k in emotion_features},
        "Mean Correlations Between Emotional Biomarkers and Learning Gain\nAcross All Participants",
        "top_emotion_features.png",
        top_n=10
    )

    plot_aggregate_correlations(
        {k: v for k, v in correlation_stats.items() if k in eye_features},
        "Mean Correlations Between Eye-Tracking Biomarkers and Learning Gain\nAcross All Participants",
        "top_eye_features.png",
        top_n=10
    )

    # Create heatmap
    print("Creating correlation heatmap...")
    plot_top_features_heatmap(
        participant_correlations,
        emotion_features,
        eye_features,
        "Top Biomarkers and Their Correlation with Learning Gain Across Participants",
        "correlation_heatmap.png"
    )

    # Write summary
    with open("analysis_summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("AGGREGATE BIOMARKER CORRELATION ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("MISSING DATA:\n")
        f.write(f"  See missingness_rates.csv and missingness_reasons.csv\n")
        f.write(f"  Imputation: within-participant median\n")
        f.write(f"  Sensitivity: {n_changed}/{len(sensitivity)} biomarkers changed direction\n\n")

        f.write("CORRELATIONS:\n")
        if not ci_df.empty and "p_value_fdr" in ci_df.columns:
            sig = ci_df[ci_df["p_value_fdr"].notna() & (ci_df["p_value_fdr"] < 0.05)]
            f.write(f"  Total tests: {len(ci_df)}\n")
            f.write(f"  Significant after FDR (q<0.05): {len(sig)}\n")

        f.write("\nMIXED-EFFECTS MODELS:\n")
        sig_me = me_raw[me_raw["p_value_fdr"].notna() & (me_raw["p_value_fdr"] < 0.05)]
        f.write(f"  Total biomarkers: {len(me_raw)}\n")
        f.write(f"  Significant after FDR (q<0.05): {len(sig_me)}\n")
        if not sig_me.empty:
            for _, row in sig_me.iterrows():
                f.write(f"    {row['feature']}: coef={row['coef']}, p_FDR={row['p_value_fdr']}\n")

        f.write("\nPERMUTATION TESTS:\n")
        if not perm_df.empty and "permutation_p_fdr" in perm_df.columns:
            sig_perm = perm_df[perm_df["permutation_p_fdr"].notna() & (perm_df["permutation_p_fdr"] < 0.05)]
            f.write(f"  Total tests: {len(perm_df)}\n")
            f.write(f"  Significant after FDR (q<0.05): {len(sig_perm)}\n")

    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
