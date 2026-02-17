import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

def get_display_name(metric):
    """Convert metric names to display format."""
    # Handle ECG metrics
    if metric.startswith('diff_ecg_'):
        metric_name = metric.replace('diff_ecg_', '')
        if metric_name == 'heart_rate':
            return 'Heart Rate Change'
        elif metric_name == 'sdnn':
            return 'HRV Change'
        elif metric_name == 'rmssd':
            return 'RMSSD Change'
        elif metric_name == 'pnn50':
            return 'PNN50 Change'
    
    # Handle emotion categories
    if metric == 'positive':
        return 'Positive Emotions'
    elif metric == 'negative':
        return 'Negative Emotions'
    elif metric == 'non_neutral':
        return 'Non-Neutral Emotions'
    
    # Handle other metrics
    name = metric.replace('_', ' ').title()
    name = name.replace('Ecg ', 'ECG ')
    name = name.replace('Pct ', 'Percentage ')
    name = name.replace('Disp ', 'Dispersion ')
    return name

# Define the directory containing JSON files (replace with your directory path)
directory = "./participant_info"

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


def calculate_percentages(metrics):
    """Calculate the percentage of each biomarker within the given metrics."""
    percentages = {}

    # Emotions
    emotions = metrics.get("emotions", {})
    total_emotions = sum(emotions.values())
    for emotion in all_emotions:
        percentages[emotion] = (
            (emotions.get(emotion, 0) / total_emotions * 100)
            if total_emotions > 0
            else 0
        )

    # Grouped emotions
    positive_sum = sum(emotions.get(emo, 0) for emo in positive_emotions)
    negative_sum = sum(emotions.get(emo, 0) for emo in negative_emotions)
    non_neutral_sum = sum(
        emotions.get(emo, 0) for emo in all_emotions if emo != "Neutral"
    )
    percentages["positive"] = (
        (positive_sum / total_emotions * 100) if total_emotions > 0 else 0
    )
    percentages["negative"] = (
        (negative_sum / total_emotions * 100) if total_emotions > 0 else 0
    )
    percentages["non_neutral"] = (
        (non_neutral_sum / total_emotions * 100) if total_emotions > 0 else 0
    )

    # Blinks
    blinks = metrics.get("blinks", {})
    total_blinks = sum(blinks.values())
    for blink in ["short", "medium", "long"]:
        percentages[f"blinks_{blink}"] = (
            (blinks.get(blink, 0) / total_blinks * 100) if total_blinks > 0 else 0
        )

    # Fixations
    fixations = metrics.get("fixations", {})
    total_fixations = (
        sum(sum(dispersion_dict.values()) for dispersion_dict in fixations.values())
        if fixations
        else 0
    )
    for duration in ["short", "medium", "long"]:
        for dispersion in ["low_dispersion", "high_dispersion"]:
            key = f"fixations_{duration}_{dispersion}"
            count = fixations.get(duration, {}).get(dispersion, 0)
            percentages[key] = (
                (count / total_fixations * 100) if total_fixations > 0 else 0
            )

    # Pupil diameter
    pupil_diameter = metrics.get("pupil_diameter", {})
    total_pupil = sum(pupil_diameter.values())
    for size in ["small", "medium", "large"]:
        percentages[f"pupil_diameter_{size}"] = (
            (pupil_diameter.get(size, 0) / total_pupil * 100) if total_pupil > 0 else 0
        )

    # ECG metrics (store raw values, not percentages)
    ecg_metrics = metrics.get("ecg_metrics", {})
    for key, value in ecg_metrics.items():
        percentages[f"ecg_{key}"] = value

    return percentages

def calculate_ecg_differences(video_ecg, break_ecg):
    """Calculate the difference between video and break ECG metrics."""
    differences = {}
    for key in video_ecg:
        if key.startswith('ecg_'):
            video_val = video_ecg.get(key, 0)
            break_val = break_ecg.get(key, 0)
            differences[f"diff_{key}"] = video_val - break_val
    return differences

def safe_correlation(x, y):
    """Calculate correlation safely handling edge cases."""
    # Convert inputs to numpy arrays and remove any NaN values
    x = np.array(x)
    y = np.array(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Check for valid correlation conditions
    if len(x) < 2 or len(set(x)) <= 1 or len(set(y)) <= 1:
        return 0, 1.0
    
    try:
        return pearsonr(x, y)
    except Exception as e:
        print(f"Error in correlation calculation: {e}")
        return 0, 1.0

# List to collect correlations for aggregate plot
all_correlations = []

# Process each JSON file in the directory
for json_file in os.listdir(directory):
    if json_file.endswith(".json"):
        with open(os.path.join(directory, json_file)) as f:
            data = json.load(f)

        participant_id = data[0]["participant_id"]
        print(f"\nProcessing participant {participant_id}")

        # Collect change percentages, ECG differences, and deltas
        change_percentages = {biomarker: [] for biomarker in all_biomarkers}
        ecg_differences = {f"diff_ecg_{metric}": [] for metric in ["heart_rate", "sdnn", "rmssd", "pnn50"]}
        deltas = []

        for entry in data:
            overall_percentages = calculate_percentages(entry)
            break_percentages = calculate_percentages(entry["break_metrics"])

            # Calculate regular biomarker changes
            for biomarker in all_biomarkers:
                if not biomarker.startswith('ecg_'):  # Skip ECG metrics here
                    change = overall_percentages.get(biomarker, 0) - break_percentages.get(
                        biomarker, 0
                    )
                    change_percentages[biomarker].append(change)

            # Calculate ECG differences
            ecg_diffs = calculate_ecg_differences(overall_percentages, break_percentages)
            for key, value in ecg_diffs.items():
                ecg_differences[key].append(value)

            deltas.append(entry["delta"])

        print(f"Number of data points: {len(deltas)}")

        # Compute correlations for regular biomarkers
        correlations = {}
        for biomarker in all_biomarkers:
            if not biomarker.startswith('ecg_'):  # Skip ECG metrics here
                if len(change_percentages[biomarker]) > 1:
                    corr, _ = safe_correlation(change_percentages[biomarker], deltas)
                    correlations[biomarker] = corr
                else:
                    correlations[biomarker] = 0

        # Compute correlations for ECG differences
        for ecg_key in ecg_differences:
            if len(ecg_differences[ecg_key]) > 1:
                corr, _ = safe_correlation(ecg_differences[ecg_key], deltas)
                correlations[ecg_key] = corr
            else:
                correlations[ecg_key] = 0

        # Create separate bar plots for different metric types
        metric_groups = {
            'Emotion': [b for b in all_biomarkers if any(e in b.lower() for e in ['emotion', 'neutral', 'happy', 'sad', 'fear', 'disgust', 'angry', 'surprise', 'positive', 'negative', 'non_neutral'])],
            'Eye-Tracking': [b for b in all_biomarkers if any(e in b for e in ['blink', 'fixation', 'pupil'])],
            'ECG': [k for k in correlations.keys() if k.startswith('diff_ecg_')]
        }

        # For individual participants, create a single bar plot with all metrics
        print(f"Creating correlation plot...")
        # Sort all metrics by correlation strength
        all_metrics = []
        for metrics in metric_groups.values():
            all_metrics.extend(metrics)

        sorted_metrics = sorted(
            [(m, correlations[m]) for m in all_metrics],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        if sorted_metrics:  # If we have any metrics
            fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_metrics) * 0.4)))
            
            # Prepare data for plotting
            metric_names = [m[0] for m in sorted_metrics]
            corr_values = [m[1] for m in sorted_metrics]
            
            # Create horizontal bars
            colors = plt.cm.RdBu((np.array(corr_values) + 1) / 2)
            bars = ax.barh(range(len(metric_names)), corr_values, color=colors)
            
            # Customize the plot
            ax.set_yticks(range(len(metric_names)))
            ax.set_yticklabels([get_display_name(m) for m in metric_names], fontsize=10, weight='bold')
            ax.set_xlabel("Correlation with Learning Gain", fontsize=12)
            
            # Add correlation values as text
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + np.sign(width) * 0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{width:.2f}',
                       ha='left' if width >= 0 else 'right',
                       va='center',
                       weight='bold'
                       )
            
            # Add gridlines
            ax.grid(True, axis='x', alpha=0.3)
            ax.set_axisbelow(True)
            
            # Set x-axis limits symmetrically
            max_abs_corr = max(abs(min(corr_values)), abs(max(corr_values)))
            ax.set_xlim(-max_abs_corr * 1.1, max_abs_corr * 1.1)
            
            plt.tight_layout()
            plt.savefig(
                f"participant_{participant_id}_correlation.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

        # Save all data to CSV
        df_data = {
            'Metric': [],
            'Correlation': [],
            'Type': []
        }
        
        for group_name, metrics in metric_groups.items():
            for metric in metrics:
                if metric in correlations:
                    df_data['Metric'].append(get_display_name(metric))
                    df_data['Correlation'].append(correlations[metric])
                    df_data['Type'].append(group_name)
        
        df = pd.DataFrame(df_data)
        df.to_csv(f"participant_{participant_id}_correlations.csv", index=False)

        # Collect correlations for aggregate plot
        all_correlations.append({"participant_id": str(participant_id), **correlations})

print("\nCreating aggregate correlation heatmaps...")

# Create DataFrame for aggregate correlations
corr_df = pd.DataFrame(all_correlations)
corr_df.set_index("participant_id", inplace=True)
corr_df = corr_df.sort_index()

# Create separate heatmaps for each metric group
for group_name, metrics in metric_groups.items():
    metrics = [m for m in metrics if m in corr_df.columns]
    if not metrics:
        print(f"No metrics found for {group_name} in aggregate analysis")
        continue
        
    print(f"Creating {group_name} aggregate heatmap...")
    plt.figure(figsize=(max(10, len(metrics) * 0.6), max(8, len(corr_df) * 0.6)))
    sns.heatmap(
        corr_df[metrics],
        cmap="RdBu",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 8},
        cbar_kws={"label": "Correlation Coefficient"}
    )
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Participant ID", fontsize=12)
    
    # Format x-axis labels
    x_labels = [get_display_name(label) for label in metrics]
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"aggregate_{group_name.lower()}_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

print("\nProcessing complete. Plots and CSV files have been saved.")
