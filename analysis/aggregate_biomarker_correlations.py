import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set styling for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

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
    """Calculate various emotion metrics from the emotions dictionary."""
    # Guard against None or empty dictionaries
    if not emotions_dict:
        return {
            'diversity': None,
            'non_neutral_percent': None,
            'positive_percent': None,
            'negative_percent': None,
            'percentages': {}
        }
    
    total_counts = sum(emotions_dict.values())
    
    # Calculate emotion diversity (number of different emotions)
    emotion_diversity = len(emotions_dict)
    
    # Calculate percentage of non-neutral emotions
    neutral_count = emotions_dict.get('Neutral', 0)
    non_neutral_percent = 0 if total_counts == 0 else (total_counts - neutral_count) / total_counts * 100
    
    # Calculate positive and negative emotion percentages
    positive_emotions = ['Happy', 'Surprise']  # Consider Surprise as positive in learning context
    negative_emotions = ['Sad', 'Angry', 'Fear', 'Disgust']
    
    positive_count = sum(emotions_dict.get(emotion, 0) for emotion in positive_emotions)
    negative_count = sum(emotions_dict.get(emotion, 0) for emotion in negative_emotions)
    
    positive_percent = 0 if total_counts == 0 else positive_count / total_counts * 100
    negative_percent = 0 if total_counts == 0 else negative_count / total_counts * 100
    
    # Calculate percentages for each emotion
    emotion_percentages = {emotion: count/total_counts*100 for emotion, count in emotions_dict.items()} if total_counts > 0 else {}
    
    return {
        'diversity': emotion_diversity,
        'non_neutral_percent': non_neutral_percent,
        'positive_percent': positive_percent,
        'negative_percent': negative_percent,
        'percentages': emotion_percentages
    }

def calculate_eye_metrics(video_data):
    """Calculate eye-related metrics from the video data."""
    # Check if the components are present
    blinks = video_data.get('blinks', {})
    fixations = video_data.get('fixations', {})
    pupil_diameter = video_data.get('pupil_diameter', {})
    
    # Initialize results dictionary with None values to represent missing data
    result = {
        'total_blinks': None,
        'short_blinks': None,
        'medium_blinks': None,
        'long_blinks': None
    }
    
    # Only calculate blink metrics if blink data exists
    if blinks:
        # If blink counts are 0, keep them as 0 (not missing)
        result['total_blinks'] = sum(blinks.values())
        result['short_blinks'] = blinks.get('short', 0)
        result['medium_blinks'] = blinks.get('medium', 0)
        result['long_blinks'] = blinks.get('long', 0)
    
    # Calculate fixation metrics if data exists
    if fixations:
        for duration in ['short', 'medium', 'long']:
            if duration in fixations:
                low_disp = fixations[duration].get('low_dispersion', 0)
                high_disp = fixations[duration].get('high_dispersion', 0)
                total = low_disp + high_disp
                result[f'{duration}_total'] = total
                # For ratios, only calculate if the total is non-zero
                result[f'{duration}_low_disp_ratio'] = low_disp / total if total > 0 else 0
            else:
                result[f'{duration}_total'] = None
                result[f'{duration}_low_disp_ratio'] = None
    
    # Calculate pupil diameter metrics if data exists
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
    else:
        result['small_ratio'] = None
        result['medium_ratio'] = None
        result['large_ratio'] = None
    
    return result

def extract_features(participants_data):
    """Extract features from all participants data for analysis."""
    all_features = []
    
    for participant_name, videos_data in participants_data.items():
        for video_data in videos_data:
            video_name = os.path.basename(video_data.get('video', 'unknown'))
            difficulty = video_data.get('difficulty', 'unknown')
            delta = video_data.get('delta', 0)  # delta of 0 is a valid value
            
            # Extract emotion metrics
            emotion_metrics = calculate_emotion_metrics(video_data.get('emotions', {}))
            
            # Extract eye metrics
            eye_metrics = calculate_eye_metrics(video_data)
            
            # Combine all features
            features = {
                'participant': participant_name,
                'video': video_name,
                'difficulty': difficulty,
                'delta': delta,
                **emotion_metrics,
                **eye_metrics
            }
            
            all_features.append(features)
    
    return pd.DataFrame(all_features)

def safe_correlation(x, y):
    """Calculate correlation safely handling edge cases and missing data."""
    # Filter out None/NaN values while keeping legitimate zeros
    valid_data = [(x_val, y_val) for x_val, y_val in zip(x, y) 
                 if x_val is not None and y_val is not None 
                 and not (pd.isna(x_val) or pd.isna(y_val))]
    
    # If no valid data pairs remain, return no correlation
    if not valid_data:
        return 0
    
    # Convert to numpy arrays
    x_filtered, y_filtered = zip(*valid_data)
    x_array = np.array(x_filtered)
    y_array = np.array(y_filtered)
    
    # Check if all values in either array are the same (constant)
    if len(set(x_array)) <= 1 or len(set(y_array)) <= 1:
        return 0  # No correlation if one array is constant
    
    # Need at least 2 data points for correlation
    if len(x_array) < 2:
        return 0
    
    try:
        # Calculate Pearson correlation
        correlation_matrix = np.corrcoef(x_array, y_array)
        return correlation_matrix[0, 1]  # Get the correlation coefficient
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 0  # Default to no correlation on error

def calculate_participant_correlations(df):
    """Calculate correlations between features and delta for each participant."""
    participants = df['participant'].unique()
    all_features = df.columns.drop(['participant', 'video', 'difficulty', 'delta'])
    
    correlations = {}
    
    for participant in participants:
        participant_df = df[df['participant'] == participant]
        
        # Skip if not enough data points
        if len(participant_df) < 2:
            continue
        
        participant_corrs = {}
        for feature in all_features:
            if feature in participant_df.columns:
                # Only include features with valid data
                valid_data = participant_df.dropna(subset=[feature])
                if len(valid_data) >= 2:
                    corr = safe_correlation(valid_data['delta'].values, valid_data[feature].values)
                    participant_corrs[feature] = corr
        
        correlations[participant] = participant_corrs
    
    return correlations

def aggregate_correlations(correlations):
    """Aggregate correlations across participants."""
    # Initialize dictionaries to track correlation statistics
    all_correlations = defaultdict(list)
    abs_correlations = defaultdict(list)
    
    # Collect all correlations by feature
    for participant, corrs in correlations.items():
        for feature, corr in corrs.items():
            # Only include non-None correlations
            if corr is not None:
                all_correlations[feature].append(corr)
                abs_correlations[feature].append(abs(corr))
    
    # Calculate statistics only for features with data
    correlation_stats = {}
    for feature in all_correlations:
        if all_correlations[feature]:  # Check if list is not empty
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

def get_display_feature_name(feature):
    """Convert feature names to proper display format for research plots."""
    # First replace underscores with spaces and remove prefixes
    display_name = feature.replace('_', ' ').replace('percentages.', '')
    
    # Further refinements for specific features
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
    
    # Apply specific replacements if available
    if display_name.lower() in replacements:
        return replacements[display_name.lower()]
    
    # For emotion percentages not specifically mapped
    if display_name.lower() in ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry']:
        return f'{display_name.capitalize()} Emotion Percentage'
    
    # For anything else, just capitalize each word
    return ' '.join(word.capitalize() for word in display_name.split())

def plot_aggregate_correlations(correlation_stats, title, filename, top_n=15, use_abs=True):
    """Plot aggregate correlations across participants."""
    # Skip if no valid statistics
    if not correlation_stats:
        print(f"No valid correlation statistics for {filename}")
        return
    
    # Convert to dataframe for easier sorting
    stats_df = pd.DataFrame(correlation_stats).T
    
    # If we don't have enough features, adjust top_n
    if len(stats_df) < top_n:
        top_n = len(stats_df)
        if top_n == 0:
            print(f"No features to plot for {filename}")
            return
    
    # Sort by absolute mean correlation if requested
    if use_abs:
        stats_df = stats_df.sort_values('abs_mean', ascending=False)
    else:
        stats_df = stats_df.sort_values('mean', ascending=False)
    
    # Take top N features
    stats_df = stats_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot horizontal bars for mean correlation
    bars = plt.barh(
        y=stats_df.index,
        width=stats_df['mean'],
        xerr=stats_df['std'],
        alpha=0.8,
        color=[plt.cm.RdBu(0.5 * (x + 1)) for x in stats_df['mean']]
    )
    
    # Add mean values as text
    for i, bar in enumerate(bars):
        plt.text(
            0.01 if stats_df['mean'].iloc[i] < 0 else -0.01,
            bar.get_y() + bar.get_height()/2,
            f'{stats_df["mean"].iloc[i]:.2f}',
            va='center',
            ha='left' if stats_df['mean'].iloc[i] < 0 else 'right',
            color='black',
            fontweight='bold'
        )
    
    # Improve feature labels for display
    labels = [get_display_feature_name(label) for label in stats_df.index]
    plt.yticks(range(len(labels)), labels)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Mean Pearson Correlation Coefficient with Learning Gain (Delta Score)', fontsize=14)
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_distribution(correlations, feature, title, filename):
    """Plot the distribution of correlations for a specific feature across participants."""
    # Collect correlations for the feature
    feature_corrs = [corrs.get(feature, 0) for participant, corrs in correlations.items() 
                    if feature in corrs and corrs[feature] is not None]
    
    if not feature_corrs:
        print(f"No data available for feature {feature}")
        return
    
    # Create a dataframe for the feature correlations
    df = pd.DataFrame({
        'participant': [participant for participant, corrs in correlations.items() 
                        if feature in corrs and corrs[feature] is not None],
        'correlation': feature_corrs
    })
    
    # Check if we have enough data
    if len(df) < 2:
        print(f"Not enough participants with data for feature {feature}")
        return
    
    # Sort by correlation value
    df = df.sort_values('correlation')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    bars = plt.bar(
        df['participant'],
        df['correlation'],
        alpha=0.8,
        color=[plt.cm.RdBu(0.5 * (x + 1)) for x in df['correlation']],
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add correlation values as text
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            0.01 if df['correlation'].iloc[i] >= 0 else -0.01,
            f'{df["correlation"].iloc[i]:.2f}',
            ha='center',
            va='bottom' if df['correlation'].iloc[i] >= 0 else 'top',
            color='black',
            fontweight='bold'
        )
    
    # Display feature name nicely
    display_feature = get_display_feature_name(feature)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f"Distribution of {display_feature} Correlation with Learning Gain\nAcross All Participants", fontsize=16, fontweight='bold')
    plt.ylabel('Pearson Correlation Coefficient', fontsize=14)
    plt.xlabel('Participant', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add statistics
    mean_corr = np.mean(df['correlation'])
    std_corr = np.std(df['correlation'])
    plt.figtext(0.5, 0.01, 
              f"Mean correlation: {mean_corr:.2f} ± {std_corr:.2f} (standard deviation)",
              ha="center", fontsize=10, 
              bbox={"facecolor":"white", "alpha":0.8, "pad":5, "edgecolor":"lightgray"})
    
    plt.tight_layout(pad=2.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_features_heatmap(correlations, emotion_features, eye_features, title, filename, top_n=5):
    """Plot a heatmap of top biomarker correlations across participants."""
    # Prepare data for heatmap
    participants = list(correlations.keys())
    
    if not participants:
        print(f"No participant data for heatmap {filename}")
        return
    
    # Get top emotion features by absolute mean correlation
    top_emotion_features = []
    for feature in emotion_features:
        feature_corrs = [abs(corrs.get(feature, 0)) for p, corrs in correlations.items() 
                        if feature in corrs and corrs[feature] is not None]
        if feature_corrs:
            top_emotion_features.append((feature, np.mean(feature_corrs)))
    
    # Get top eye features by absolute mean correlation
    top_eye_features = []
    for feature in eye_features:
        feature_corrs = [abs(corrs.get(feature, 0)) for p, corrs in correlations.items() 
                        if feature in corrs and corrs[feature] is not None]
        if feature_corrs:
            top_eye_features.append((feature, np.mean(feature_corrs)))
    
    # If we don't have enough features, adjust
    if len(top_emotion_features) < top_n:
        emotion_top_n = len(top_emotion_features)
    else:
        emotion_top_n = top_n
        
    if len(top_eye_features) < top_n:
        eye_top_n = len(top_eye_features)
    else:
        eye_top_n = top_n
    
    # Sort and select top features
    top_emotion_features.sort(key=lambda x: x[1], reverse=True)
    top_emotion_features = [f[0] for f in top_emotion_features[:emotion_top_n]]
    
    top_eye_features.sort(key=lambda x: x[1], reverse=True)
    top_eye_features = [f[0] for f in top_eye_features[:eye_top_n]]
    
    # Combine top features
    all_top_features = top_emotion_features + top_eye_features
    
    if not all_top_features:
        print(f"No valid features for heatmap {filename}")
        return
    
    # Create heatmap data
    heatmap_data = []
    for participant in participants:
        row = []
        for feature in all_top_features:
            # Use 0 for missing data in heatmap
            row.append(correlations[participant].get(feature, 0))
        heatmap_data.append(row)
    
    # Convert to dataframe
    heatmap_df = pd.DataFrame(heatmap_data, index=participants, columns=all_top_features)
    
    # Display feature names nicely
    display_columns = [get_display_feature_name(col) for col in all_top_features]
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        heatmap_df,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient with Learning Gain'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    ax.set_xticklabels(display_columns, rotation=45, ha='right')
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def group_by_difficulty(df):
    """Group data by difficulty and analyze correlations separately."""
    difficulty_groups = {}
    
    for difficulty in df['difficulty'].unique():
        difficulty_df = df[df['difficulty'] == difficulty]
        difficulty_groups[difficulty] = difficulty_df
    
    return difficulty_groups

def plot_difficulty_comparison(df, feature, title, filename):
    """Plot comparison of a feature's values by difficulty level and learning outcome."""
    # Skip if feature doesn't exist or has no valid data
    if feature not in df.columns or df[feature].count() < 2:
        print(f"No valid data for feature {feature} in difficulty comparison")
        return
    
    # Remove rows with missing values for this feature
    valid_df = df.dropna(subset=[feature])
    
    if len(valid_df) < 2:
        print(f"Not enough valid data for feature {feature} after filtering")
        return
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Define colors for positive and zero/negative delta
    colors = {'Positive': 'green', 'Zero/Negative': 'red'}
    
    # Create categories for learning outcome
    valid_df = valid_df.copy()  # Create a copy to avoid SettingWithCopyWarning
    valid_df['learning_category'] = valid_df['delta'].apply(lambda x: 'Positive' if x > 0 else 'Zero/Negative')
    
    # Check if we have data for both categories
    if len(valid_df['learning_category'].unique()) < 2:
        print(f"Missing either positive or negative learning data for feature {feature}")
        # Continue anyway, just with the available category
    
    # Create boxplot
    ax = sns.boxplot(
        x='difficulty',
        y=feature,
        hue='learning_category',
        data=valid_df,
        palette=colors
    )
    
    # Add individual points
    sns.stripplot(
        x='difficulty',
        y=feature,
        hue='learning_category',
        data=valid_df,
        dodge=True,
        alpha=0.6,
        palette=colors,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Display feature name nicely
    display_feature = get_display_feature_name(feature)
    
    plt.title(f"Comparison of {display_feature} by Video Difficulty and Learning Outcome", fontsize=16, fontweight='bold')
    plt.ylabel(display_feature, fontsize=14)
    plt.xlabel('Video Difficulty Level', fontsize=14)
    plt.legend(title="Learning Outcome", frameon=True, fancybox=True)
    
    # Add statistics
    categories = []
    for difficulty in valid_df['difficulty'].unique():
        for outcome in valid_df['learning_category'].unique():
            subset = valid_df[(valid_df['difficulty'] == difficulty) & (valid_df['learning_category'] == outcome)]
            if len(subset) > 0:
                categories.append(f"{difficulty}-{outcome}: n={len(subset)}, mean={subset[feature].mean():.2f}")
    
    plt.figtext(0.5, 0.01, 
              "Sample sizes and means: " + " | ".join(categories),
              ha="center", fontsize=10, 
              bbox={"facecolor":"white", "alpha":0.8, "pad":5, "edgecolor":"lightgray"})
    
    plt.tight_layout(pad=2.5)
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
    
    # Drop the original percentages column
    if 'percentages' in features_df.columns:
        features_df = features_df.drop('percentages', axis=1)
    
    # Create output directory
    output_dir = 'aggregate_analysis'
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Calculate correlations for each participant
    print("Calculating participant correlations...")
    participant_correlations = calculate_participant_correlations(features_df)
    
    # Aggregate correlations
    print("Aggregating correlations...")
    correlation_stats = aggregate_correlations(participant_correlations)
    
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
    
    # Filter features that actually exist in our data
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
    
    # Plot distribution of correlations for top features
    print("Creating correlation distribution plots...")
    
    # Top emotion features
    emotion_stats = {k: v for k, v in correlation_stats.items() if k in emotion_features}
    if emotion_stats:
        top_emotion_features = sorted(emotion_stats.keys(), key=lambda x: emotion_stats[x]['abs_mean'], reverse=True)
        # Limit to 5 or fewer features
        top_emotion_features = top_emotion_features[:min(5, len(top_emotion_features))]
        
        for feature in top_emotion_features:
            plot_correlation_distribution(
                participant_correlations,
                feature,
                "Distribution of Emotional Biomarker Correlations",
                f"correlation_dist_{feature.replace('.', '_')}.png"
            )
    
    # Top eye features
    eye_stats = {k: v for k, v in correlation_stats.items() if k in eye_features}
    if eye_stats:
        top_eye_features = sorted(eye_stats.keys(), key=lambda x: eye_stats[x]['abs_mean'], reverse=True)
        # Limit to 5 or fewer features
        top_eye_features = top_eye_features[:min(5, len(top_eye_features))]
        
        for feature in top_eye_features:
            plot_correlation_distribution(
                participant_correlations,
                feature,
                "Distribution of Eye-Tracking Biomarker Correlations",
                f"correlation_dist_{feature}.png"
            )
    
    # Create heatmap of top features
    print("Creating correlation heatmap...")
    plot_top_features_heatmap(
        participant_correlations,
        emotion_features,
        eye_features,
        "Top Biomarkers and Their Correlation with Learning Gain Across Participants",
        "correlation_heatmap.png"
    )
    
    # Analyze by difficulty
    print("Analyzing by difficulty level...")
    difficulty_groups = group_by_difficulty(features_df)
    
    # Plot difficulty comparisons for top features
    all_top_features = []
    if emotion_stats:
        all_top_features.extend(top_emotion_features)
    if eye_stats:
        all_top_features.extend(top_eye_features)
    
    for feature in all_top_features:
        plot_difficulty_comparison(
            features_df,
            feature,
            "Biomarker Comparison by Difficulty",
            f"difficulty_comparison_{feature.replace('.', '_')}.png"
        )
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 