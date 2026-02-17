import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from scipy.stats import pearsonr
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
        return 0, 1.0
    
    # Convert to numpy arrays
    x_filtered, y_filtered = zip(*valid_data)
    x_array = np.array(x_filtered)
    y_array = np.array(y_filtered)
    
    # Check if all values in either array are the same (constant)
    if len(set(x_array)) <= 1 or len(set(y_array)) <= 1:
        return 0, 1.0  # No correlation if one array is constant
    
    # Need at least 2 data points for correlation
    if len(x_array) < 2:
        return 0, 1.0
    
    try:
        return pearsonr(x_array, y_array)
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 0, 1.0  # Default to no correlation on error

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

def plot_correlation_heatmap(df, features, title, filename):
    """Create correlation heatmap between delta and selected features."""
    # If no features or insufficient data, skip the plot
    if not features or len(df) < 2:
        print(f"Not enough data for correlation analysis in {filename}")
        return {}
    
    # Only include features that have enough valid data
    valid_features = []
    for feature in features:
        # Count valid data points (not None or NaN)
        valid_count = df[feature].count()
        if valid_count >= 2 and feature in df.columns:
            valid_features.append(feature)
    
    if not valid_features:
        print(f"No valid features with sufficient data found for {filename}")
        return {}
    
    # Calculate correlations manually to handle edge cases
    delta_corr = {}
    for feature in valid_features:
        corr, _ = safe_correlation(df['delta'], df[feature])
        delta_corr[feature] = corr
    
    if not delta_corr:
        print(f"No valid correlations found for {filename}")
        return {}
    
    # Convert to Series and sort
    delta_corr = pd.Series(delta_corr).sort_values(ascending=False)
    
    # Create a new figure
    plt.figure(figsize=(10, 8))
    
    # Plot horizontal bar chart
    bars = plt.barh(
        y=delta_corr.index,
        width=delta_corr.values,
        color=[plt.cm.RdBu(0.5 * (x + 1)) for x in delta_corr.values]
    )
    
    # Add correlation values as text
    for i, bar in enumerate(bars):
        plt.text(
            0.01 if delta_corr.values[i] < 0 else -0.01,
            bar.get_y() + bar.get_height()/2,
            f'{delta_corr.values[i]:.2f}',
            va='center',
            ha='left' if delta_corr.values[i] < 0 else 'right',
            color='black',
            fontweight='bold'
        )
    
    # Create nice feature labels
    feature_labels = [get_display_feature_name(feature) for feature in delta_corr.index]
    plt.yticks(range(len(feature_labels)), feature_labels)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Pearson Correlation Coefficient with Learning Gain (Delta Score)', fontsize=14)
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return delta_corr

def plot_individual_correlations(df, participant, feature, title_prefix, filename_prefix):
    """Create scatter plots showing the correlation between delta and a feature for one participant."""
    # Filter data for the specified participant
    participant_df = df[df['participant'] == participant].copy()
    
    # Check if we have valid data
    if participant_df.empty or feature not in participant_df.columns:
        print(f"No data available for participant {participant} and feature {feature}")
        return
    
    # Remove rows with None/NaN values for this feature, but keep zeros
    participant_df = participant_df.dropna(subset=[feature])
    
    # Check if we have enough data after filtering
    if len(participant_df) < 2:
        print(f"Not enough valid data for {participant} and feature {feature}")
        return
    
    # Check if we have variation in the data
    if len(set(participant_df[feature])) <= 1 or len(set(participant_df['delta'])) <= 1:
        print(f"Not enough variation in data for {participant} and feature {feature}")
        return
    
    # Calculate correlation safely
    correlation, p_value = safe_correlation(participant_df['delta'], participant_df[feature])
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Create colormap based on difficulty
    colors = {'Easy': 'green', 'Hard': 'purple'}
    markers = {'Easy': 'o', 'Hard': 's'}  # circles for Easy, squares for Hard
    
    # Plot points
    for difficulty in participant_df['difficulty'].unique():
        subset = participant_df[participant_df['difficulty'] == difficulty]
        plt.scatter(
            subset[feature], 
            subset['delta'],
            alpha=0.8,
            label=f"{difficulty} Difficulty",
            c=colors.get(difficulty, 'blue'),
            marker=markers.get(difficulty, 'o'),
            s=100,
            edgecolor='black',
            linewidth=0.5
        )
    
    # Add video names as annotations
    for i, row in participant_df.iterrows():
        video_name = os.path.basename(row['video']).split('.')[0]
        # Shorten very long video names
        if len(video_name) > 15:
            video_name = video_name[:12] + '...'
        
        plt.annotate(
            video_name,
            (row[feature], row['delta']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Add regression line if possible
    try:
        if len(participant_df) > 1 and len(set(participant_df[feature])) > 1:
            m, b = np.polyfit(participant_df[feature], participant_df['delta'], 1)
            x_range = np.linspace(participant_df[feature].min(), participant_df[feature].max(), 100)
            plt.plot(x_range, m * x_range + b, '--', color='red', alpha=0.7, 
                     label=f"Linear Regression (r={correlation:.2f})")
    except Exception as e:
        print(f"Could not fit regression line: {e}")
    
    # Format title and feature name for display
    display_feature = get_display_feature_name(feature)
    
    plt.title(f"Relationship Between {display_feature} and Learning Gain\nParticipant: {participant} (r={correlation:.2f}, p={p_value:.3f})", 
              fontsize=14, fontweight='bold')
    plt.xlabel(display_feature, fontsize=12)
    plt.ylabel('Learning Gain (Delta Score)', fontsize=12)
    plt.legend(title="Video Category", frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Add correlation statistics as text
    significance = "significant" if p_value < 0.05 else "not significant"
    plt.figtext(0.5, 0.01, 
                f"Pearson correlation: r={correlation:.2f}, p={p_value:.3f} ({significance})", 
                ha="center", fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.8, "pad":5, "edgecolor":"lightgray"})
    
    # Ensure enough space for title
    plt.tight_layout(pad=2.0)
    
    # Save figure
    safe_feature_name = feature.replace('.', '_').replace(' ', '_')
    plt.savefig(f"{filename_prefix}_{participant}_{safe_feature_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_participant_biomarkers(df, participant):
    """Analyze biomarkers for a specific participant."""
    participant_df = df[df['participant'] == participant]
    
    if participant_df.empty:
        print(f"No data available for participant {participant}")
        return
    
    if len(participant_df) < 2:
        print(f"Not enough data points for correlation analysis for participant {participant}")
        return
    
    # Emotion features
    emotion_features = [
        'diversity', 'non_neutral_percent', 'positive_percent', 'negative_percent'
    ]
    
    # Add specific emotion percentages if they exist in the data
    emotion_columns = [col for col in participant_df.columns if col.startswith('percentages.')]
    emotion_features.extend(emotion_columns)
    
    # Eye features
    eye_features = [
        'total_blinks', 'short_blinks', 'medium_blinks', 'long_blinks',
        'short_total', 'medium_total', 'long_total',
        'short_low_disp_ratio', 'medium_low_disp_ratio', 'long_low_disp_ratio',
        'small_ratio', 'medium_ratio', 'large_ratio'
    ]
    
    # Filter out features that don't exist in the dataframe
    valid_emotion_features = [f for f in emotion_features if f in participant_df.columns]
    valid_eye_features = [f for f in eye_features if f in participant_df.columns]
    
    # Create correlation plots
    emotion_corr = plot_correlation_heatmap(
        participant_df, 
        valid_emotion_features, 
        f"Correlation Between Emotional Biomarkers and Learning Gain\nParticipant: {participant}",
        f"emotion_correlation_{participant}.png"
    )
    
    eye_corr = plot_correlation_heatmap(
        participant_df, 
        valid_eye_features, 
        f"Correlation Between Eye-Tracking Biomarkers and Learning Gain\nParticipant: {participant}",
        f"eye_correlation_{participant}.png"
    )
    
    # Get top correlating features (positive or negative correlation)
    if emotion_corr is not None and not isinstance(emotion_corr, dict) and not emotion_corr.empty:
        top_emotion_features = emotion_corr.abs().sort_values(ascending=False).head(3).index.tolist()
        for feature in top_emotion_features:
            # Check if we have valid data for this feature
            if participant_df[feature].count() >= 2:
                plot_individual_correlations(
                    participant_df, 
                    participant, 
                    feature, 
                    "Emotional Biomarker", 
                    "emotion_scatter"
                )
    
    if eye_corr is not None and not isinstance(eye_corr, dict) and not eye_corr.empty:
        top_eye_features = eye_corr.abs().sort_values(ascending=False).head(3).index.tolist()
        for feature in top_eye_features:
            # Check if we have valid data for this feature
            if participant_df[feature].count() >= 2:
                plot_individual_correlations(
                    participant_df, 
                    participant, 
                    feature, 
                    "Eye-Tracking Biomarker", 
                    "eye_scatter"
                )

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
    
    # Create output directory for plots
    output_dir = 'learning_biomarker_analysis'
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Analyze each participant
    for participant in features_df['participant'].unique():
        print(f"Analyzing biomarkers for {participant}...")
        analyze_participant_biomarkers(features_df, participant)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 