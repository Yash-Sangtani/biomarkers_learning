import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import defaultdict
from scipy.stats import pearsonr
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set styling for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.dpi'] = 300  # High resolution for publication quality

def get_display_feature_name(feature):
    """Convert feature names to proper display format for research plots."""
    # First replace underscores with spaces and remove prefixes
    display_name = feature.replace('_', ' ').replace('percentages.', '')
    
    # Further refinements for specific features
    replacements = {
        'non_neutral_percent': 'Non-Neutral Emotion Percentage',
        'positive_percent': 'Positive Emotion Percentage',
        'negative_percent': 'Negative Emotion Percentage',
        'diversity': 'Emotion Diversity',
        'neutral_percent': 'Neutral Emotion Percentage',
        'happy_percent': 'Happy Emotion Percentage',
        'sad_percent': 'Sad Emotion Percentage',
        'surprise_percent': 'Surprise Emotion Percentage',
        'fear_percent': 'Fear Emotion Percentage',
        'disgust_percent': 'Disgust Emotion Percentage',
        'angry_percent': 'Angry Emotion Percentage',
        'short_blink_percent': 'Short Blink Percentage',
        'medium_blink_percent': 'Medium Blink Percentage',
        'long_blink_percent': 'Long Blink Percentage',
        'short_fixation_low_disp_percent': 'Short Fixation Low Dispersion Percentage',
        'short_fixation_high_disp_percent': 'Short Fixation High Dispersion Percentage',
        'medium_fixation_low_disp_percent': 'Medium Fixation Low Dispersion Percentage',
        'medium_fixation_high_disp_percent': 'Medium Fixation High Dispersion Percentage',
        'long_fixation_low_disp_percent': 'Long Fixation Low Dispersion Percentage',
        'long_fixation_high_disp_percent': 'Long Fixation High Dispersion Percentage',
        'small_pupil_percent': 'Small Pupil Diameter Percentage',
        'medium_pupil_percent': 'Medium Pupil Diameter Percentage',
        'large_pupil_percent': 'Large Pupil Diameter Percentage',
        'pct_change_': 'Change from Break: ',
        'diff_ecg_heart_rate': 'Heart Rate Difference (Video - Break)',
        'diff_ecg_sdnn': 'Heart Rate Variability (SDNN) Difference',
        'diff_ecg_rmssd': 'RMSSD Difference (Video - Break)',
        'diff_ecg_pnn50': 'PNN50 Difference (Video - Break)',
        'ecg_heart_rate': 'Heart Rate',
        'ecg_sdnn': 'Heart Rate Variability (SDNN)',
        'ecg_rmssd': 'RMSSD',
        'ecg_pnn50': 'PNN50'
    }
    
    # Apply replacements if the feature contains any of the keys
    for key, value in replacements.items():
        if key in feature:
            if key.startswith('pct_change_'):
                # For percentage change features, get the base feature name and prepend "Change from Break"
                base_feature = feature.replace('pct_change_', '')
                for k, v in replacements.items():
                    if k == base_feature:
                        return f"Change from Break: {v}"
                return f"Change from Break: {base_feature.replace('_', ' ').title()}"
            return value
    
    # If no specific replacement, return the general cleaned-up version
    return display_name.title()

def load_participant_data(participants_folder='participants'):
    """Load all participant JSON files from the participants folder using participant IDs."""
    participants_data = []
    
    # Loop through all JSON files in the participants folder
    for json_file in os.listdir(participants_folder):
        if not json_file.endswith('.json') or json_file.startswith('.'):
            continue
            
        file_path = os.path.join(participants_folder, json_file)
        
        with open(file_path, 'r') as f:
            participant_data = json.load(f)
            participants_data.extend(participant_data)
    
    return participants_data

def calculate_emotion_metrics(emotions_dict):
    """Calculate percentages for emotion metrics."""
    # Calculate total emotions displayed
    total_emotions = sum(emotions_dict.values())
    
    # Handle case where there are no emotions detected
    if total_emotions == 0:
        return {
            'emotion_diversity': 0,
            'non_neutral_percent': 0,
            'positive_percent': 0,
            'negative_percent': 0,
            'neutral_percent': 0,
            'happy_percent': 0,
            'sad_percent': 0,
            'surprise_percent': 0,
            'fear_percent': 0,
            'disgust_percent': 0,
            'angry_percent': 0
        }
    
    # Calculate emotion diversity (number of different emotions displayed)
    emotion_diversity = len(emotions_dict)
    
    # Calculate percentages for each emotion category (defaults to 0 if emotion not present)
    neutral_count = emotions_dict.get('Neutral', 0)
    neutral_percent = (neutral_count / total_emotions) * 100
    
    happy_count = emotions_dict.get('Happy', 0)
    happy_percent = (happy_count / total_emotions) * 100
    
    sad_count = emotions_dict.get('Sad', 0)
    sad_percent = (sad_count / total_emotions) * 100
    
    surprise_count = emotions_dict.get('Surprise', 0)
    surprise_percent = (surprise_count / total_emotions) * 100
    
    fear_count = emotions_dict.get('Fear', 0)
    fear_percent = (fear_count / total_emotions) * 100
    
    disgust_count = emotions_dict.get('Disgust', 0)
    disgust_percent = (disgust_count / total_emotions) * 100
    
    angry_count = emotions_dict.get('Angry', 0)
    angry_percent = (angry_count / total_emotions) * 100
    
    # Calculate non-neutral emotion percentage
    non_neutral_percent = 100 - neutral_percent
    
    # Calculate positive emotion percentage (Happy, Surprise)
    positive_percent = happy_percent + surprise_percent
    
    # Calculate negative emotion percentage (Sad, Fear, Disgust, Angry)
    negative_percent = sad_percent + fear_percent + disgust_percent + angry_percent
    
    return {
        'emotion_diversity': emotion_diversity,
        'non_neutral_percent': non_neutral_percent,
        'positive_percent': positive_percent,
        'negative_percent': negative_percent,
        'neutral_percent': neutral_percent,
        'happy_percent': happy_percent,
        'sad_percent': sad_percent,
        'surprise_percent': surprise_percent,
        'fear_percent': fear_percent,
        'disgust_percent': disgust_percent,
        'angry_percent': angry_percent
    }

def calculate_eye_metrics(metrics_data):
    """Calculate percentages for eye-tracking metrics."""
    # Extract data
    blinks = metrics_data.get('blinks', {})
    fixations = metrics_data.get('fixations', {})
    pupil_diameter = metrics_data.get('pupil_diameter', {})
    
    # Calculate blink percentages
    short_blinks = blinks.get('short', 0)
    medium_blinks = blinks.get('medium', 0)
    long_blinks = blinks.get('long', 0)
    total_blinks = short_blinks + medium_blinks + long_blinks
    
    # Handle case where there are no blinks detected
    if total_blinks == 0:
        short_blink_percent = 0
        medium_blink_percent = 0
        long_blink_percent = 0
    else:
        short_blink_percent = (short_blinks / total_blinks) * 100
        medium_blink_percent = (medium_blinks / total_blinks) * 100
        long_blink_percent = (long_blinks / total_blinks) * 100
    
    # Calculate fixation percentages
    short_fixation_low = fixations.get('short', {}).get('low_dispersion', 0)
    short_fixation_high = fixations.get('short', {}).get('high_dispersion', 0)
    medium_fixation_low = fixations.get('medium', {}).get('low_dispersion', 0)
    medium_fixation_high = fixations.get('medium', {}).get('high_dispersion', 0)
    long_fixation_low = fixations.get('long', {}).get('low_dispersion', 0)
    long_fixation_high = fixations.get('long', {}).get('high_dispersion', 0)
    
    # Calculate total fixations by duration
    total_short_fixations = short_fixation_low + short_fixation_high
    total_medium_fixations = medium_fixation_low + medium_fixation_high
    total_long_fixations = long_fixation_low + long_fixation_high
    total_fixations = total_short_fixations + total_medium_fixations + total_long_fixations
    
    # Handle case where there are no fixations detected
    if total_short_fixations == 0:
        short_fixation_low_disp_percent = 0
        short_fixation_high_disp_percent = 0
    else:
        short_fixation_low_disp_percent = (short_fixation_low / total_short_fixations) * 100
        short_fixation_high_disp_percent = (short_fixation_high / total_short_fixations) * 100
    
    if total_medium_fixations == 0:
        medium_fixation_low_disp_percent = 0
        medium_fixation_high_disp_percent = 0
    else:
        medium_fixation_low_disp_percent = (medium_fixation_low / total_medium_fixations) * 100
        medium_fixation_high_disp_percent = (medium_fixation_high / total_medium_fixations) * 100
    
    if total_long_fixations == 0:
        long_fixation_low_disp_percent = 0
        long_fixation_high_disp_percent = 0
    else:
        long_fixation_low_disp_percent = (long_fixation_low / total_long_fixations) * 100
        long_fixation_high_disp_percent = (long_fixation_high / total_long_fixations) * 100
    
    # Calculate pupil diameter percentages
    small_pupil = pupil_diameter.get('small', 0)
    medium_pupil = pupil_diameter.get('medium', 0)
    large_pupil = pupil_diameter.get('large', 0)
    total_pupil = small_pupil + medium_pupil + large_pupil
    
    # Handle case where there are no pupil diameter measurements
    if total_pupil == 0:
        small_pupil_percent = 0
        medium_pupil_percent = 0
        large_pupil_percent = 0
    else:
        small_pupil_percent = (small_pupil / total_pupil) * 100
        medium_pupil_percent = (medium_pupil / total_pupil) * 100
        large_pupil_percent = (large_pupil / total_pupil) * 100
    
    return {
        'total_blinks': total_blinks,
        'short_blink_percent': short_blink_percent,
        'medium_blink_percent': medium_blink_percent,
        'long_blink_percent': long_blink_percent,
        'short_fixation_low_disp_percent': short_fixation_low_disp_percent,
        'short_fixation_high_disp_percent': short_fixation_high_disp_percent,
        'medium_fixation_low_disp_percent': medium_fixation_low_disp_percent,
        'medium_fixation_high_disp_percent': medium_fixation_high_disp_percent,
        'long_fixation_low_disp_percent': long_fixation_low_disp_percent,
        'long_fixation_high_disp_percent': long_fixation_high_disp_percent,
        'small_pupil_percent': small_pupil_percent,
        'medium_pupil_percent': medium_pupil_percent,
        'large_pupil_percent': large_pupil_percent
    }

def calculate_percentage_change(video_metrics, break_metrics):
    """Calculate percentage change from break metrics to video metrics."""
    pct_change = {}
    
    # Process all metrics in video_metrics
    for key in video_metrics:
        video_val = video_metrics[key]
        break_val = break_metrics.get(key, 0)  # Default to 0 if metric not present in break
        
        # If break value is 0 but video has value, use the video value as 100% increase
        if break_val == 0 and video_val != 0:
            pct_change[f"pct_change_{key}"] = 100.0
        # If both are 0, no change
        elif break_val == 0 and video_val == 0:
            pct_change[f"pct_change_{key}"] = 0.0
        # Normal case - calculate percentage change
        else:
            pct_change[f"pct_change_{key}"] = ((video_val - break_val) / abs(break_val)) * 100
    
    return pct_change

def calculate_ecg_metrics_difference(video_ecg, break_ecg):
    """Calculate the difference between video and break ECG metrics."""
    diff_metrics = {}
    
    # If either video_ecg or break_ecg is None or empty, return empty dict
    if not video_ecg or not break_ecg:
        return diff_metrics
    
    # Calculate difference for each ECG metric
    for key in video_ecg:
        video_val = video_ecg.get(key, 0)
        break_val = break_ecg.get(key, 0)
        diff_metrics[f"diff_ecg_{key}"] = video_val - break_val
    
    return diff_metrics

def extract_features_with_break(participants_data):
    """Extract biomarker features from participant data, including break metrics."""
    all_features = []
    break_features = []
    pct_change_features = []
    
    for video_data in participants_data:
        participant_id = video_data.get('participant_id')
        video_name = video_data.get('video', '').split('/')[-1]
        difficulty = video_data.get('difficulty')
        delta = video_data.get('delta')
        break_metrics_data = video_data.get('break_metrics', {})
        
        if participant_id is not None and delta is not None:
            # Extract video metrics
            video_emotion_metrics = calculate_emotion_metrics(video_data.get('emotions', {}))
            video_eye_metrics = calculate_eye_metrics(video_data)
            video_ecg_metrics = video_data.get('ecg_metrics', {})
            
            # Extract break metrics
            break_emotion_metrics = calculate_emotion_metrics(break_metrics_data.get('emotions', {}))
            break_eye_metrics = calculate_eye_metrics(break_metrics_data)
            break_ecg_metrics = break_metrics_data.get('ecg_metrics', {})
            
            # Calculate percentage change from break to video
            pct_change_emotion = calculate_percentage_change(video_emotion_metrics, break_emotion_metrics)
            pct_change_eye = calculate_percentage_change(video_eye_metrics, break_eye_metrics)
            
            # Calculate ECG differences (not percentage)
            ecg_diff = calculate_ecg_metrics_difference(video_ecg_metrics, break_ecg_metrics)
            
            # Create feature dictionaries
            video_features = {
                'participant': participant_id,
                'video': video_name,
                'difficulty': difficulty,
                'delta': delta,
                **video_emotion_metrics,
                **video_eye_metrics,
                **{f'ecg_{k}': v for k, v in video_ecg_metrics.items()}  # Add ECG metrics with prefix
            }
            
            break_feature = {
                'participant': participant_id,
                'video': video_name,
                'difficulty': difficulty,
                'delta': 0,  # Break periods have no learning (delta = 0)
                **break_emotion_metrics,
                **break_eye_metrics,
                **{f'ecg_{k}': v for k, v in break_ecg_metrics.items()}  # Add ECG metrics with prefix
            }
            
            pct_change_feature = {
                'participant': participant_id,
                'video': video_name,
                'difficulty': difficulty,
                'delta': delta,
                **pct_change_emotion,
                **pct_change_eye,
                **ecg_diff  # Add ECG differences
            }
            
            all_features.append(video_features)
            break_features.append(break_feature)
            pct_change_features.append(pct_change_feature)
    
    return pd.DataFrame(all_features), pd.DataFrame(break_features), pd.DataFrame(pct_change_features)

def safe_correlation(x, y):
    """Calculate correlation safely handling edge cases and missing data."""
    # Filter out None/NaN values while keeping legitimate zeros
    valid_data = [(x_val, y_val) for x_val, y_val in zip(x, y) 
                 if x_val is not None and y_val is not None 
                 and not (pd.isna(x_val) or pd.isna(y_val))]
    
    # If no valid data pairs remain, return no correlation and no p-value
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
        # Calculate Pearson correlation and p-value
        correlation, p_value = pearsonr(x_array, y_array)
        return correlation, p_value
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 0, 1.0  # Default to no correlation on error

def calculate_correlations_with_delta(df, difficulty_filter=None):
    """Calculate correlations between features and delta for all participants."""
    # Apply difficulty filter if specified
    if difficulty_filter is not None:
        df = df[df['difficulty'] == difficulty_filter]
    
    participants = df['participant'].unique()
    feature_columns = [col for col in df.columns if col not in ['participant', 'video', 'difficulty', 'delta']]
    
    participant_correlations = {}
    participant_p_values = {}
    
    for participant in participants:
        participant_df = df[df['participant'] == participant]
        
        # Debug print
        print(f"Processing participant {participant}:")
        print(f"Number of data points: {len(participant_df)}")
        
        feature_correlations = {}
        feature_p_values = {}
        
        for feature in feature_columns:
            # Skip features that are all NaN
            if participant_df[feature].isna().all():
                continue
                
            corr, p_val = safe_correlation(participant_df[feature].values, 
                                         participant_df['delta'].values)
            feature_correlations[feature] = corr
            feature_p_values[feature] = p_val
        
        # Always store correlations even if they're all zero
        participant_correlations[participant] = feature_correlations
        participant_p_values[participant] = feature_p_values
    
    return participant_correlations, participant_p_values

def create_bar_plot(correlations, p_values, participant_id, features, title, output_path, ecg_keys=None):
    """Create a bar plot showing percentage change correlations or ECG difference correlations."""
    # Get correlations for the specific participant
    if participant_id not in correlations:
        print(f"No correlation data for participant {participant_id}")
        return
    
    corr = correlations[participant_id]
    p_vals = p_values[participant_id]
    
    # Filter features
    valid_features = []
    valid_correlations = []
    significant_markers = []
    
    if ecg_keys:
        # For ECG metrics, use the provided keys directly
        for key in ecg_keys:
            if key in corr:
                valid_features.append(key)
                valid_correlations.append(corr[key])
                # Mark if correlation is statistically significant (p < 0.05)
                significant_markers.append(p_vals[key] < 0.05)
    else:
        # For regular biomarkers, use pct_change_ prefix
        for feature in features:
            pct_change_key = f"pct_change_{feature}"
            if pct_change_key in corr:
                valid_features.append(feature)
                valid_correlations.append(corr[pct_change_key])
                # Mark if correlation is statistically significant (p < 0.05)
                significant_markers.append(p_vals[pct_change_key] < 0.05)
    
    # Skip if no valid features
    if not valid_features:
        print(f"No valid features for participant {participant_id}")
        return
    
    # Set up the bar plot
    plt.figure(figsize=(12, 8))
    y_pos = range(len(valid_features))
    
    # Create bars
    bars = plt.barh(y_pos, valid_correlations, align='center', color='#2E86C1', alpha=0.7)
    
    # Add significance markers
    for i, bar in enumerate(bars):
        if significant_markers[i]:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, '*', va='center', ha='left', color='#E74C3C', fontsize=12)
    
    # Add feature labels
    if ecg_keys:
        plt.yticks(y_pos, [get_display_feature_name(f) for f in valid_features])
    else:
        plt.yticks(y_pos, [get_display_feature_name(f) for f in valid_features])
    
    # Add labels and title
    plt.xlabel('Correlation with Learning Gain')
    plt.title(title)
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def save_biomarker_data_to_csv(participant_id, video_metrics, break_metrics, output_path):
    """Save biomarker data to CSV for each participant."""
    # Prepare data for CSV
    data = []
    
    # Add emotion and eye metrics
    for feature in video_metrics.keys():
        if not feature.startswith(('participant', 'video', 'difficulty', 'delta')):
            # Skip non-metric columns
            data.append({
                'Biomarker': get_display_feature_name(feature),
                'Video %': video_metrics[feature],
                'Break %': break_metrics.get(feature, 0)
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def main():
    """Main function to generate all required plots and CSVs."""
    print("Loading participant data...")
    participants_data = load_participant_data('participant_info')
    
    print("Extracting features...")
    df_video, df_break, df_pct_change = extract_features_with_break(participants_data)
    
    # Print debug information
    print("\nParticipant data summary:")
    for participant in df_video['participant'].unique():
        print(f"\nParticipant {participant}:")
        print(f"Video data points: {len(df_video[df_video['participant'] == participant])}")
        print(f"Break data points: {len(df_break[df_break['participant'] == participant])}")
        print(f"Percentage change data points: {len(df_pct_change[df_pct_change['participant'] == participant])}")
    
    # Define difficulty filters
    difficulty_filters = [None, "Easy", "Hard"]
    filter_names = ["all_videos", "easy_videos", "hard_videos"]
    
    # Create output directory
    output_dir = 'plots_new/break_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create directories for each participant
    participants = df_video['participant'].unique()
    for participant in participants:
        participant_dir = os.path.join(output_dir, f"participant_{participant}")
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir)
    
    # Define feature groups
    emotion_features = [
        'emotion_diversity', 'non_neutral_percent', 'positive_percent', 'negative_percent',
        'neutral_percent', 'happy_percent', 'sad_percent', 'surprise_percent',
        'fear_percent', 'disgust_percent', 'angry_percent'
    ]
    
    eye_features = [
        'short_blink_percent', 'medium_blink_percent', 'long_blink_percent',
        'short_fixation_low_disp_percent', 'short_fixation_high_disp_percent',
        'medium_fixation_low_disp_percent', 'medium_fixation_high_disp_percent',
        'long_fixation_low_disp_percent', 'long_fixation_high_disp_percent',
        'small_pupil_percent', 'medium_pupil_percent', 'large_pupil_percent'
    ]
    
    # Add ECG features
    ecg_features = ['heart_rate', 'sdnn', 'rmssd', 'pnn50']
    
    all_features = emotion_features + eye_features
    
    # Generate plots for each difficulty filter
    for i, difficulty_filter in enumerate(difficulty_filters):
        print(f"\nProcessing {filter_names[i]}...")
        
        # Calculate correlations for both percentage change and ECG differences
        correlations, p_values = calculate_correlations_with_delta(df_pct_change, difficulty_filter)
        
        print(f"Creating bar plots for {filter_names[i]}...")
        
        # Create individual participant plots
        for participant in participants:
            difficulty_text = f" ({difficulty_filter} Videos)" if difficulty_filter else " (All Videos)"
            
            # Create plots for each feature group
            create_bar_plot(
                correlations, p_values,
                participant, emotion_features,
                f"Emotion Biomarkers % Change Correlation with Learning Gain\nParticipant {participant}{difficulty_text}",
                os.path.join(output_dir, f"participant_{participant}", f"emotion_bar_plot_{filter_names[i]}.png")
            )
            
            create_bar_plot(
                correlations, p_values,
                participant, eye_features,
                f"Eye-Tracking Biomarkers % Change Correlation with Learning Gain\nParticipant {participant}{difficulty_text}",
                os.path.join(output_dir, f"participant_{participant}", f"eye_tracking_bar_plot_{filter_names[i]}.png")
            )
            
            # Create plot for ECG features (difference, not percentage)
            ecg_keys = [f'diff_ecg_{key}' for key in ecg_features]
            create_bar_plot(
                correlations, p_values,
                participant, [],  # Empty list as we're using the ecg_keys directly
                f"ECG Metrics Difference Correlation with Learning Gain\nParticipant {participant}{difficulty_text}",
                os.path.join(output_dir, f"participant_{participant}", f"ecg_bar_plot_{filter_names[i]}.png"),
                ecg_keys  # Pass the ECG keys as a separate parameter
            )
            
            create_bar_plot(
                correlations, p_values,
                participant, all_features,
                f"All Biomarkers % Change Correlation with Learning Gain\nParticipant {participant}{difficulty_text}",
                os.path.join(output_dir, f"participant_{participant}", f"all_biomarkers_bar_plot_{filter_names[i]}.png")
            )
            
            # Save biomarker data to CSV
            video_metrics = df_video[df_video['participant'] == participant].iloc[0].to_dict()
            break_metrics = df_break[df_break['participant'] == participant].iloc[0].to_dict()
            save_biomarker_data_to_csv(
                participant,
                video_metrics,
                break_metrics,
                os.path.join(output_dir, f"participant_{participant}", f"biomarker_data_{filter_names[i]}.csv")
            )
    
    print("\nAll plots and analyses completed successfully!")

if __name__ == "__main__":
    main() 