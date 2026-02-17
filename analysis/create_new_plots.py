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
        'large_pupil_percent': 'Large Pupil Diameter Percentage'
    }
    
    # Apply replacements if the feature is in our dictionary
    for key, value in replacements.items():
        if key in feature:
            return value
    
    # If no specific replacement, return the general cleaned-up version
    return display_name.title()

def load_participant_data(results_folder='results'):
    """Load all participant JSON files from the results folder using participant IDs."""
    participants_data = []
    
    # Loop through all directories in the results folder
    for folder_name in os.listdir(results_folder):
        folder_path = os.path.join(results_folder, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
            
        # Look for JSON files in the participant folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            
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

def calculate_eye_metrics(video_data):
    """Calculate percentages for eye-tracking metrics."""
    # Extract data
    blinks = video_data.get('blinks', {})
    fixations = video_data.get('fixations', {})
    pupil_diameter = video_data.get('pupil_diameter', {})
    
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

def extract_features(participants_data):
    """Extract biomarker features from participant data."""
    all_features = []
    
    for video_data in participants_data:
        participant_id = video_data.get('participant_id')
        video_name = video_data.get('video', '').split('/')[-1]
        difficulty = video_data.get('difficulty')
        delta = video_data.get('delta')
        
        if participant_id is not None and delta is not None:
            # Extract emotion metrics
            emotion_metrics = calculate_emotion_metrics(video_data.get('emotions', {}))
            
            # Extract eye-tracking metrics
            eye_metrics = calculate_eye_metrics(video_data)
            
            # Create feature dictionary
            features = {
                'participant': participant_id,
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

def calculate_participant_correlations(df):
    """Calculate correlations between features and delta for each participant."""
    participants = df['participant'].unique()
    all_features = df.columns.drop(['participant', 'video', 'difficulty', 'delta'])
    
    correlations = {}
    p_values = {}
    
    for participant in participants:
        participant_df = df[df['participant'] == participant]
        
        # Skip if not enough data points
        if len(participant_df) < 2:
            continue
        
        participant_corr = {}
        participant_p_values = {}
        
        for feature in all_features:
            corr, p_val = safe_correlation(participant_df[feature].values, 
                                     participant_df['delta'].values)
            participant_corr[feature] = corr
            participant_p_values[feature] = p_val
        
        correlations[participant] = participant_corr
        p_values[participant] = participant_p_values
    
    return correlations, p_values

def create_radar_plot_for_participant(df, participant_id, features, title, output_path, difficulty_filter=None):
    """Create a radar plot for a single participant showing correlations for selected features."""
    # Filter data for the participant
    participant_df = df[df['participant'] == participant_id]
    
    # Apply difficulty filter if specified
    if difficulty_filter is not None:
        participant_df = participant_df[participant_df['difficulty'] == difficulty_filter]
    
    # Skip if not enough data points
    if len(participant_df) < 2:
        print(f"Not enough data points for participant {participant_id} with difficulty {difficulty_filter}")
        return
    
    # Calculate correlations for each feature
    feature_correlations = {}
    for feature in features:
        corr, _ = safe_correlation(participant_df[feature].values, 
                             participant_df['delta'].values)
        feature_correlations[feature] = corr
    
    # Set up the radar plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(features)
    
    # Angles for each feature (equally spaced around the circle)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Feature names for display
    feature_names = [get_display_feature_name(feature) for feature in features]
    
    # Correlation values (add first value at the end to close the loop)
    values = [feature_correlations[feature] for feature in features]
    values += values[:1]
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, alpha=0.25)
    
    # Set y-axis limits to be symmetric
    ax.set_ylim(-1, 1)
    
    # Add feature labels
    plt.xticks(angles[:-1], feature_names, size=12)
    
    # Adjust label positions for better readability
    for i, label in enumerate(ax.get_xticklabels()):
        angle_rad = angles[i]
        
        # Adjust horizontal alignment based on angle
        if -0.5 * np.pi < angle_rad <= 0.5 * np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        
        # Rotate labels for better readability
        label.set_rotation(np.degrees(angle_rad))
    
    # Add radial grid lines at 0, 0.5, and -0.5 correlation values
    ax.set_rticks([-1, -0.5, 0, 0.5, 1])
    ax.set_rlabel_position(0)  # Move radial labels to a better position
    
    # Add a subtitle with specifics
    difficulty_text = f"Difficulty: {difficulty_filter}" if difficulty_filter else "All Videos"
    fig.suptitle(f"{title}\n{difficulty_text}", fontsize=16, y=0.98)
    
    # Add a reference line at zero correlation
    ax.plot(angles, [0] * len(angles), '--', color='gray', alpha=0.75, linewidth=1)
    
    # Show correlation strength through circular grid
    grid_values = [-0.5, 0.5]
    for grid_val in grid_values:
        ax.plot(angles, [grid_val] * len(angles), '--', color='gray', alpha=0.5, linewidth=0.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_delta_bar_chart(df, participant_id, output_path):
    """Create a bar chart showing delta scores for each video by difficulty."""
    # Filter data for the participant
    participant_df = df[df['participant'] == participant_id]
    
    # Skip if no data
    if len(participant_df) == 0:
        print(f"No data for participant {participant_id}")
        return
    
    # Sort by difficulty and then by video name
    participant_df = participant_df.sort_values(by=['difficulty', 'video'])
    
    # Set up colors based on difficulty
    colors = {'Easy': '#4CAF50', 'Hard': '#F44336'}
    
    # Create the bar chart
    plt.figure(figsize=(12, 8))
    
    # Create bars with different colors based on difficulty
    bars = plt.bar(participant_df['video'], participant_df['delta'], 
                  color=[colors[diff] for diff in participant_df['difficulty']])
    
    # Add labels and title
    plt.xlabel('Video', fontsize=14)
    plt.ylabel('Learning Gain (Delta Score)', fontsize=14)
    plt.title(f'Learning Gain by Video for Participant {participant_id}', fontsize=16, pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add difficulty legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['Easy'], label='Easy'),
        mpatches.Patch(facecolor=colors['Hard'], label='Hard')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            y_pos = height + 0.1
        else:
            y_pos = height - 0.3
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height}', ha='center', va='bottom', fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_radar_plot_by_biomarker(correlations, feature, output_path):
    """Create a radar plot showing correlations for a specific biomarker across all participants."""
    # Get all participant IDs
    participants = list(correlations.keys())
    
    # Skip if no participants
    if not participants:
        print(f"No participant data for feature {feature}")
        return
    
    # Get correlation values for this feature
    values = [correlations[p].get(feature, 0) for p in participants]
    
    # Add first value at the end to close the loop
    values += values[:1]
    participants_display = participants + participants[:1]
    
    # Set up the radar plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(participants)
    
    # Angles for each participant (equally spaced around the circle)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, alpha=0.25)
    
    # Set y-axis limits to be symmetric
    ax.set_ylim(-1, 1)
    
    # Add participant ID labels
    plt.xticks(angles[:-1], [f"Participant {p}" for p in participants], size=12)
    
    # Adjust label positions for better readability
    for i, label in enumerate(ax.get_xticklabels()):
        angle_rad = angles[i]
        
        # Adjust horizontal alignment based on angle
        if -0.5 * np.pi < angle_rad <= 0.5 * np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        
        # Rotate labels for better readability
        label.set_rotation(np.degrees(angle_rad))
    
    # Add radial grid lines at 0, 0.5, and -0.5 correlation values
    ax.set_rticks([-1, -0.5, 0, 0.5, 1])
    ax.set_rlabel_position(0)  # Move radial labels to a better position
    
    # Add a title
    fig.suptitle(f"Correlation Between {get_display_feature_name(feature)} and Learning Gain\nAcross All Participants", 
                fontsize=16, y=0.98)
    
    # Add a reference line at zero correlation
    ax.plot(angles, [0] * len(angles), '--', color='gray', alpha=0.75, linewidth=1)
    
    # Show correlation strength through circular grid
    grid_values = [-0.5, 0.5]
    for grid_val in grid_values:
        ax.plot(angles, [grid_val] * len(angles), '--', color='gray', alpha=0.5, linewidth=0.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_correlation_heatmap(correlations, output_path):
    """Create a heatmap showing correlations between biomarkers and learning gain for all participants."""
    # Get all participants and features
    participants = list(correlations.keys())
    
    # Skip if no participants
    if not participants:
        print("No participant data for correlation heatmap")
        return
    
    # Get all features from the first participant
    features = list(correlations[participants[0]].keys())
    
    # Create a DataFrame for the heatmap
    heatmap_data = []
    for feature in features:
        row = {'Feature': get_display_feature_name(feature)}
        for participant in participants:
            row[f"Participant {participant}"] = correlations[participant].get(feature, 0)
        heatmap_data.append(row)
    
    df_heatmap = pd.DataFrame(heatmap_data)
    df_heatmap.set_index('Feature', inplace=True)
    
    # Create the heatmap
    plt.figure(figsize=(16, 14))
    
    # Create a custom diverging colormap centered at 0
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Create the heatmap
    sns.heatmap(df_heatmap, annot=True, cmap=cmap, center=0, 
               vmin=-1, vmax=1, square=True, linewidths=.5, fmt=".2f")
    
    # Add title and labels
    plt.title('Correlation Between Biomarkers and Learning Gain Across All Participants', fontsize=16, pad=20)
    plt.ylabel('Biomarker', fontsize=14)
    plt.xlabel('Participant ID', fontsize=14)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def save_p_values(p_values, output_path):
    """Save p-values to a CSV file."""
    # Get all participants and features
    participants = list(p_values.keys())
    
    # Skip if no participants
    if not participants:
        print("No participant data for p-values")
        return
    
    # Get all features from the first participant
    features = list(p_values[participants[0]].keys())
    
    # Create a DataFrame for the p-values
    p_value_data = []
    for feature in features:
        row = {'Feature': get_display_feature_name(feature)}
        for participant in participants:
            row[f"Participant {participant}"] = p_values[participant].get(feature, 1.0)
        p_value_data.append(row)
    
    df_p_values = pd.DataFrame(p_value_data)
    
    # Save to CSV
    df_p_values.to_csv(output_path, index=False)
    print(f"P-values saved to {output_path}")

def analyze_break_data(results_folder='results'):
    """Analyze break data from timestamped_events.csv files."""
    break_data = {}
    
    # Loop through all participant folders
    for folder_name in os.listdir(results_folder):
        folder_path = os.path.join(results_folder, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Look for timestamped_events.csv
        csv_path = os.path.join(folder_path, 'timestamped_events.csv')
        if not os.path.exists(csv_path):
            continue
        
        # Read CSV file
        events_df = pd.read_csv(csv_path)
        
        # Extract participant ID
        if 'participant_id' in events_df.columns and len(events_df) > 0:
            participant_id = events_df['participant_id'].iloc[0]
        else:
            continue
        
        # Find break events
        break_starts = events_df[events_df['event_type'] == 'Break Start']
        break_ends = events_df[events_df['event_type'] == 'Break End']
        
        # Skip if no break events
        if len(break_starts) == 0 or len(break_ends) == 0:
            continue
        
        # Combine break starts and ends
        breaks = []
        for i in range(min(len(break_starts), len(break_ends))):
            start_time = break_starts.iloc[i]['timestamp']
            end_time = break_ends.iloc[i]['timestamp']
            duration = end_time - start_time
            breaks.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
        
        break_data[participant_id] = breaks
    
    return break_data

def main():
    """Main function to generate all required plots."""
    print("Loading participant data...")
    participants_data = load_participant_data()
    
    print("Extracting features...")
    df = extract_features(participants_data)
    
    print("Calculating correlations...")
    correlations, p_values = calculate_participant_correlations(df)
    
    # Create a new plots_new directory if it doesn't exist
    if not os.path.exists('plots_new'):
        os.makedirs('plots_new')
    
    # Create directories for each participant
    participants = df['participant'].unique()
    for participant in participants:
        participant_dir = os.path.join('plots_new', f"participant_{participant}")
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir)
    
    # List of emotional biomarkers
    emotion_features = [
        'emotion_diversity', 'non_neutral_percent', 'positive_percent', 'negative_percent',
        'neutral_percent', 'happy_percent', 'sad_percent', 'surprise_percent',
        'fear_percent', 'disgust_percent', 'angry_percent'
    ]
    
    # List of eye-tracking biomarkers
    eye_features = [
        'short_blink_percent', 'medium_blink_percent', 'long_blink_percent',
        'short_fixation_low_disp_percent', 'short_fixation_high_disp_percent',
        'medium_fixation_low_disp_percent', 'medium_fixation_high_disp_percent',
        'long_fixation_low_disp_percent', 'long_fixation_high_disp_percent',
        'small_pupil_percent', 'medium_pupil_percent', 'large_pupil_percent'
    ]
    
    # All biomarkers combined
    all_features = emotion_features + eye_features
    
    print("Creating plots for each participant...")
    for participant in participants:
        print(f"Generating plots for Participant {participant}...")
        participant_dir = os.path.join('plots_new', f"participant_{participant}")
        
        # Create radar plot for all videos
        create_radar_plot_for_participant(
            df, participant, all_features,
            f"Correlation Between Biomarkers and Learning Gain for Participant {participant}",
            os.path.join(participant_dir, f"radar_plot_all_videos.png")
        )
        
        # Create radar plot for easy videos
        create_radar_plot_for_participant(
            df, participant, all_features,
            f"Correlation Between Biomarkers and Learning Gain for Participant {participant}",
            os.path.join(participant_dir, f"radar_plot_easy_videos.png"),
            difficulty_filter="Easy"
        )
        
        # Create radar plot for hard videos
        create_radar_plot_for_participant(
            df, participant, all_features,
            f"Correlation Between Biomarkers and Learning Gain for Participant {participant}",
            os.path.join(participant_dir, f"radar_plot_hard_videos.png"),
            difficulty_filter="Hard"
        )
        
        # Create delta bar chart
        create_delta_bar_chart(
            df, participant,
            os.path.join(participant_dir, f"delta_bar_chart.png")
        )
    
    print("Creating biomarker-specific radar plots...")
    # Create a directory for biomarker plots if it doesn't exist
    biomarker_dir = os.path.join('plots_new', 'biomarkers')
    if not os.path.exists(biomarker_dir):
        os.makedirs(biomarker_dir)
    
    # Create radar plots for each biomarker
    for feature in all_features:
        create_radar_plot_by_biomarker(
            correlations, feature,
            os.path.join(biomarker_dir, f"radar_plot_{feature}.png")
        )
    
    print("Creating correlation heatmap...")
    # Create correlation heatmap
    create_correlation_heatmap(
        correlations,
        os.path.join('plots_new', 'correlation_heatmap.png')
    )
    
    print("Saving p-values...")
    # Save p-values
    save_p_values(
        p_values,
        os.path.join('plots_new', 'p_values.csv')
    )
    
    print("Analyzing break data...")
    # Analyze break data
    break_data = analyze_break_data()
    
    # Save break data analysis results
    with open(os.path.join('plots_new', 'break_data_analysis.txt'), 'w') as f:
        f.write("Break Data Analysis\n")
        f.write("=================\n\n")
        f.write("This analysis examines the break periods between videos to potentially establish baseline/resting readings.\n\n")
        
        for participant_id, breaks in break_data.items():
            f.write(f"Participant {participant_id}:\n")
            f.write(f"  Number of breaks: {len(breaks)}\n")
            
            if breaks:
                durations = [b['duration'] for b in breaks]
                avg_duration = sum(durations) / len(durations)
                f.write(f"  Average break duration: {avg_duration:.2f} seconds\n")
                
                for i, b in enumerate(breaks):
                    f.write(f"  Break {i+1}: {b['duration']:.2f} seconds\n")
            
            f.write("\n")
        
        f.write("\nRecommendations for Incorporating Break Data:\n")
        f.write("1. Break periods can be used to establish baseline/resting physiological readings for each participant.\n")
        f.write("2. Compare emotion and eye-tracking metrics during videos against these baseline readings.\n")
        f.write("3. Calculate deviation from baseline for each metric to normalize individual differences.\n")
        f.write("4. Create additional visualizations comparing video engagement to resting state.\n")
    
    print("All plots and analyses completed successfully!")

if __name__ == "__main__":
    main() 