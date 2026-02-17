import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Dictionary mapping names to participant IDs
PARTICIPANT_NAME_TO_ID = {
    'JD': 1,
    'Navya': 2,
    'Dhruv': 2,
    'Rahil': 2,
    'Siddhanth': 2,
    'Krish': 2,
    'Shikhar': 5,
    'Aashana': 4,
    'Aarav': 7  # Assuming Aarav is participant_id 7, adjust if needed
}

# Dictionary mapping participant IDs to names for lookup
PARTICIPANT_ID_TO_NAME = {v: k for k, v in PARTICIPANT_NAME_TO_ID.items()}

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

def load_participant_data(results_folder='results'):
    """Load all participant JSON files from the results directory structure."""
    participants_data = {}
    
    for participant_folder in os.listdir(results_folder):
        participant_folder_path = os.path.join(results_folder, participant_folder)
        
        # Skip non-directories or hidden folders
        if not os.path.isdir(participant_folder_path) or participant_folder.startswith('.'):
            continue
        
        json_file_path = os.path.join(participant_folder_path, f'{participant_folder}.json')
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as file:
                    participant_data = json.load(file)
                    # Store by participant ID, not name
                    for item in participant_data:
                        participant_id = item.get('participant_id')
                        if participant_id is not None:
                            if participant_id not in participants_data:
                                participants_data[participant_id] = []
                            participants_data[participant_id].append(item)
            except Exception as e:
                print(f"Error loading {json_file_path}: {e}")
    
    return participants_data

def calculate_emotion_metrics(emotions_dict):
    """Calculate various emotion metrics from the emotions dictionary, using percentages."""
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
    """Calculate eye-related metrics from the video data, focusing on percentages."""
    # Check if the components are present
    blinks = video_data.get('blinks', {})
    fixations = video_data.get('fixations', {})
    pupil_diameter = video_data.get('pupil_diameter', {})
    
    # Initialize results dictionary with None values to represent missing data
    result = {
        'total_blinks': None,
        'short_blinks_percent': None,
        'medium_blinks_percent': None,
        'long_blinks_percent': None
    }
    
    # Only calculate blink metrics if blink data exists
    if blinks:
        total_blinks = sum(blinks.values())
        result['total_blinks'] = total_blinks
        
        # Calculate blink percentages if total is not zero
        if total_blinks > 0:
            result['short_blinks_percent'] = blinks.get('short', 0) / total_blinks * 100
            result['medium_blinks_percent'] = blinks.get('medium', 0) / total_blinks * 100
            result['long_blinks_percent'] = blinks.get('long', 0) / total_blinks * 100
    
    # Calculate fixation metrics if data exists
    if fixations:
        for duration in ['short', 'medium', 'long']:
            if duration in fixations:
                low_disp = fixations[duration].get('low_dispersion', 0)
                high_disp = fixations[duration].get('high_dispersion', 0)
                total = low_disp + high_disp
                
                # Store total count
                result[f'{duration}_total'] = total
                
                # Store percentages
                if total > 0:
                    result[f'{duration}_low_disp_percent'] = low_disp / total * 100
                    result[f'{duration}_high_disp_percent'] = high_disp / total * 100
                else:
                    result[f'{duration}_low_disp_percent'] = None
                    result[f'{duration}_high_disp_percent'] = None
            else:
                result[f'{duration}_total'] = None
                result[f'{duration}_low_disp_percent'] = None
                result[f'{duration}_high_disp_percent'] = None
    
    # Calculate pupil diameter metrics if data exists
    if pupil_diameter:
        total_pupil_readings = sum(pupil_diameter.values())
        if total_pupil_readings > 0:
            result['small_pupil_percent'] = pupil_diameter.get('small', 0) / total_pupil_readings * 100
            result['medium_pupil_percent'] = pupil_diameter.get('medium', 0) / total_pupil_readings * 100
            result['large_pupil_percent'] = pupil_diameter.get('large', 0) / total_pupil_readings * 100
        else:
            result['small_pupil_percent'] = None
            result['medium_pupil_percent'] = None
            result['large_pupil_percent'] = None
    else:
        result['small_pupil_percent'] = None
        result['medium_pupil_percent'] = None
        result['large_pupil_percent'] = None
    
    return result

def extract_features(participants_data):
    """Extract features from all participants data for analysis, using percentages."""
    all_features = []
    
    for participant_id, videos_data in participants_data.items():
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
                'participant_id': participant_id,
                'participant_name': PARTICIPANT_ID_TO_NAME.get(participant_id, f"Participant {participant_id}"),
                'video': video_name,
                'difficulty': difficulty,
                'delta': delta,
                **emotion_metrics,
                **eye_metrics
            }
            
            all_features.append(features)
    
    return pd.DataFrame(all_features)

def safe_correlation(x, y):
    """Calculate correlation safely handling edge cases and missing data. Returns correlation and p-value."""
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

def create_radar_plot_for_participant(df, participant_id, features, title, output_path):
    """Create a radar plot for a single participant showing multiple biomarkers."""
    # Filter data for this participant
    participant_df = df[df['participant_id'] == participant_id]
    
    if len(participant_df) == 0:
        print(f"No data found for participant ID {participant_id}")
        return
    
    # Filter features that have valid data
    valid_features = []
    feature_values = {}
    
    for feature in features:
        if feature in participant_df.columns:
            # Calculate correlation with delta
            corr, p_value = safe_correlation(participant_df['delta'], participant_df[feature])
            if not np.isnan(corr):
                valid_features.append(feature)
                feature_values[feature] = {
                    'correlation': corr,
                    'p_value': p_value
                }
    
    if not valid_features:
        print(f"No valid features found for participant ID {participant_id}")
        return
    
    # Set up the radar plot
    angles = np.linspace(0, 2*np.pi, len(valid_features), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Get correlation values (absolute for better visualization)
    values = [abs(feature_values[feat]['correlation']) for feat in valid_features]
    values += values[:1]  # Close the circle
    
    # Plot the correlations
    ax.plot(angles, values, 'o-', linewidth=2, label='Absolute Correlation')
    ax.fill(angles, values, alpha=0.25)
    
    # Add p-value indicators
    for i, feature in enumerate(valid_features):
        p_value = feature_values[feature]['p_value']
        if p_value < 0.05:
            ax.plot(angles[i], abs(feature_values[feature]['correlation']), 'r*', ms=10, 
                   label='p < 0.05' if i == 0 else "")
    
    # Add direction indicators (positive/negative)
    for i, feature in enumerate(valid_features):
        corr = feature_values[feature]['correlation']
        direction = '+' if corr >= 0 else '-'
        ax.text(angles[i], abs(corr) + 0.1, direction, ha='center', va='center', 
               fontweight='bold', color='green' if corr >= 0 else 'red')
    
    # Set labels and style
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([get_display_feature_name(feat) for feat in valid_features], fontsize=8)
    ax.set_yticklabels([])  # Remove radial labels
    
    # Add grid lines
    ax.grid(True)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    participant_name = PARTICIPANT_ID_TO_NAME.get(participant_id, f"Participant {participant_id}")
    plt.title(f"{title} - Participant ID: {participant_id} ({participant_name})", pad=20, fontweight='bold')
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar plot created and saved to {output_path}")
    
    return feature_values

def create_delta_bar_chart(df, participant_id, output_path):
    """Create a bar chart showing delta scores for each video by difficulty."""
    # Filter data for this participant
    participant_df = df[df['participant_id'] == participant_id]
    
    if len(participant_df) == 0:
        print(f"No data found for participant ID {participant_id}")
        return
    
    # Group by difficulty
    easy_videos = participant_df[participant_df['difficulty'] == 'Easy']
    hard_videos = participant_df[participant_df['difficulty'] == 'Hard']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot easy videos
    if not easy_videos.empty:
        x_pos = np.arange(len(easy_videos))
        ax.bar(x_pos, easy_videos['delta'], width=0.4, label='Easy Videos', color='skyblue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([os.path.splitext(os.path.basename(v))[0] for v in easy_videos['video']], rotation=45, ha='right')
    
    # Plot hard videos
    if not hard_videos.empty:
        x_pos = np.arange(len(easy_videos), len(easy_videos) + len(hard_videos))
        ax.bar(x_pos, hard_videos['delta'], width=0.4, label='Hard Videos', color='salmon')
        ax.set_xticks(list(ax.get_xticks()) + list(x_pos))
        ax.set_xticklabels(list(ax.get_xticklabels()) + 
                           [os.path.splitext(os.path.basename(v))[0] for v in hard_videos['video']], 
                           rotation=45, ha='right')
    
    participant_name = PARTICIPANT_ID_TO_NAME.get(participant_id, f"Participant {participant_id}")
    ax.set_title(f"Learning Gain by Video Difficulty - Participant ID: {participant_id} ({participant_name})", 
                fontweight='bold')
    ax.set_ylabel("Learning Gain (Delta Score)")
    ax.set_xlabel("Video")
    ax.legend()
    
    # Add value labels on top of each bar
    for i, v in enumerate(list(easy_videos['delta']) + list(hard_videos['delta'])):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Delta bar chart created and saved to {output_path}")

def create_radar_plot_by_biomarker(feature_correlations, feature, output_path):
    """Create a radar plot showing correlations for all participants for a single biomarker."""
    participants = []
    correlations = []
    p_values = []
    
    # Collect data across participants
    for participant_id, features in feature_correlations.items():
        if feature in features:
            participants.append(participant_id)
            correlations.append(features[feature]['correlation'])
            p_values.append(features[feature]['p_value'])
    
    if not participants:
        print(f"No participant data found for biomarker: {feature}")
        return
    
    # Set up the radar plot
    angles = np.linspace(0, 2*np.pi, len(participants), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Add participant at the beginning again to close the circle
    participant_labels = [PARTICIPANT_ID_TO_NAME.get(pid, f"Participant {pid}") for pid in participants]
    participant_labels += [participant_labels[0]]
    correlations_plot = correlations + [correlations[0]]
    p_values += [p_values[0]]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot the correlations
    ax.plot(angles, correlations_plot, 'o-', linewidth=2)
    ax.fill(angles, correlations_plot, alpha=0.25)
    
    # Add p-value indicators
    for i, p_value in enumerate(p_values):
        if i < len(angles) and p_value < 0.05:
            ax.plot(angles[i], correlations_plot[i], 'r*', ms=10, 
                   label='p < 0.05' if i == 0 else "")
    
    # Set labels and style
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(participant_labels[:-1], fontsize=10)
    
    # Set y-axis limits to show both positive and negative correlations
    ax.set_ylim(-1, 1)
    
    # Add grid lines
    ax.grid(True)
    
    # Add a legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], markersize=10, label='Correlation'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='p < 0.05')
    ]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add a title
    plt.title(f"Correlation Between {get_display_feature_name(feature)} and Learning Gain Across Participants", 
             pad=20, fontweight='bold')
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Biomarker radar plot created and saved to {output_path}")

def create_correlation_heatmap(feature_correlations, p_values, output_path):
    """Create a correlation heatmap with biomarkers on x-axis and participant IDs on y-axis."""
    # Get unique features and participants
    all_features = set()
    for participant_features in feature_correlations.values():
        all_features.update(participant_features.keys())
    
    all_features = sorted(list(all_features))
    all_participants = sorted(list(feature_correlations.keys()))
    
    # Create matrix for heatmap
    correlation_matrix = np.zeros((len(all_participants), len(all_features)))
    significance_matrix = np.zeros((len(all_participants), len(all_features)), dtype=bool)
    
    # Fill in the matrix
    for i, participant_id in enumerate(all_participants):
        for j, feature in enumerate(all_features):
            if feature in feature_correlations[participant_id]:
                correlation_matrix[i, j] = feature_correlations[participant_id][feature]['correlation']
                significance_matrix[i, j] = p_values[participant_id][feature]['p_value'] < 0.05
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Use a diverging colormap centered at 0
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Plot the heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap=cmap,
                vmin=-1, 
                vmax=1,
                center=0,
                linewidths=.5,
                xticklabels=[get_display_feature_name(f) for f in all_features],
                yticklabels=[f"Participant ID: {p} ({PARTICIPANT_ID_TO_NAME.get(p, 'Unknown')})" 
                            for p in all_participants])
    
    # Mark significant correlations
    for i in range(len(all_participants)):
        for j in range(len(all_features)):
            if significance_matrix[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    plt.title("Biomarker-Learning Correlations Across Participants", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    
    # Add a legend for significance
    ax.legend([plt.Rectangle((0,0),1,1, fill=False, edgecolor='black', lw=2)], 
              ['p < 0.05'], 
              loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap created and saved to {output_path}")

def save_p_values(feature_correlations, output_path):
    """Save all p-values to a CSV file for further analysis."""
    data = []
    
    for participant_id, features in feature_correlations.items():
        participant_name = PARTICIPANT_ID_TO_NAME.get(participant_id, f"Participant {participant_id}")
        
        for feature, stats in features.items():
            data.append({
                'participant_id': participant_id,
                'participant_name': participant_name,
                'feature': feature,
                'feature_name': get_display_feature_name(feature),
                'correlation': stats['correlation'],
                'p_value': stats['p_value'],
                'significant': stats['p_value'] < 0.05
            })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"P-values saved to {output_path}")

def main():
    # Load participant data
    print("Loading participant data...")
    participants_data = load_participant_data()
    print(f"Loaded data for {len(participants_data)} participants")
    
    # Extract features
    print("Extracting features...")
    df = extract_features(participants_data)
    print(f"Extracted data for {len(df)} video sessions")
    
    # Create output directories
    os.makedirs('plots_new', exist_ok=True)
    
    # Define feature groups
    emotion_features = [
        'diversity',
        'non_neutral_percent',
        'positive_percent',
        'negative_percent',
        'percentages.Neutral',
        'percentages.Happy',
        'percentages.Sad',
        'percentages.Surprise',
        'percentages.Fear',
        'percentages.Disgust',
        'percentages.Angry'
    ]
    
    eye_features = [
        'total_blinks',
        'short_blinks_percent',
        'medium_blinks_percent',
        'long_blinks_percent',
        'short_low_disp_percent',
        'medium_low_disp_percent',
        'long_low_disp_percent',
        'small_pupil_percent',
        'medium_pupil_percent',
        'large_pupil_percent'
    ]
    
    all_features = emotion_features + eye_features
    
    # Process each participant
    all_participant_correlations = {}
    all_participant_p_values = {}
    
    for participant_id in participants_data.keys():
        print(f"\nProcessing Participant ID: {participant_id} ({PARTICIPANT_ID_TO_NAME.get(participant_id, 'Unknown')})")
        
        # Create participant directory
        participant_dir = os.path.join('plots_new', f'participant_{participant_id}')
        os.makedirs(participant_dir, exist_ok=True)
        
        # 1. Create radar plot for all videos
        all_videos_output = os.path.join(participant_dir, 'all_videos_radar.png')
        all_correlations = create_radar_plot_for_participant(
            df, participant_id, all_features, 
            "Biomarker Correlations with Learning Gain (All Videos)",
            all_videos_output
        )
        
        # Store correlations for later use
        if all_correlations:
            all_participant_correlations[participant_id] = {}
            all_participant_p_values[participant_id] = {}
            
            for feature, stats in all_correlations.items():
                all_participant_correlations[participant_id][feature] = stats['correlation']
                all_participant_p_values[participant_id][feature] = stats
        
        # 2. Create radar plot for easy videos
        easy_df = df[df['difficulty'] == 'Easy']
        easy_videos_output = os.path.join(participant_dir, 'easy_videos_radar.png')
        create_radar_plot_for_participant(
            easy_df, participant_id, all_features, 
            "Biomarker Correlations with Learning Gain (Easy Videos Only)",
            easy_videos_output
        )
        
        # 3. Create radar plot for hard videos
        hard_df = df[df['difficulty'] == 'Hard']
        hard_videos_output = os.path.join(participant_dir, 'hard_videos_radar.png')
        create_radar_plot_for_participant(
            hard_df, participant_id, all_features, 
            "Biomarker Correlations with Learning Gain (Hard Videos Only)",
            hard_videos_output
        )
        
        # 4. Create delta bar chart
        delta_output = os.path.join(participant_dir, 'delta_by_video.png')
        create_delta_bar_chart(df, participant_id, delta_output)
    
    # Create radar plots for each biomarker
    print("\nCreating biomarker radar plots...")
    biomarker_dir = os.path.join('plots_new', 'biomarkers')
    os.makedirs(biomarker_dir, exist_ok=True)
    
    for feature in all_features:
        output_path = os.path.join(biomarker_dir, f"{feature.replace('.', '_')}_radar.png")
        create_radar_plot_by_biomarker(all_participant_p_values, feature, output_path)
    
    # Create correlation matrix
    print("\nCreating correlation matrix...")
    matrix_output = os.path.join('plots_new', 'correlation_matrix.png')
    create_correlation_heatmap(all_participant_p_values, all_participant_p_values, matrix_output)
    
    # Save p-values
    print("\nSaving p-values...")
    p_values_output = os.path.join('plots_new', 'biomarker_p_values.csv')
    save_p_values(all_participant_p_values, p_values_output)
    
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main() 