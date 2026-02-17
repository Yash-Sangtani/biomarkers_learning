import json
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_average_scores(data):
    """Calculate average baseline and test scores for each participant."""
    baseline_scores = []
    test_scores = []
    
    for entry in data:
        baseline_scores.append(entry['baseline'])
        test_scores.append(entry['test'])
    
    return np.mean(baseline_scores), np.mean(test_scores)

# Define the directory containing JSON files
directory = "./participant_info"

# Store scores for each participant
participant_scores = {}

# Process each JSON file in the directory
for json_file in os.listdir(directory):
    if json_file.endswith(".json"):
        with open(os.path.join(directory, json_file)) as f:
            data = json.load(f)
            
        participant_id = data[0]["participant_id"]
        avg_baseline, avg_test = calculate_average_scores(data)
        participant_scores[participant_id] = {
            'baseline': avg_baseline,
            'test': avg_test
        }

# Create the bar plot
participants = sorted(participant_scores.keys())
x = np.arange(len(participants))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, [participant_scores[p]['baseline'] for p in participants], 
                width, label='Baseline', color='#1f77b4')
rects2 = ax.bar(x + width/2, [participant_scores[p]['test'] for p in participants], 
                width, label='Test', color='#ff7f0e')

# Customize the plot
ax.set_ylabel('Score', fontsize=12, weight='bold')
ax.set_xlabel('Participant ID', fontsize=12, weight='bold')
ax.set_title('Average Baseline vs Test Scores by Participant', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f'P{p}' for p in participants], fontsize=10, weight='bold')
ax.set_ylim(0, 5)  # Set y-axis limit from 0 to 5
ax.grid(True, axis='y', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(fontsize=10)

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    weight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('score_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Score comparison plot has been generated and saved as 'score_comparison.png'") 