import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing JSON files (replace with your actual directory)
directory = "./results-video-break/participant_info/"

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

    return percentages


# Process each JSON file in the directory
for json_file in os.listdir(directory):
    if json_file.endswith(".json"):
        with open(os.path.join(directory, json_file)) as f:
            data = json.load(f)

        # Assume all entries in the file belong to the same participant
        participant_id = data[0]["participant_id"]

        # Dictionaries to store change percentages per biomarker for each difficulty
        easy_changes = {biomarker: [] for biomarker in all_biomarkers}
        hard_changes = {biomarker: [] for biomarker in all_biomarkers}

        # Process each video entry (typically 6 videos per participant)
        for entry in data:
            difficulty = entry.get("difficulty", "Easy")  # Default to Easy if missing
            overall_percentages = calculate_percentages(entry)
            break_percentages = calculate_percentages(entry["break_metrics"])

            # Calculate change % for each biomarker: overall - break
            for biomarker in all_biomarkers:
                change = overall_percentages.get(biomarker, 0) - break_percentages.get(
                    biomarker, 0
                )
                if difficulty.lower() == "easy":
                    easy_changes[biomarker].append(change)
                else:
                    hard_changes[biomarker].append(change)

        # Compute average change % for each biomarker per difficulty group
        avg_easy_changes = {
            biomarker: np.mean(changes) if changes else 0
            for biomarker, changes in easy_changes.items()
        }
        avg_hard_changes = {
            biomarker: np.mean(changes) if changes else 0
            for biomarker, changes in hard_changes.items()
        }

        # Compute the difference: (Easy average - Hard average) for each biomarker
        avg_diff = {
            biomarker: avg_easy_changes[biomarker] - avg_hard_changes[biomarker]
            for biomarker in all_biomarkers
        }

        # For a neat and consistent plot, sort biomarkers alphabetically (or change as desired)
        sorted_biomarkers = sorted(all_biomarkers)
        diff_values = [avg_diff[biomarker] for biomarker in sorted_biomarkers]

        # Create a horizontal bar plot for the participant
        plt.figure(figsize=(10, 8))
        bars = plt.barh(sorted_biomarkers, diff_values, color="skyblue")
        plt.xlabel("Difference in Average Change % (Easy - Hard)", fontsize=12)
        plt.title(
            f"Participant {participant_id} - Biomarker Change Difference",
            fontsize=14,
            pad=15,
        )
        plt.axvline(0, color="grey", linewidth=0.8)
        plt.tight_layout()

        # Annotate each bar with its value
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center",
                ha="left" if width >= 0 else "right",
                fontsize=9,
            )

        plt.savefig(
            f"participant_{participant_id}_difference.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

print(
    "Bar plots for difference (Easy - Hard) have been generated for all participants."
)
