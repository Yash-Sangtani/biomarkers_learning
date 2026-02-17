import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_and_plot_metrics(
    events_csv,
    webcam_csv,
    blink_csv,
    fixation_csv,
    pupil_csv,  # New parameter for pupil diameter CSV
    json_file,
    output_json="participant_data_with_metrics.json",
    emotion_plot_file="emotion_counts_per_video.png",
    blink_plot_file="blink_counts_per_video.png",
    fixation_plot_file="fixation_stats_per_video.png",
    pupil_plot_file="pupil_diameter_counts_per_video.png",
):
    """
    Process CSV files to count emotions, blink, fixation, and pupil diameter metrics within video intervals,
    update the JSON data, and generate plots.

    For blinks and fixations, quantile thresholds (0.33 and 0.66) are computed for the duration values within each video interval.
    Pupil diameter is also categorized using quantiles.

    Parameters:
      events_csv (str): Path to the CSV file containing event logs.
      webcam_csv (str): Path to the CSV file containing webcam emotion data.
      blink_csv (str): Path to the CSV file containing blink data.
      fixation_csv (str): Path to the CSV file containing fixation data.
      pupil_csv (str): Path to the CSV file containing pupil diameter data (with 'diameter3d' column).
      json_file (str): Path to the JSON file with video metadata.
      output_json (str): Path where the updated JSON will be saved.
      emotion_plot_file (str): File path for saving the emotion plot.
      blink_plot_file (str): File path for saving the blink plot.
      fixation_plot_file (str): File path for saving the fixation plot.
      pupil_plot_file (str): File path for saving the pupil diameter plot.
    """

    # -----------------------------
    # 1. Load Data
    # -----------------------------
    events_df = pd.read_csv(events_csv)
    webcam_df = pd.read_csv(webcam_csv)
    blink_df = pd.read_csv(blink_csv)
    fixation_df = pd.read_csv(fixation_csv)
    pupil_df = pd.read_csv(pupil_csv)  # Load pupil diameter data

    # Sort by timestamps
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    webcam_df = webcam_df.sort_values("system_timestamp").reset_index(drop=True)
    blink_df = blink_df.sort_values("start_timestamp_unix").reset_index(drop=True)
    fixation_df = fixation_df.sort_values("start_timestamp_unix").reset_index(drop=True)
    pupil_df = pupil_df.sort_values("pupil_timestamp_unix").reset_index(drop=True)
    pupil_df.dropna(subset=["diameter_3d"], inplace=True)

    # Get unique emotions from webcam data
    unique_emotions = webcam_df["predicted_emotion"].unique()

    # -----------------------------
    # 2. Build maps from event intervals
    # -----------------------------
    # Keys: (participant_id, video_name)
    # Values: dictionaries with metrics
    emotion_map = {}
    blink_map = {}  # Blink counts per category: short, medium, long
    fixation_map = {}  # Fixation counts categorized by duration and dispersion
    pupil_map = {}  # Pupil diameter counts in 3 categories: small, medium, large

    # Dispersion threshold for fixations remains fixed
    dispersion_thresh = 0.75

    # Loop through events to define each video interval (using "Video Start" events)
    for i in range(len(events_df) - 1):
        event_type = events_df.loc[i, "event_type"]
        if event_type != "Video Start":
            continue

        participant_id = events_df.loc[i, "participant_id"]
        start_ts = events_df.loc[i, "timestamp"]
        details_str = events_df.loc[i, "details"]
        # End timestamp: next event's timestamp
        end_ts = events_df.loc[i + 1, "timestamp"]

        # Extract video name from details (expected format: "Video lecture_videos/K-Means.webm started at ...")
        match = re.search(r"Video\s+(.*?)\s+started", details_str)
        if not match:
            continue
        video_name = match.group(1)

        key = (participant_id, video_name)

        # ---- Process Emotions ----
        subset_emotions = webcam_df[
            (webcam_df["system_timestamp"] >= start_ts)
            & (webcam_df["system_timestamp"] < end_ts)
        ]
        counts_emotions = subset_emotions["predicted_emotion"].value_counts().to_dict()
        emotion_map[key] = counts_emotions

        # ---- Process Blinks using quantiles for duration ----
        subset_blinks = blink_df[
            (blink_df["start_timestamp_unix"] >= start_ts)
            & (blink_df["start_timestamp_unix"] < end_ts)
        ]
        if not subset_blinks.empty:
            blink_lower = subset_blinks["duration"].quantile(0.33)
            blink_upper = subset_blinks["duration"].quantile(0.66)
            blink_counts = {"short": 0, "medium": 0, "long": 0}
            for _, row in subset_blinks.iterrows():
                duration = row["duration"]
                if duration < blink_lower:
                    blink_counts["short"] += 1
                elif duration < blink_upper:
                    blink_counts["medium"] += 1
                else:
                    blink_counts["long"] += 1
        else:
            blink_counts = {"short": 0, "medium": 0, "long": 0}
        blink_map[key] = blink_counts

        # ---- Process Fixations using quantiles for duration ----
        subset_fixations = fixation_df[
            (fixation_df["start_timestamp_unix"] >= start_ts)
            & (fixation_df["start_timestamp_unix"] < end_ts)
        ]
        if not subset_fixations.empty:
            fix_lower = subset_fixations["duration"].quantile(0.33)
            fix_upper = subset_fixations["duration"].quantile(0.66)
        else:
            fix_lower = fix_upper = 0
        fixation_categories = {
            "short": {"low_dispersion": 0, "high_dispersion": 0},
            "medium": {"low_dispersion": 0, "high_dispersion": 0},
            "long": {"low_dispersion": 0, "high_dispersion": 0},
        }
        for _, row in subset_fixations.iterrows():
            duration = row["duration"]
            dispersion = row["dispersion"]
            if duration < fix_lower:
                duration_cat = "short"
            elif duration < fix_upper:
                duration_cat = "medium"
            else:
                duration_cat = "long"
            if dispersion < dispersion_thresh:
                fixation_categories[duration_cat]["low_dispersion"] += 1
            else:
                fixation_categories[duration_cat]["high_dispersion"] += 1
        fixation_map[key] = fixation_categories

        # ---- Process Pupil Diameter using quantiles ----
        subset_pupil = pupil_df[
            (pupil_df["pupil_timestamp_unix"] >= start_ts)
            & (pupil_df["pupil_timestamp_unix"] < end_ts)
        ]
        if not subset_pupil.empty:
            p_low = subset_pupil["diameter_3d"].quantile(0.33)
            p_high = subset_pupil["diameter_3d"].quantile(0.66)
            pupil_counts = {"small": 0, "medium": 0, "large": 0}
            for _, row in subset_pupil.iterrows():
                d = row["diameter_3d"]
                if d < p_low:
                    pupil_counts["small"] += 1
                elif d < p_high:
                    pupil_counts["medium"] += 1
                else:
                    pupil_counts["large"] += 1
        else:
            pupil_counts = {"small": 0, "medium": 0, "large": 0}
        pupil_map[key] = pupil_counts

    # -----------------------------
    # 3. Update JSON with metrics
    # -----------------------------
    with open(json_file, "r") as f:
        data = json.load(f)  # List of dictionaries

    for item in data:
        p_id = item.get("participant_id")
        vid = item.get("video")  # e.g., "lecture_videos/K-Means.webm"
        key = (p_id, vid)
        item["emotions"] = emotion_map.get(key, {})
        item["blinks"] = blink_map.get(key, {"short": 0, "medium": 0, "long": 0})
        item["fixations"] = fixation_map.get(
            key,
            {
                "short": {"low_dispersion": 0, "high_dispersion": 0},
                "medium": {"low_dispersion": 0, "high_dispersion": 0},
                "long": {"low_dispersion": 0, "high_dispersion": 0},
            },
        )
        item["pupil_diameter"] = pupil_map.get(
            key, {"small": 0, "medium": 0, "large": 0}
        )

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Updated JSON written to {output_json}")

    # -----------------------------
    # 4. Create Plots
    # -----------------------------
    # -- Emotion Plot --
    plot_data_emotions = []
    for item in data:
        video = item.get("video")
        emotions = item.get("emotions", {})
        row = {"video": video}
        for emo in unique_emotions:
            row[emo] = emotions.get(emo, 0)
        plot_data_emotions.append(row)
    df_emotions = pd.DataFrame(plot_data_emotions).fillna(0)
    df_emotions = df_emotions.groupby("video", as_index=False).sum()
    df_emotions.set_index("video", inplace=True)
    df_emotions_log = np.log1p(df_emotions)
    ax1 = df_emotions_log.plot(kind="bar", figsize=(12, 6))
    plt.title("Log-Transformed Emotion Counts per Video")
    plt.xlabel("Video")
    plt.ylabel("Log(Emotion Count)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Emotion")
    plt.tight_layout()
    plt.savefig(emotion_plot_file)
    plt.show()
    print(f"✅ Emotion plot saved to {emotion_plot_file}")

    # -- Blink Plot --
    blink_plot_data = []
    for item in data:
        video = item.get("video")
        blinks = item.get("blinks", {})
        row = {"video": video}
        for cat in ["short", "medium", "long"]:
            row[cat] = blinks.get(cat, 0)
        blink_plot_data.append(row)
    df_blinks = pd.DataFrame(blink_plot_data).fillna(0)
    df_blinks = df_blinks.groupby("video", as_index=False).sum()
    df_blinks.set_index("video", inplace=True)
    ax2 = df_blinks.plot(kind="bar", figsize=(12, 6))
    plt.title("Blink Counts per Video")
    plt.xlabel("Video")
    plt.ylabel("Blink Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Blink Duration Category")
    plt.tight_layout()
    plt.savefig(blink_plot_file)
    plt.show()
    print(f"✅ Blink plot saved to {blink_plot_file}")

    # -- Fixation Plot --
    fixation_plot_data = []
    for item in data:
        video = item.get("video")
        fix = item.get("fixations", {})
        row = {"video": video}
        for duration_cat in ["short", "medium", "long"]:
            low = fix.get(duration_cat, {}).get("low_dispersion", 0)
            high = fix.get(duration_cat, {}).get("high_dispersion", 0)
            row[f"{duration_cat}_low"] = low
            row[f"{duration_cat}_high"] = high
        fixation_plot_data.append(row)
    df_fix = pd.DataFrame(fixation_plot_data).fillna(0)
    df_fix = df_fix.groupby("video", as_index=False).sum()
    df_fix.set_index("video", inplace=True)
    ax3 = df_fix.plot(kind="bar", figsize=(12, 6))
    plt.title("Fixation Counts per Video by Duration and Dispersion")
    plt.xlabel("Video")
    plt.ylabel("Fixation Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Fixation Category")
    plt.tight_layout()
    plt.savefig(fixation_plot_file)
    plt.show()
    print(f"✅ Fixation plot saved to {fixation_plot_file}")

    # -- Pupil Diameter Plot --
    pupil_plot_data = []
    for item in data:
        video = item.get("video")
        pupil = item.get("pupil_diameter", {})
        row = {"video": video}
        for cat in ["small", "medium", "large"]:
            row[cat] = pupil.get(cat, 0)
        pupil_plot_data.append(row)
    df_pupil = pd.DataFrame(pupil_plot_data).fillna(0)
    df_pupil = df_pupil.groupby("video", as_index=False).sum()
    df_pupil.set_index("video", inplace=True)
    ax4 = df_pupil.plot(kind="bar", figsize=(12, 6))
    plt.title("Pupil Diameter Counts per Video")
    plt.xlabel("Video")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Pupil Diameter Category")
    plt.tight_layout()
    plt.savefig(pupil_plot_file)
    plt.show()
    print(f"✅ Pupil diameter plot saved to {pupil_plot_file}")


# -----------------------------
# Example usage for each participant directory
# -----------------------------
for dir in os.listdir("./data/"):
    pupil_dir_path = os.path.join("./data", dir, "eye", "exports", "001")
    res_dir_path = os.path.join("./results", dir)
    if not os.path.isdir(pupil_dir_path):
        continue

    events_csv_path = os.path.join(res_dir_path, "timestamped_events.csv")
    webcam_csv_path = os.path.join(res_dir_path, "webcam_emotions.csv")
    blink_csv_path = os.path.join(pupil_dir_path, "blinks_unix_datetime.csv")
    fixation_csv_path = os.path.join(pupil_dir_path, "fixations_unix_datetime.csv")
    pupil_csv_path = os.path.join(pupil_dir_path, "pupil_positions_unix_datetime.csv")
    json_file_path = os.path.join(res_dir_path, "mcq.json")
    json_output_file = os.path.join(res_dir_path, f"{dir}.json")
    emotion_plot_file = os.path.join(
        res_dir_path, "plots", "emotion_counts_per_video.png"
    )
    blink_plot_file = os.path.join(res_dir_path, "plots", "blink_counts_per_video.png")
    fixation_plot_file = os.path.join(
        res_dir_path, "plots", "fixation_stats_per_video.png"
    )
    pupil_plot_file = os.path.join(
        res_dir_path, "plots", "pupil_diameter_counts_per_video.png"
    )

    # Ensure the plots directory exists
    os.makedirs(os.path.join(res_dir_path, "plots"), exist_ok=True)

    process_and_plot_metrics(
        events_csv_path,
        webcam_csv_path,
        blink_csv_path,
        fixation_csv_path,
        pupil_csv_path,
        json_file_path,
        output_json=json_output_file,
        emotion_plot_file=emotion_plot_file,
        blink_plot_file=blink_plot_file,
        fixation_plot_file=fixation_plot_file,
        pupil_plot_file=pupil_plot_file,
    )
