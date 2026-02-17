import os

import pandas as pd


# -----------------------------
# 1. Load CSV files
# -----------------------------
# Replace these with your actual CSV filenames/paths
def add_system_time(ecg_csv_path, webcam_csv_path, output_csv_path):
    ecg_df = pd.read_csv(ecg_csv_path)
    webcam_df = pd.read_csv(webcam_csv_path)
    ecg_final_ts = ecg_df["timestamp"].iloc[-1]
    webcam_final_ts = webcam_df["timestamp"].iloc[-1]

    video_start_ts = ecg_final_ts - webcam_final_ts
    webcam_df["system_timestamp"] = video_start_ts + webcam_df["timestamp"]

    webcam_df.to_csv(output_csv_path, index=False)
    print(f"Updated webcam data saved to: {output_csv_path}")


# get all files in each dir in the data folder

for dir in os.listdir("./results/"):
    if not os.path.isdir(f"./results/{dir}"):
        continue

    for file in os.listdir(f"./results/{dir}"):
        if file.startswith("ecg"):
            ecg_csv_path = f"./results/{dir}/{file}"
        if file.startswith("webcam"):
            webcam_csv_path = f"./results/{dir}/{file}"

    output_csv_path = f"results/{dir}/webcam_emotions.csv"
    add_system_time(ecg_csv_path, webcam_csv_path, output_csv_path)
