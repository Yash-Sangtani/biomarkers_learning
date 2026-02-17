import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)


# Define function to process a single CSV file
def process_csv_file(csv_path):
    print(f"Processing {os.path.basename(csv_path)}...")

    # Load the CSV file
    events_df = pd.read_csv(csv_path)

    # Define the correct answers for each video
    correct_answers = {
        "lecture_videos/Genes.webm": {
            "baseline_answers": [
                "It carries genetic instructions for making proteins",
                "They are segments of DNA that dictate protein production",
                "Long strands of DNA wrapped around histone proteins",
                "DNA is the molecular blueprint; genes are functional segments; chromosomes organize DNA",
                "By reading the ordered sequence of nucleotides in DNA",
            ],
            "test_answers": [
                "It provides the template for protein synthesis",
                "Genes are segments of DNA that dictate protein formation",
                "Long strands of DNA wrapped around histone proteins",
                "They provide the instructions for making proteins that influence traits",
                "It ensures genetic balance from both parents",
            ],
        },
        "lecture_videos/Super vs Unsuper Learning.mkv": {
            "baseline_answers": [
                "Uses labeled data",
                "Target variable",
                "It discovers patterns in unlabeled data",
                "Clustering",
                "By finding similarities in features",
            ],
            "test_answers": [
                "It learns from labeled examples",
                "Target variable",
                "It identifies patterns in unlabeled data",
                "Clustering",
                "To group data points based on their similarity",
            ],
        },
        "lecture_videos/K-Means.webm": {
            "baseline_answers": [
                "To group similar data points into clusters",
                "Centroid",
                "By assigning it to the closest centroid",
                "Clustering is complete",
                "Randomly",
            ],
            "test_answers": [
                "To organize data points into groups based on similarity",
                "Centroid",
                "By measuring the distance to the closest centroid",
                "When centroids no longer move",
                "They are randomly placed",
            ],
        },
        "lecture_videos/UNET.mp4": {
            "baseline_answers": [
                "Transposed convolution",
                "To retain spatial information",
                "To capture high-level features while reducing spatial resolution",
                "Decoder",
                "High-resolution spatial features",
            ],
            "test_answers": [
                "Transposed convolution",
                "They help retain fine spatial details",
                "To focus on high-level feature extraction",
                "Decoder",
                "Detailed spatial features",
            ],
        },
        "lecture_videos/VelPoten_ StreamFunctions.mp4": {
            "baseline_answers": [
                "It is a function whose gradient equals the velocity field",
                "Flow streamlines",
                "By differentiating the potential with respect to spatial coordinates",
                "The flow must be irrotational",
                "To confirm it satisfies Laplace's equation",
            ],
            "test_answers": [
                "It gives the velocity field when differentiated",
                "Flow streamlines",
                "By differentiating with respect to x and y",
                "It must be zero",
                "To confirm that it obeys Laplace's equation",
            ],
        },
        "lecture_videos/Partial trace.mkv": {
            "baseline_answers": [
                "To extract a subsystem's density operator",
                "Because state vectors cannot represent individual subsystems",
                "Consistency of statistical predictions",
                "It reduces the full system's density operator to that of a subsystem",
                "They allow accurate statistical predictions for measurements",
            ],
            "test_answers": [
                "It reduces a composite system's density operator to a subsystem's operator",
                "Because state vectors cannot uniquely represent subsystems in entangled states",
                "Ensuring consistency of statistical predictions",
                "It extracts the density operator of a subsystem",
                "They enable accurate statistical predictions for measurements",
            ],
        },
    }

    # Extract responses and analyze correctness
    results = []
    participant_videos = {}
    current_participant = None

    for _, row in events_df.iterrows():
        if "Consent" in row["event_type"]:
            current_participant = row["participant_id"]
            participant_videos[current_participant] = []

        if "Baseline Start" in row["event_type"]:
            video = row["details"].split("for video ")[1]
            if current_participant:
                participant_videos[current_participant].append(video)

        elif "Baseline Responses" in row["event_type"]:
            responses = eval(row["details"].split("Responses: ")[1])
            correct = correct_answers[video]["baseline_answers"]
            correct_count = sum(
                1 if responses[key] == correct[i] else 0
                for i, key in enumerate(sorted(responses.keys()))
            )

            results.append(
                {
                    "participant_id": current_participant,
                    "video": video,
                    "type": "baseline",
                    "correct_count": correct_count,
                    "source_file": os.path.basename(csv_path),
                }
            )

        elif "Test Responses" in row["event_type"]:
            responses = eval(row["details"].split("Responses: ")[1])
            print(responses)
            correct = correct_answers[video]["test_answers"]
            correct_count = sum(
                1 if responses[key] == correct[i] else 0
                for i, key in enumerate(sorted(responses.keys())[:5])
            )

            # Extract subjective scores from responses
            subjective_keys = sorted(responses.keys())[5:7]
            print(subjective_keys)
            subjective_scores = []

            for key in subjective_keys:
                # Parse subjective score from string like "3 – Quite overwhelming"
                try:
                    score_text = responses[key]
                    if isinstance(score_text, str) and "–" in score_text:
                        score = int(score_text.split("–")[0].strip())
                    else:
                        score = int(score_text.split(" ")[0])
                    subjective_scores.append(score)
                except (ValueError, IndexError):
                    subjective_scores.append(None)

            results.append(
                {
                    "participant_id": current_participant,
                    "video": video,
                    "type": "test",
                    "correct_count": correct_count,
                    "subjective_q1": (
                        subjective_scores[0] if len(subjective_scores) > 0 else None
                    ),
                    "subjective_q2": (
                        subjective_scores[1] if len(subjective_scores) > 1 else None
                    ),
                    "source_file": os.path.basename(csv_path),
                }
            )

    return results


# Function to analyze results and create plots
def analyze_results(results_list, output_suffix=""):
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Define easy and hard videos
    easy_videos = [
        "lecture_videos/Genes.webm",
        "lecture_videos/Super vs Unsuper Learning.mkv",
        "lecture_videos/K-Means.webm",
    ]
    hard_videos = [
        "lecture_videos/UNET.mp4",
        "lecture_videos/VelPoten_ StreamFunctions.mp4",
        "lecture_videos/Partial trace.mkv",
    ]

    # Add difficulty column
    results_df["difficulty"] = results_df["video"].apply(
        lambda v: "Easy" if v in easy_videos else "Hard"
    )

    # Create pivot table with baseline and test scores for each participant and video
    pivot_results = pd.pivot_table(
        results_df,
        values="correct_count",
        index=["participant_id", "video", "difficulty", "source_file"],
        columns=["type"],
        aggfunc="first",
    ).reset_index()

    # Calculate delta (improvement)
    pivot_results["delta"] = pivot_results["test"] - pivot_results["baseline"]

    # Get subjective scores
    subjective_data = results_df[results_df["type"] == "test"][
        [
            "participant_id",
            "video",
            "difficulty",
            "subjective_q1",
            "subjective_q2",
            "source_file",
        ]
    ].copy()

    # Merge with pivot_results
    analysis_df = pd.merge(
        pivot_results,
        subjective_data,
        on=["participant_id", "video", "difficulty", "source_file"],
        how="left",
    )

    # Save processed results to JSON for future aggregate analysis
    output_json = f"./results/{output_suffix}/mcq.json"
    analysis_df.to_json(output_json, orient="records")
    print(f"Saved processed results to {output_json}")

    # Calculate video-level statistics
    video_stats = (
        analysis_df.groupby(["video", "difficulty"])
        .agg(
            {
                "baseline": "mean",
                "test": "mean",
                "delta": "mean",
                "subjective_q1": "mean",
                "subjective_q2": "mean",
            }
        )
        .reset_index()
    )

    # Calculate difficulty-level statistics
    difficulty_stats = (
        analysis_df.groupby("difficulty")
        .agg(
            {
                "baseline": "mean",
                "test": "mean",
                "delta": "mean",
                "subjective_q1": "mean",
                "subjective_q2": "mean",
            }
        )
        .reset_index()
    )

    # Print statistics
    print("==== Video Statistics ====")
    print(video_stats[["video", "difficulty", "baseline", "test", "delta"]])
    print("\n==== Difficulty Statistics ====")
    print(difficulty_stats)

    # -------------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------------

    # Plot 1: Number of correct baseline and test answers per video
    plt.figure(figsize=(14, 7))
    bar_width = 0.35
    index = np.arange(len(video_stats))

    # Sort video stats by difficulty
    video_stats = video_stats.sort_values(by=["difficulty", "video"])

    # Plot bars
    baseline_bars = plt.bar(
        index - bar_width / 2,
        video_stats["baseline"],
        bar_width,
        label="Baseline Score",
        color="lightblue",
    )
    test_bars = plt.bar(
        index + bar_width / 2,
        video_stats["test"],
        bar_width,
        label="Test Score",
        color="navy",
    )

    # Add delta values as text
    for i, row in enumerate(video_stats.itertuples()):
        plt.text(
            i,
            max(row.baseline, row.test) + 0.1,
            f"Δ: {row.delta:.1f}",
            ha="center",
            va="bottom",
            color="green",
            fontweight="bold",
        )

    # Customize plot
    plt.title("Baseline vs Test Scores per Video", fontsize=16)
    plt.xlabel("Video", fontsize=14)
    plt.ylabel("Average Correct Answers", fontsize=14)
    plt.xticks(
        index,
        [v.split("/")[-1].split(".")[0] for v in video_stats["video"]],
        rotation=45,
        ha="right",
    )
    plt.legend(loc="upper left")
    plt.ylim(0, 5)  # Assuming 5 is max score
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add section dividers between easy and hard videos
    easy_count = sum(1 for d in video_stats["difficulty"] if d == "Easy")
    if easy_count > 0 and easy_count < len(video_stats):
        plt.axvline(x=easy_count - 0.5, color="gray", linestyle="--")
        plt.text(easy_count / 2, -0.5, "Easy Videos", ha="center", fontweight="bold")
        plt.text(
            easy_count + (len(video_stats) - easy_count) / 2,
            -0.5,
            "Hard Videos",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f"results/{output_suffix}/plots/video_performance_comparison.png")

    # Plot 2: Easy vs Hard comparison with deltas
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(difficulty_stats))

    # Plot bars
    plt.bar(
        index - bar_width / 2,
        difficulty_stats["baseline"],
        bar_width,
        label="Baseline Score",
        color="lightblue",
    )
    plt.bar(
        index + bar_width / 2,
        difficulty_stats["test"],
        bar_width,
        label="Test Score",
        color="navy",
    )

    # Add delta values as text
    for i, row in enumerate(difficulty_stats.itertuples()):
        plt.text(
            i,
            max(row.baseline, row.test) + 0.2,
            f"Δ: {row.delta:.1f}",
            ha="center",
            va="bottom",
            color="green",
            fontweight="bold",
        )

    # Customize plot
    plt.title("Performance: Easy vs Hard Videos", fontsize=16)
    plt.xlabel("Difficulty Level", fontsize=14)
    plt.ylabel("Average Correct Answers", fontsize=14)
    plt.xticks(index, difficulty_stats["difficulty"])
    plt.legend(loc="upper left")
    plt.ylim(0, 5.5)  # Adjust as needed
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"results/{output_suffix}/plots/difficulty_comparison.png")

    # Plot 3: Delta scores comparison between easy and hard videos
    plt.figure(figsize=(12, 6))
    bar_width = 0.4

    # Sort video stats by difficulty, then by delta
    video_stats = video_stats.sort_values(
        by=["difficulty", "delta"], ascending=[True, False]
    )
    index = np.arange(len(video_stats))

    # Plot bars for delta with different colors for easy and hard
    bars = plt.bar(
        index,
        video_stats["delta"],
        bar_width,
        color=[
            plt.cm.Blues(0.6) if d == "Easy" else plt.cm.Reds(0.6)
            for d in video_stats["difficulty"]
        ],
    )

    # Customize plot
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.title("Improvement from Baseline to Test per Video", fontsize=16)
    plt.xlabel("Video", fontsize=14)
    plt.ylabel("Score Improvement (Test - Baseline)", fontsize=14)
    plt.xticks(
        index,
        [v.split("/")[-1].split(".")[0] for v in video_stats["video"]],
        rotation=45,
        ha="right",
    )

    # Create a custom legend for easy vs hard
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=plt.cm.Blues(0.6), label="Easy Videos"),
        Patch(facecolor=plt.cm.Reds(0.6), label="Hard Videos"),
    ]
    plt.legend(handles=legend_elements, loc="best")

    # Add section dividers between easy and hard videos
    easy_count = sum(1 for d in video_stats["difficulty"] if d == "Easy")
    if easy_count > 0 and easy_count < len(video_stats):
        plt.axvline(x=easy_count - 0.5, color="gray", linestyle="--")

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"results/{output_suffix}/plots/delta_comparison.png")

    # Plot 4: Subjective questions - separate plots for each question
    # Q1: Difficulty Level
    plt.figure(figsize=(12, 6))
    video_stats = video_stats.sort_values(by=["difficulty", "video"])
    index = np.arange(len(video_stats))

    # Plot bars
    plt.bar(
        index,
        video_stats["subjective_q1"],
        width=0.6,
        color=[
            plt.cm.Blues(0.6) if d == "Easy" else plt.cm.Reds(0.6)
            for d in video_stats["difficulty"]
        ],
    )

    # Customize plot
    plt.title("Q1: Perceived Difficulty Level per Video", fontsize=16)
    plt.xlabel("Video", fontsize=14)
    plt.ylabel("Average Difficulty Rating (1-5)", fontsize=14)
    plt.xticks(
        index,
        [v.split("/")[-1].split(".")[0] for v in video_stats["video"]],
        rotation=45,
        ha="right",
    )

    # Create a custom legend
    legend_elements = [
        Patch(facecolor=plt.cm.Blues(0.6), label="Easy Videos"),
        Patch(facecolor=plt.cm.Reds(0.6), label="Hard Videos"),
    ]
    plt.legend(handles=legend_elements, loc="best")

    # Add section dividers between easy and hard videos
    easy_count = sum(1 for d in video_stats["difficulty"] if d == "Easy")
    if easy_count > 0 and easy_count < len(video_stats):
        plt.axvline(x=easy_count - 0.5, color="gray", linestyle="--")

    plt.ylim(0, 5.5)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"results/{output_suffix}/plots/subjective_q1_difficulty.png")

    # Q2: Perceived Improvement
    plt.figure(figsize=(12, 6))

    # Plot bars
    plt.bar(
        index,
        video_stats["subjective_q2"],
        width=0.6,
        color=[
            plt.cm.Blues(0.6) if d == "Easy" else plt.cm.Reds(0.6)
            for d in video_stats["difficulty"]
        ],
    )

    # Customize plot
    plt.title("Q2: Perceived Improvement per Video", fontsize=16)
    plt.xlabel("Video", fontsize=14)
    plt.ylabel("Average Improvement Rating (1-5)", fontsize=14)
    plt.xticks(
        index,
        [v.split("/")[-1].split(".")[0] for v in video_stats["video"]],
        rotation=45,
        ha="right",
    )

    # Create a custom legend
    legend_elements = [
        Patch(facecolor=plt.cm.Blues(0.6), label="Easy Videos"),
        Patch(facecolor=plt.cm.Reds(0.6), label="Hard Videos"),
    ]
    plt.legend(handles=legend_elements, loc="best")

    # Add section dividers between easy and hard videos
    if easy_count > 0 and easy_count < len(video_stats):
        plt.axvline(x=easy_count - 0.5, color="gray", linestyle="--")

    plt.ylim(0, 5.5)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"results/{output_suffix}/plots/subjective_q2_improvement.png")

    # Plot 5: Boxplot of delta scores by difficulty
    plt.figure(figsize=(10, 6))
    boxplot = analysis_df.boxplot(column="delta", by="difficulty", grid=False)

    # Customize the boxplot after it's created
    plt.title("Score Improvement Distribution by Difficulty Level", fontsize=16)
    plt.suptitle("")  # Remove pandas default title
    plt.xlabel("Difficulty Level", fontsize=14)
    plt.ylabel("Score Improvement (Test - Baseline)", fontsize=14)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{output_suffix}/plots/delta_boxplot.png")

    # close all figures
    plt.close("all")


# Main function to handle processing single CSV or multiple CSVs
def main():
    # Check if there are multiple timestamped_events*.csv files
    input_dir = "./videos"
    csv_files = []

    for dir in os.listdir(input_dir):
        filepath = os.path.join(input_dir, dir, "ecg_webcam", "timestamped_events.csv")
        if os.path.isfile(filepath):
            csv_files.append((filepath, dir))

    # csv_files = glob("timestamped_events*.csv")

    if len(csv_files) == 1:
        # Single file analysis
        results = process_csv_file(csv_files[0][0])
        analyze_results(results, output_suffix=csv_files[0][1])
    else:
        # Process each file individually
        all_results = []
        for csv_file, dir in csv_files:
            os.makedirs(f"results/{dir}/plots", exist_ok=True)
            file_results = process_csv_file(csv_file)
            all_results.extend(file_results)
            analyze_results(file_results, output_suffix=dir)

        # Analyze all results together
        if all_results:
            os.makedirs("results/aggregate/plots", exist_ok=True)
            print("\n==== AGGREGATE ANALYSIS ACROSS ALL FILES ====")
            analyze_results(all_results, output_suffix="aggregate")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
