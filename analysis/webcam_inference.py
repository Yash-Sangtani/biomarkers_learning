import os

import cv2
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm  # Progress bar

from detect_emotion.models.make_target_model import make_target_model


class Config:
    pass


cfg = Config()
cfg.ori_shape = (256, 256)
cfg.image_crop_size = (224, 224)
cfg.normalize_mean = [0.5, 0.5, 0.5]
cfg.normalize_std = [0.5, 0.5, 0.5]
cfg.last_stride = 2
cfg.num_classes = 8
cfg.num_branches = cfg.num_classes + 1
cfg.backbone = "resnet18"
cfg.pretrained = "./emotions/weights/AffectNet_res18_acc0.6285.pth"
cfg.pretrained_choice = ""
cfg.bnneck = True
cfg.BiasInCls = False


# Emotion labels
emotion_map = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
}


def preprocess_frame(frame):
    """Preprocesses a video frame for model inference."""
    transform = T.Compose(
        [
            T.Resize(cfg.ori_shape),
            T.CenterCrop(cfg.image_crop_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
        ]
    )
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(frame_pil).unsqueeze(0)


def run_inference_on_video(video_path, output_csv, is_cuda=True, skip_frames=5):
    """Extracts frames, runs inference, and saves results to CSV with progress bar."""

    # Load model
    print("Loading model...")
    model = make_target_model(cfg)
    model.load_param(cfg)
    model.eval()

    if is_cuda and torch.cuda.is_available():
        model = model.cuda()

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"FPS: {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    print(f"Total Frames: {total_frames}")
    frame_count = 0
    results = []

    print(f"Processing video: {video_path} ({total_frames} frames)")

    with tqdm(total=total_frames // skip_frames, desc="Processing Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            if frame_count % skip_frames == 0:  # Process every Nth frame
                timestamp = frame_count / fps  # Timestamp in seconds
                frame_tensor = preprocess_frame(frame)

                if is_cuda and torch.cuda.is_available():
                    frame_tensor = frame_tensor.cuda()

                # Run inference
                with torch.no_grad():
                    pred = model(frame_tensor)
                    prob = F.softmax(pred, dim=-1)
                    idx = torch.argmax(prob.cpu()).item()

                # Save result
                result = {
                    "timestamp": timestamp,
                    "predicted_emotion": emotion_map[idx],
                }
                for i in range(cfg.num_classes):
                    result[emotion_map[i]] = round(prob[0, i].item(), 4)

                results.append(result)
                pbar.update(1)  # Update progress bar

            frame_count += 1

    cap.release()

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")


input_dir = "./data/"

for dir in os.listdir(input_dir):
    video_dir = os.path.join(input_dir, dir, "ecg_webcam", "video_recordings")
    # get file ending with fixed.avi
    video_path = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith("fixed.avi")
    ]
    output_path = os.path.join("results", dir, "webcam_emotions.csv")
    if os.path.exists(output_path):
        print(f"Skipping {dir} as results already exist")
        continue

    if len(video_path) == 0:
        continue

    video_path = video_path[0]

    run_inference_on_video(
        video_path,
        os.path.join("results", dir, "webcam_emotions.csv"),
        is_cuda=False,
    )
