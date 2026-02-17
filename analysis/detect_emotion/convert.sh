#!/bin/bash

input_dir="./videos/"

for dir in "$input_dir"/*; do
    if [[ "$(basename "$dir")" != "face" ]]; then
        for video in "$dir/ecg_webcam/video_recordings/"*.avi; do
            if [[ -f "$video" ]]; then
                fixed_video="${video%.avi}_fixed.avi"
                echo "Processing $video -> $fixed_video"
                ffmpeg -i "$video" -c copy "$fixed_video"
            fi
        done
    fi
done

