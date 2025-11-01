# -*- coding: utf-8 -*-
"""Jumbled_Video_Projectled_2.ipynb
Original file location
https://colab.research.google.com/drive/1C8PJvwbUYE3BLKXGoDiR9M7FgPZ4DAKQ
"""

# Install OpenCV
!pip install opencv-python
# Import required libraries
import cv2
import numpy as np
import time
from google.colab import drive
from google.colab import files


# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')


INPUT_VIDEO_PATH = "/content/drive/MyDrive/Jumbled_Video_Project/jumbled_video.mp4"
# Output path for the reconstructed video
OUTPUT_VIDEO_PATH = "reconstructed_video.mp4"
FRAME_COUNT = 300 # 10 seconds * 30 FPS

# PHASE 1: Frame extraction
def extract_frames(video_path):
    print("Starting frame extraction")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Couldn't open video file at {video_path}. Check the path and Drive mount.")
        return None

    # Getting video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames, width, height, fps

# Phase 3: Write the reconstructed video
def create_video(frames, output_path, width, height, fps):
    print(f"Saving video to {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Failed to create video writer.")
        return

    for f in frames:
        out.write(f)

    out.release()
    print("Done! Video saved.")

# PHASE 2: The reordering algorithm

def calculate_similarity(frame1, frame2):

    # Example: Calculate MSE (lower value means higher similarity)
    err = np.sum((frame1.astype("float") - frame2.astype("float")) ** 2)
    err /= float(frame1.shape[0] * frame1.shape[1] * frame1.shape[2])
    return err

def reorder_frames(jumbled_frames):

    N = len(jumbled_frames)
    if N == 0:
        return []

    # Initialize variables
    ordered_frames = []
    used_indices = [False] * N

    # Starting with the first frame in the jumbled list as the initial frame (index 0)
    current_index = 0
    ordered_frames.append(jumbled_frames[current_index])
    used_indices[current_index] = True

    for _ in range(N - 1):
        current_frame = ordered_frames[-1]
        best_next_index = -1
        min_diff = float('inf')


        for next_index in range(N):
            if not used_indices[next_index]:
                next_frame = jumbled_frames[next_index]
                diff = calculate_similarity(current_frame, next_frame)

                if diff < min_diff:
                    min_diff = diff
                    best_next_index = next_index

        # If a next frame was found, add it to the sequence
        if best_next_index != -1:
            ordered_frames.append(jumbled_frames[best_next_index])
            used_indices[best_next_index] = True
        else:

            print("Error: Could not find a next frame.")
            break

    print("Frames reordered using the Greedy Search approach.")
    return ordered_frames

if __name__ == "__main__":
    # 1 Extraction
    frame_data = extract_frames(INPUT_VIDEO_PATH)
    if frame_data is None:
        exit()

    jumbled_frames, width, height, fps = frame_data

    print("\n Starting Reordering Algorithm ")
    # 2 Reordering (The time measured here is DELIVERABLE 4)
    reordering_start_time = time.time()
    ordered_frames = reorder_frames(jumbled_frames)
    reordering_end_time = time.time()

    reordering_time = reordering_end_time - reordering_start_time
    # Reversal code
    ordered_frames.reverse()

    print("Frames list has been reversed and reordering complete.")

    # 3 Reconstruction (This uses the now-reversed 'ordered_frames' list)
    create_video(ordered_frames, OUTPUT_VIDEO_PATH, width, height, fps)

    print(f"\nDownloading {OUTPUT_VIDEO_PATH}...")
    files.download(OUTPUT_VIDEO_PATH)

    # Execution time log (DELIVERABLE 4)
    print("\n FINAL EXECUTION TIME LOG")
    print(f"Reordering Algorithm Execution Time: {reordering_time:.4f} seconds")