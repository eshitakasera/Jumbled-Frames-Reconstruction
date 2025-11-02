"""
Original file is located at
    https://colab.research.google.com/drive/1-nwqTNY_1L6HyfsOSeslqlDEYHowPmDP
"""

# Install required libraries (TensorFlow for the AI model, OpenCV for video processing)
!pip install tensorflow opencv-python
!pip install scikit-image
# Install scikit-image for SSIM
# Import required libraries
import cv2
import numpy as np
import time
from google.colab import drive
from google.colab import files
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Mount Google Drive
print("Mounting Google Drive")
drive.mount('/content/drive')
INPUT_VIDEO_PATH = "/content/drive/MyDrive/Jumbled_Video_Project/jumbled_video.mp4"
OUTPUT_VIDEO_PATH = "reconstructed_video_AI.mp4"
print("Setting up MobileNetV2 model")
# The 'feature_model' object is initialized
feature_model = MobileNetV2(
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3) # MobileNetV2 expects 224x224 input
)
print("Model ready.")

# Frame Extraction
def extract_frames(video_path):
    print("Starting frame extraction")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file at {video_path}.")
        return None

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

# PHASE 2 THE REORDERING ALGORITHM (AI-BASED)

def get_embedding(frame):
    # Resize frame and preprocess for MobileNetV2
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    emb = feature_model.predict(img, verbose=0)
    return emb[0]

def compute_all_embeddings(frames):
    print(f"Computing embeddings for {len(frames)} frames")
    embeddings = []
    for f in frames:
        embeddings.append(get_embedding(f))
    return embeddings

def calculate_similarity_emb(emb1, emb2):
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return 1 - sim

def reorder_frames_with_embeddings(frames, embeddings):
    N = len(frames)
    if N == 0:
        return []

    ordered_frames = []
    used = [False] * N

    current_index = 0
    ordered_frames.append(frames[current_index])
    used[current_index] = True

    # Loop until all frames are used
    for step in range(N - 1):
        current_emb = embeddings[current_index]
        min_diff = float('inf')
        best_next = -1

        # Search all unused frames
        for next_index in range(N):
            if not used[next_index]:
                next_emb = embeddings[next_index]
                diff = calculate_similarity_emb(current_emb, next_emb)

                if diff < min_diff:
                    min_diff = diff
                    best_next = next_index

        # Add the best frame to the sequence and update the current index
        if best_next != -1:
            ordered_frames.append(frames[best_next])
            used[best_next] = True
            current_index = best_next
        else:
            print("Could not find a next frame.")
            break

    ordered_frames.reverse()
    print("Frames reordered, reversed, and sequenced using AI embeddings.")
    return ordered_frames

# PHASE 3 VIDEO RECONSTRUCTION
def create_video(frames, output_path, width, height, fps):
    print(f"Creating reconstructed video at {output_path}...")
    # Define the codec and create VideoWriter object ('mp4v' is a common MP4 codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("VideoWriter could not be opened.")
        return

    for frame in frames:
        out.write(frame)

    out.release()
    print("Video reconstruction complete.")

if __name__ == "__main__":
    # 1. Extraction
    frame_data = extract_frames(INPUT_VIDEO_PATH)
    if frame_data is None:
        exit()

    frames, width, height, fps = frame_data

    print("\n Starting Reordering Algorithm (AI Embeddings)")

    # 2. Embeddings and Reordering
    reordering_start_time = time.time()

    # Pre-calculate all embeddings (expensive step)
    embeddings = compute_all_embeddings(frames)

    # Reorder frames using the embeddings (includes the fix)
    ordered_frames = reorder_frames_with_embeddings(frames, embeddings)

    reordering_end_time = time.time()
    reordering_time = reordering_end_time - reordering_start_time

    # 3. Reconstruction
    create_video(ordered_frames, OUTPUT_VIDEO_PATH, width, height, fps)

    # 4. Download the Reconstructed Video
    print(f"\nDownloading {OUTPUT_VIDEO_PATH}...")
    files.download(OUTPUT_VIDEO_PATH)

    # EXECUTION TIME LOG
    print("\n Final Execution Time Log ")
    print(f"Total AI Reordering Execution Time: {reordering_time:.4f} seconds (includes embedding computation)")