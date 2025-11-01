# Video Reconstruction Project

## Project Overview

This project focuses on reconstructing a jumbled video sequence by analyzing frame-to-frame
similarity and determining the most likely chronological order. The main objective is to rebuild
a coherent video from disordered frames using a lightweight, interpretable algorithm. The
reconstruction is based on Mean Squared Error (MSE) as the frame similarity metric and a
Greedy Search Strategy to find the optimal order efficiently.

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python 3.8+ and pip installed. Install all dependencies using:
pip install -r requirements.txt
If using Google Colab, the script automatically installs OpenCV:
!pip install opencv-python

### 2. Mount Google Drive (Colab Only)

Upload your jumbledvideo.mp4 to your Google Drive (for example, MyDrive/JumbledVideoProject).
Then run:
from google.colab import drive
drive.mount(’/content/drive’)
Update the input path in the script:
INPUTVIDEOPATH = "/content/drive/MyDrive/JumbledVideoProject/jumbledvideo.mp4"

### 3. Run the Code

Run the script either in Google Colab or your local Python environment:
python src/reconstructvideo.py
It will:

- Extract all frames from the video.
- Reorder them using a similarity-based greedy algorithm.
- Save the reconstructed video as reconstructedvideo.mp4.


### 4. Test Using Evaluation Video

Replace the path of INPUTVIDEOPATH with the evaluation video provided. Run the code
again — the script will automatically reconstruct the new video and print the execution time.

## Algorithm Explanation

### 1. Frame Extraction

The video is read using OpenCV’s cv2.VideoCapture, and each frame is stored as a
NumPy array. This step ensures that every frame can be numerically compared on a pixel-
by-pixel basis.

### 2. Similarity Metric: Mean Squared Error (MSE)

Each pair of frames is compared by computing the MSE:

#### MSE(A, B) = 1/n $$\sum_{i=0}^{n}$$ (A~i~ - B~i~)^2^


Here:
- A~i~ and B~i~ denote the pixel intensities of frames A and B,
- n is the total number of pixels.

A lower MSE value indicates higher similarity between two frames. This metric was chosen
because it provides a direct, interpretable measure of visual difference and can be computed
efficiently using NumPy.

### 3. Greedy Frame Reordering

After computing the pairwise MSE between all frames, the algorithm reconstructs the sequence
using a greedy approach:

1. Start from an arbitrary frame (e.g., the first in the input list).
2. Compute the similarity between the current frame and all remaining unvisited frames.
3. Select the frame with the lowest MSE (most similar) as the next frame in sequence.
4. Repeat until all frames are ordered.

This process has a computational complexity of O(N^2^), which is manageable for small to
medium-sized videos (e.g., under 500 frames).

### 4. Direction Correction (Reversal)

In some cases, the greedy algorithm reconstructs the sequence in reverse order. To correct this,
the final list of ordered frames is reversed before reconstruction, resulting in proper forward
playback.


### 5. Video Reconstruction

Once the order is finalized, the frames are written back into a single .mp4 file using OpenCV’s
VideoWriter, preserving the original resolution and frame rate.

## Thought Process and Design Rationale

### Choosing the Similarity Metric

Initially, several similarity measures were considered:

- MSE (Mean Squared Error) – simple, fast, and interpretable.
- SSIM (Structural Similarity Index) – more accurate but computationally heavier.
- Feature-based Matching (ORB/SIFT) – robust to lighting changes but complex to im-
    plement.

MSE was selected as it strikes a balance between speed and accuracy for baseline reconstruc-
tion.

### Selecting the Algorithm Type

An exhaustive search (brute-force) would require factorial time O(N!), which is infeasible
even for small N. Instead, a Greedy Search was adopted, reducing the complexity to O(N^2^).
While this doesn’t guarantee a globally optimal ordering, it yields sufficiently accurate results
for visually coherent reconstruction.

### Optimizations Considered

To improve performance, several ideas were explored:

- Vectorization: NumPy operations were used to compute MSE efficiently without ex-
    plicit Python loops.
- Downsampling: Comparing resized (smaller) frames reduces computation time without
    significantly affecting accuracy.
- Parallelization: Multiprocessing can further accelerate MSE computations for high
    frame counts.

### Possible Enhancements

For future iterations, the algorithm could be extended with:

- SSIM or Histogram-based Similarity for perceptual accuracy.
- Optical Flow to capture motion consistency.
- Deep Feature Embeddings from pretrained CNNs for semantic-level similarity.


## Key Design Considerations

Aspect Decision Reason

Similarity Metric MSE Simple, effective, fast for small frame counts
Algorithm Type Greedy Reduces complexity from O(N!) to O(N^2^)
Performance NumPy vectorization and optional multiprocessing Handles larger videos
Accuracy Moderate Can be improved with feature descriptors or deep models.

## Output

- Final reconstructed video: [reconstructedvideo.mp](https://drive.google.com/file/d/1P48nHAz4zko-y0d-JCa-qrqp7QshQ7BD/view?usp=drivesdk)
- Execution time printed and logged for analysis.
- Output visually validated for temporal smoothness and continuity.


