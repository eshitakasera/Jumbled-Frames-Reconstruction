# Video Reconstruction Project: Dual Approach (MSE & DeepLearning_Approach)

##  Project Overview

This project focuses on reconstructing a **jumbled video sequence** by analyzing frame-to-frame
similarity and determining the most likely chronological order.  
The goal is to rebuild a **coherent video** from disordered frames using two distinct algorithms:

1. **Lightweight, interpretable Mean Squared Error (MSE) approach**
2. **Advanced, high-accuracy AI-based feature embedding approach**

Both methods use a **Greedy Search Strategy** to find the optimal order efficiently.

---

## Setup Instructions

### 1. Install Dependencies

Ensure you have **Python 3.8+** and `pip` installed.
**For MSE Solution (Baseline):**
```bash
pip install -r requirements.txt 
# Or manually:
pip install opencv-python numpy
```

For AI Solution (Advanced):
```bash
# This requires TensorFlow and scikit-image (for potential SSIM use)
pip install tensorflow opencv-python scikit-image
```

### 2. Mount Google Drive (Colab Only)
Upload your jumbledvideo.mp4 to your Google Drive (e.g., MyDrive/JumbledVideoProject).

Then run these commands in Colab:
```bash
from google.colab import drive
drive.mount('/content/drive')
```

Update the input path in your script:
```bash
INPUT_VIDEO_PATH = "/content/drive/MyDrive/JumbledVideoProject/jumbledvideo.mp4"
```

### 3. Run the Code

Select the appropriate script and execute it:
```bash
# To run the Baseline MSE Solution
python src/reconstructvideo_mse.py

# To run the Advanced AI Solution
python src/reconstructvideo_ai.py
```
The script will:

- Extract all frames from the video.
- Reorder them using the chosen similarity-based greedy algorithm.
- Save the reconstructed video (reconstructedvideo.mp4 or reconstructed_video_AI.mp4).
  
### Algorithm Explanation

This project implements the same Greedy Frame Reordering strategy, but with two different similarity metrics.

### 1. Baseline Solution: Mean Squared Error (MSE)

This approach is **lightweight and fast**, relying purely on raw pixel differences.

**Similarity Metric: Mean Squared Error (MSE)**
Each pair of frames is compared by computing:
##### MSE(**A**,**B**) = 1/n $$\sum_{i=0}^{n}$$ (A~i~ - B~i~)^2

Here:
- A~i~ and B~i~ denote the pixel intensities of frames A and B.
- n is the total number of pixels.

A **lower MSE value** indicates **higher visual similarity**.

### 2. Advanced Solution: AI Feature Embeddings

This approach uses a pre-trained Convolutional Neural Network (CNN) to compare frames based on semantic content rather than just raw pixels — making it more robust to lighting changes or compression artifacts.

Feature Extraction: MobileNetV2
- Each frame is resized to 224 × 224.
- It is passed through MobileNetV2 (pre-trained on ImageNet) to extract a feature vector (embedding).
- This embedding represents the high-level semantic structure of the frame.

**Similarity Metric: Cosine Similarity**
The similarity between two frame embeddings Emb~A~ and Emb~B~ is calculated as:

**Similarity** **= Emb~A~.Emb~B~**/**∣∣Emb~A~∣∣.∣∣Emb~A~∣∣**

The algorithm then uses **(1 - Cosine Similarity)** as the **difference score**.
A **lower difference score** (higher cosine similarity) means frames are more likely consecutive.

### 3. Greedy Frame Reordering (Common to Both)

Both approaches share the same greedy reconstruction logic:

Start from an arbitrary frame.

At each step, select the next unvisited frame with the lowest difference score (either MSE or 
Cosine Similarity
1.  Cosine Similarity).
2.  Repeat until all frames are ordered.
3.  Reverse the final sequence to correct potential backward ordering.

###  Why This Method Was Chosen

The dual-approach design was chosen to balance **interpretability** and **accuracy** across different use cases:

- **MSE (Baseline)** was selected for its **simplicity, speed, and transparency**.  
  It provides an interpretable baseline that allows users to clearly understand how frame similarity is measured using pixel-level differences.

- **AI-based (MobileNetV2)** was introduced to overcome the limitations of MSE.  
  By leveraging **deep feature embeddings**, this approach captures **semantic similarity** between frames — making it robust against lighting changes, compression noise, or camera motion.

Together, these two methods enable both **quick prototyping** and **high-fidelity reconstruction**, offering flexibility based on computational resources and video complexity.


###  Key design considerations (accuracy, time complexity, parallelism, etc.)

| **Aspect** | **MSE Solution (Baseline)** | **AI Solution (Advanced)** |
| :---------- | :-------------------------- | :-------------------------- |
| **Metric** | Raw Pixel Difference | Semantic Feature Difference |
| **Speed / Complexity** | Very Fast (O(N²)) | Slower (due to model inference) but more accurate |
| **Accuracy** | Moderate (Sensitive to lighting/noise) | High (Robust to visual variations) |
| **Requirements** | OpenCV, NumPy | TensorFlow, MobileNetV2, scikit-image |
| **Use Case** | Simple or synthetic videos | Complex, real-world videos |
| **Interpretability** | High | Moderate |
| **Parallelism** | Can be easily parallelized (frame comparisons are independent) | Supports GPU acceleration (TensorFlow-based parallelism) |


###  Output

**Final reconstructed videos:**
- Reconstructed_Video_EshitaKasera.mp4

**Execution Time:** Printed and logged for comparison.
**Evaluation:** Visually validated for temporal smoothness and continuity.

###  Summary

| **Approach** | **Strength** | **Weakness** |
| :------------ | :------------ | :------------ |
| **MSE (Classical)** | Fast, simple, easy to interpret | Struggles with lighting/compression changes |
| **AI (MobileNetV2)** | Robust and semantically aware | Slower, depends on GPU for best performance |


