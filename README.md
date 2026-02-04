# Robust Emotion Recognition via Attentive Denoising Autoencoders

This repository contains the source code and technical report for the **BLG 454E - Learning from Data** Term Project. The proposed framework achieved **2nd Place** in the private Kaggle competition among the class participants at **Istanbul Technical University (ITU)**.

## 🏆 Achievements
* **Kaggle Competition Rank:** 2nd Place 
* **Kaggle Leaderboard Score:** 0.47644 (Macro F1)
* **Local Validation Score:** 0.4980 (Group F1) 

## 📌 Project Overview
The goal of this project is to categorize 1793-dimensional physiological feature vectors into four emotional quadrants based on the valence-arousal representation:
* **Class 0 (LVLA):** Low Valence - Low Arousal 
* **Class 1 (HVLA):** High Valence - Low Arousal
* **Class 2 (LVHA):** Low Valence - High Arousal
* **Class 3 (HVHA):** High Valence - High Arousal 

The framework is specifically designed to handle high-dimensional noise and achieve subject-independent generalization using 5-fold Group Cross-Validation.


## 🛠️ Tech Stack & Methodology
* **Framework:** PyTorch 
* **Architecture:** Attentive Compact Denoising Autoencoder (DAE) with Residual Connections.
* **Key Components:**
    * **Feature Attention:** A channel-wise gating mechanism to suppress irrelevant noise and focus on informative signals.
    * **Denoising Autoencoder (DAE):** Learns stable latent representations through reconstruction-based regularization.
    * **Iterative Self-Training:** A multi-generational Teacher-Student strategy utilizing confidence-based pseudo-labeling on unlabeled test data.
    * **Robust Preprocessing:** `QuantileTransformer` with a Gaussian distribution to neutralize extreme sensor artifacts and outliers.

## 📊 Performance Progression
The model demonstrated a clear performance gain through iterative pseudo-label expansion and class-specific probability boosting.

| Iteration | Mean F1-Score | Pseudo-Label Count |
| :--- | :---: | :---: |
| **Generation 1** | 0.4654 | 3,488 samples  |
| **Generation 2** | **0.4980**  | **5,551 samples** |
| **Kaggle Result** | **0.47644** | **Competition 2nd Place** |



## 💾 Dataset Access
Due to GitHub's file size limitations, the raw `train.csv` and `test.csv` files are not included in this repository.
* **Train Set:** 22,496 samples (features, labels, person IDs).
* **Test Set:** 10,656 samples (features only).
* **Access:** The dataset can be accessed via the [Kaggle Competition Page](https://www.kaggle.com/t/500fa16813f747bc860725061d58100f).


## 🚀 How to Run
1. Ensure the following libraries are installed: `torch`, `pandas`, `numpy`, `scikit-learn`, `tqdm`.
2. Place the dataset files in the root directory.
3. Execute the script:
   ```bash
   python emotionRecognation.py
