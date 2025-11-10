# Toward Robust EEG Stress Recognition: Cross-Domain Transfer Across Datasets

## Project Overview

This project investigates the **generalizability and robustness of EEG-based stress classification models** across datasets.  
We aim to determine whether models trained on one EEG dataset (SAM40) can effectively generalize to another (EEGMAT), and to compare model performance when trained directly on EEGMAT versus SAM40.

### **Objectives**
1. **Cross-Dataset Testing:** Evaluate SAM40-trained models on the EEGMAT dataset to assess generalization under domain shift.  
2. **Cross-Dataset Training:** Train identical model architectures on EEGMAT and compare performance to SAM40-trained counterparts.  

---

## Motivation

Stress, a key factor in both health and performance, can be measured via neural signatures such as **alpha suppression**, **beta power changes**, and **frontal asymmetry** in EEG recordings.  
Developing accurate, generalizable stress detection models supports applications in healthcare, aviation, and human–computer interaction, enabling real-time stress monitoring for safety and wellness.

---

## Datasets

### **1. SAM40 Dataset**
- **Source:** Ghosh et al. (2022), *Data in Brief 40:107772*  
- **Participants:** 40  
- **Channels:** 32-channel Emotiv Epoc Flex system  
- **Tasks:** Stroop, arithmetic, mirror-image recognition, relaxation (each 25 s × 3 trials)  
- **Preprocessing:** Savitzky–Golay baseline correction and wavelet-thresholding for artifact removal  

### **2. EEGMAT Dataset**
- **Source:** [PhysioNet EEGMAT Dataset](https://physionet.org/physiobank/database/eegmat)  
- **Participants:** 18–20  
- **Channels:** 64-channel BioSemi system @ 256 Hz  
- **Tasks:** Alternating rest and mental arithmetic stress (~15 min each)  
- **Includes:** ECG for multimodal stress correlation  

### **Preprocessing Pipeline**
- Channel mapping to standard **10–20 layout**  
- Resampling to **128 Hz**  
- Artifact rejection via **Independent Component Analysis (ICA)** or **Artifact Subspace Reconstruction (ARS)**  

---

## Methods

### **1. Model Replication and Transfer Testing**
- Reimplement pretrained SAM40 stress classification architectures from:  
  - [sarshardorosti/eeg-stress-classification](https://github.com/sarshardorosti/eeg-stress-classification)  
  - [wavesresearch/eeg_stress_detection](https://github.com/wavesresearch/eeg_stress_detection)  
- Evaluate these models’ **cross-dataset performance** by testing them on EEGMAT.

### **2. Model Training on EEGMAT**
- Extract **statistical** and **spectral features** from EEGMAT.  
- Apply **channel selection** based on [Marthinsen et al. (2023)](https://ceur-ws.org/Vol-3576/), identifying the most informative EEG channels for stress detection.  
- Train and evaluate multiple classifiers:
  - **Convolutional Neural Network (CNN)**
  - **Support Vector Machine (SVM)**
  - **K-Nearest Neighbors (KNN)**

### **3. Cross-Dataset Evaluation**
- Models trained on EEGMAT will be validated on SAM40.  
- Compare metrics to assess **transferability**, **stability**, and **dataset sensitivity**.

---

## Evaluation Plan

| Metric | Purpose |
|:--|:--|
| **Weighted F1-score** | Balanced measure of precision and recall |
| **Balanced Accuracy** | Adjusts for class imbalance |
| **ROC-AUC** | Evaluates binary separability |

### **Validation Strategy**
- **5-fold subject-independent cross-validation** on EEGMAT.  
- **Cross-dataset validation:** Train on SAM40 → Test on EEGMAT.  
- **Statistical tests:** Paired *t*-test or Wilcoxon signed-rank test for significance.  

### **Visualization Tools**
- Confusion matrices per subject.  
- Topographic brain maps showing discriminative EEG channels.

---

## Expected Results & Hypotheses

- Re-implemented SAM40 pipelines on EEGMAT expected to reach **≈80–85% F1**.  
- Domain-adapted models may yield **+3–5 F1 points** improvement under cross-dataset transfer.  
- Channel pruning to ≤16 channels should maintain ≥95% accuracy, supporting lightweight portable EEG systems.

---

## Deliverables

- Reproducible **Python / PyTorch / scikit-learn** code repository.  
- **Pretrained and retrained models** for SAM40 and EEGMAT.  
- **Final report** including:
  - Quantitative performance tables.  
  - EEG channel heatmaps and brain topographies.  
  - Comparative analysis of model generalization.

---

## References

1. Sharma & Chopra (2020). *EEG signal analysis and detection of stress using classification techniques.* Journal of Information and Optimization Sciences.  
2. Ghosh et al. (2022). *SAM-40: Dataset of 40 subject EEG recordings to monitor stress.* Data in Brief 40:107772.  
3. Shikha et al. (2025). *Ensemble classifier for EEG-based stress classification.* Springer Nature.  
4. Goldberger et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation.  
5. Dorosti, S. (2023). [EEG Stress Classification](https://github.com/sarshardorosti/eeg-stress-classification).  
6. Waves Research Group (2023). [EEG Stress Detection](https://github.com/wavesresearch/eeg_stress_detection).  
7. Marthinsen et al. (2023). *Psychological stress detection with optimally selected EEG channels using machine learning techniques.* CEUR-WS.org.


