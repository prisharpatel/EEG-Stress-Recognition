
### 🔍 *Classifying mental states using EEG signals through Machine Learning and Deep Learning approaches.*

---

## 📘 Overview

This project focuses on classifying **mental states** using **EEG (Electroencephalogram)** data collected during **mental arithmetic tasks**.
The project follows a complete pipeline — from **signal preprocessing** to **feature extraction**, **machine learning**, and **deep learning** model comparison.

---

## 🧩 Project Highlights

✅ Comprehensive **EEG preprocessing pipeline** using MNE
✅ Extraction of **time**, **frequency**, and **time-frequency** domain features
✅ ML models like **SVM**, **Random Forest**, and **Logistic Regression**
✅ DL models like **1D-CNN**, **LSTM**, and **Hybrid CNN-LSTM**
✅ Visualization of performance metrics & comparisons

---

## 📁 Folder Structure

```bash
EEGMAT_dataset/
│
├── eeg_features.csv
├── subject-info.csv
│
├── eeg-during-mental-arithmetic-tasks-1.0.0/     # Raw EEG data (.edf)
│
├── processed_data/                               # Preprocessed EEG data (.fif)
│
├── EEG_CLEAN/                                    # Preprocessing & feature extraction scripts
│   ├── eeg_preprocessing.py
│   ├── feature_extraction.py
│
├── MODELS/                                       # ML & DL model scripts
│   ├── classification_model.py
│   ├── deep_learning_model.py
│   ├── final_dl_comparison.py
│
├── MODELS_WEIGHTS/                               # Trained model weights (.h5)
│
├── RESULTS/                                      # Visual results and performance plots
│   ├── COMPLETE_DL_ANALYSIS_RESULTS.png
│   ├── FINAL_DL_vs_ML_COMPARISON.png
│   ├── model_comparison_results.png
│
└── REPORT/
    └── EEG_Mental_State_Classification_Report.pdf
```

---

## ⚙️ Workflow Summary

### 🧹 **1. Preprocessing**

* Bandpass filter (1–45 Hz)
* Notch filter (50 Hz)
* ICA-based artifact removal
* Exported clean signals as `.fif`

### 🧠 **2. Feature Extraction**

Extracted:

* **Time-domain:** Mean, variance, skewness, kurtosis
* **Frequency-domain:** PSD (Delta, Theta, Alpha, Beta, Gamma)
* **Entropy features & band power ratios**
  Saved results to `eeg_features.csv`.

### 🤖 **3. Machine Learning Models**

Trained models:

* **Random Forest**
* **SVM**
* **Logistic Regression**

| Model               |   Accuracy   | Type |
| :------------------ | :----------: | :--: |
| Random Forest       | 🟩 **94.4%** |  ML  |
| SVM                 |     91.2%    |  ML  |
| Logistic Regression |     88.6%    |  ML  |

### 🧬 **4. Deep Learning Models**

Trained on preprocessed EEG sequences:

* 1D-CNN
* LSTM
* CNN-LSTM

| Model    |   Accuracy   | Type |
| :------- | :----------: | :--: |
| 1D-CNN   |     83.5%    |  DL  |
| LSTM     |     84.8%    |  DL  |
| CNN-LSTM | 🟦 **86.1%** |  DL  |

---

## 📈 Results & Insights

* **Random Forest** achieved the **highest overall accuracy**.
* **Deep Learning models** showed potential for larger EEG datasets.
* Visualized confusion matrices and learning curves for model comparison.

🖼️ *Example output plot:*

```
→ RESULTS/FINAL_DL_vs_ML_COMPARISON.png
```

---

## 🧰 Tech Stack

| Category          | Tools / Libraries         |
| ----------------- | ------------------------- |
| Language          | Python                    |
| Data Processing   | NumPy, Pandas             |
| Signal Processing | MNE, SciPy                |
| ML                | Scikit-learn              |
| DL                | TensorFlow / Keras        |
| Visualization     | Matplotlib, Seaborn       |
| Environment       | Jupyter Notebook, VS Code |

---

## 🖥️ Run the Project

### 1️⃣ **Create Environment**

```bash
conda create -n eeg python=3.10
conda activate eeg
pip install numpy pandas mne scipy scikit-learn tensorflow matplotlib seaborn
```

### 2️⃣ **Preprocess EEG Data**

```bash
python EEG_CLEAN/eeg_preprocessing.py
```

### 3️⃣ **Extract Features**

```bash
python EEG_CLEAN/feature_extraction.py
```

### 4️⃣ **Run ML Models**

```bash
python MODELS/classification_model.py
```

### 5️⃣ **Run Deep Learning Models**

```bash
python MODELS/deep_learning_model.py
```

### 6️⃣ **Compare Models**

```bash
python MODELS/final_dl_comparison.py
```

📂 *Outputs are automatically saved in `RESULTS/`.*

---

