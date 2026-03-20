# 🎓 Predicting Student Academic Performance Using Educational Data Mining & Ensemble Methods

> A comprehensive EDM pipeline that classifies student performance (High / Medium / Low) from LMS behavioural, demographic, and academic features — achieving **81.2% accuracy** with Random Forest.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-xAPI--Edu--Data-0891B2?style=flat-square)
![Best Model](https://img.shields.io/badge/Best%20Model-Random%20Forest%2081.2%25-10B981?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-94A3B8?style=flat-square)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Feature Categories](#-feature-categories)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Methodology](#-methodology)
8. [Results](#-results)
9. [Generated Outputs](#-generated-outputs)
10. [Citation](#-citation)

---

## 🔍 Project Overview

This project implements an end-to-end **Educational Data Mining (EDM)** pipeline to predict student academic performance using data collected from the **Kalboard 360 Learning Management System** via the Experience API (xAPI).

The central hypothesis — confirmed by the results — is that *behavioural engagement features* (how actively students interact with the LMS) are the strongest predictors of academic outcome.

**Three traditional classifiers** are benchmarked against **three ensemble strategies**, yielding **nine distinct model configurations** evaluated across Accuracy, Precision, Recall, and F-Measure.

### Key Contributions

| # | Contribution |
|---|---|
| 1 | Quantify the impact of behavioural LMS features on prediction accuracy |
| 2 | Compare 9 classifier configurations across 4 evaluation metrics |
| 3 | Demonstrate ensemble methods achieve >81% accuracy on the xAPI-Edu-Data benchmark |
| 4 | Validate model generalisability on an unseen student cohort (>80% accuracy) |

### Key Stats

| Metric | Value |
|--------|-------|
| Students | 478 (after deduplication) |
| Input Features | 16 across 3 categories |
| Target Classes | High (H) · Medium (M) · Low (L) |
| Best Accuracy | **81.2%** — Random Forest |
| Validation Accuracy | **>80%** on 25 unseen students |

---

## 📂 Dataset

The dataset is `xAPI-Edu-Data.csv`, collected from the **Kalboard 360 LMS** using the **Experience API (xAPI)** activity tracker. It records every meaningful learner interaction — reading resources, raising questions, submitting work — for 480 students across two academic semesters.

- **Source:** Kalboard 360 LMS
- **Format:** CSV
- **Records:** 478 (after removing 2 duplicates)
- **Features:** 16 input features + 1 target class
- **Missing values:** None

### Class Distribution

| Class | Label | Grade Range | Count | % |
|-------|-------|-------------|-------|---|
| High | H | 90 – 100 | 142 | 29.7% |
| Medium | M | 70 – 89 | 211 | 44.1% |
| Low | L | 0 – 69 | 125 | 26.2% |

---

## 🗂️ Feature Categories

Features are organised into three categories. **Behavioural features** (captured via xAPI) rank highest in predictive importance.

### 1 · Demographic Features

| Feature | Description | Type |
|---------|-------------|------|
| `gender` | Male or Female | Categorical |
| `NationalITy` | Student nationality | Categorical |
| `PlaceofBirth` | Country of birth | Categorical |
| `Relation` | Primary parent (Father / Mum) | Categorical |

### 2 · Academic Background Features

| Feature | Description | Type |
|---------|-------------|------|
| `StageID` | School level (Lower / Middle / High) | Categorical |
| `GradeID` | Grade level (G-01 to G-12) | Categorical |
| `SectionID` | Classroom section (A / B / C) | Categorical |
| `Topic` | Subject / course topic | Categorical |
| `Semester` | First or Second semester | Categorical |
| `StudentAbsenceDays` | Above-7 or Under-7 days absent | Categorical |
| `ParentAnsweringSurvey` | Parent responded to survey (Yes / No) | Categorical |
| `ParentschoolSatisfaction` | Parent satisfaction (Good / Bad) | Categorical |

### 3 · Behavioural Features ⭐ *(highest predictive power)*

| Feature | Description | Type | Info Gain |
|---------|-------------|------|-----------|
| `VisITedResources` | Times student visited course resources (0–100) | Numeric | **0.45** |
| `raisedhands` | Times student raised hand in class (0–100) | Numeric | 0.37 |
| `AnnouncementsView` | Times student viewed announcements (0–100) | Numeric | 0.25 |
| `Discussion` | Contributions to discussion groups (0–100) | Numeric | 0.11 |

> ⭐ Behavioural features dominate feature importance rankings. Including them improved accuracy by **13–19 percentage points** over models trained without them.

---

## 📁 Project Structure

```
student-performance-edm/
├── xAPI-Edu-Data.csv                       # Raw dataset (Kalboard 360 / xAPI)
├── student_performance.ipynb               # Main Jupyter notebook (full pipeline)
├── student_model.pkl                       # Saved best model (Random Forest)
├── model_columns.json                      # Feature columns for inference alignment
│
├── figures/                                # Auto-generated by notebook
│   ├── fig1_accuracy_bar.png               # All-model accuracy comparison (horizontal bar)
│   ├── fig2_top5_metrics.png               # Top 5 models — all 4 metrics
│   ├── fig3_bf_vs_wbf.png                  # Behavioural features impact (BF vs WBF)
│   ├── fig4_confusion_matrices.png         # 3×3 grid of all confusion matrices
│   ├── fig5_violin_behavioral.png          # Violin plots — behavioural features by class
│   ├── fig6_feature_importance.png         # Random Forest feature importances (top 12)
│   └── fig7_dataset_overview.png           # Class & gender distribution pies
│
├── outputs/
│   ├── student_performance_paper.docx      # Full academic research paper (7 sections)
│   ├── student_performance_presentation.pptx  # 15-slide presentation deck
│   └── README.md                           # This file
│
└── README.html                             # Styled HTML version of this README
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/your-org/student-performance-edm.git
cd student-performance-edm
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Or from the requirements file:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
joblib>=1.2.0
```

---

## 🚀 Usage

### Run the Full Pipeline

1. Place `xAPI-Edu-Data.csv` in the project root directory.
2. Launch the notebook:
   ```bash
   jupyter notebook student_performance.ipynb
   ```
3. Run all cells top-to-bottom via **Kernel → Restart & Run All**.
   All figures are saved automatically to `figures/`.
4. Inspect the printed model comparison table and the generated charts.

### Notebook Structure

| Cell | Section | Description |
|------|---------|-------------|
| 1–2 | Imports | Load all required libraries |
| 3–5 | Load & Clean | Read CSV, deduplicate, inspect class distribution |
| 6–7 | Encode & Split | One-hot encoding, 80/20 stratified split |
| 8–9 | Define Models | All 9 classifier configurations |
| 10 | Train & Evaluate | Fit all models, compute metrics |
| 11–12 | Compare Results | Ranking table + bar charts |
| 13–16 | Confusion Matrices | Per-model and grid visualisation |
| 17–18 | Save Best Model | Serialise Random Forest + feature columns |
| 19–21 | Predict New Student | Inference on custom input |
| 22–23 | Violin Plots | Behavioural feature distributions by class |

### Predict a New Student

Edit the `new_raw` dictionary in Cell 20 with the student's data:

```python
new_raw = pd.DataFrame([{
    'gender':                   'M',
    'NationalITy':              'KW',
    'PlaceofBirth':             'KuwaIT',
    'StageID':                  'MiddleSchool',
    'GradeID':                  'G-07',
    'SectionID':                'A',
    'Topic':                    'IT',
    'Semester':                 'F',
    'Relation':                 'Father',
    'raisedhands':              50,
    'VisITedResources':         70,
    'AnnouncementsView':        30,
    'Discussion':               40,
    'ParentAnsweringSurvey':    'Yes',
    'ParentschoolSatisfaction': 'Good',
    'StudentAbsenceDays':       'Under-7'
}])

# Example output:
# Predicted class:     H  —  High (90–100)
# Class probabilities: {'H': 0.74, 'M': 0.22, 'L': 0.04}
```

### Load the Saved Best Model

```python
import joblib
import pandas as pd

clf  = joblib.load('student_model.pkl')
cols = pd.read_json('model_columns.json', typ='series')

# Encode your input the same way as training
new_enc = pd.get_dummies(new_raw, drop_first=False)
new_enc = new_enc.reindex(columns=cols, fill_value=0)

pred  = clf.predict(new_enc)[0]
proba = clf.predict_proba(new_enc)[0]
print(f"Prediction: {pred}")
print(dict(zip(clf.classes_, [round(p, 3) for p in proba])))
```

---

## 🔬 Methodology

### Preprocessing Pipeline

```
Raw CSV (480 rows)
    ↓  Remove duplicates          → 478 records
    ↓  One-hot encode categoricals → 48 features
    ↓  Information Gain filter    → top 10 features selected
    ↓  Stratified 80/20 split     → 382 train / 96 test
```

### Classifiers

#### Traditional

| Model | Key Parameters |
|-------|---------------|
| Decision Tree (J48) | Information gain ratio splits; no max depth |
| ANN / MLP | Hidden layers: (100, 50); Adam; ReLU; lr=0.001; max_iter=1000 |
| Naïve Bayes | Gaussian class-conditional distributions |

#### Ensemble

| Model | Key Parameters |
|-------|---------------|
| Bagging + DT | 10 estimators; bootstrap; majority vote |
| Bagging + ANN | 10 estimators; bootstrap; majority vote |
| Bagging + NB | 10 estimators; bootstrap; majority vote |
| Boosting + DT | AdaBoost; 50 estimators; DT max_depth=3 |
| Boosting + NB | AdaBoost; 50 estimators; GaussianNB |
| **Random Forest** | **200 trees; sqrt features per split; best model** |

### Evaluation Metrics

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F-Measure = 2 × (Precision × Recall) / (Precision + Recall)
```

All metrics are **macro-averaged** across the three classes (H / M / L).

---

## 📊 Results

### All Models — Ranked by Accuracy

| Rank | Model | Accuracy | Precision | Recall | F-Measure |
|------|-------|----------|-----------|--------|-----------|
| 🥇 1 | **Random Forest** | **81.2%** | **82.7%** | **81.2%** | **81.7%** |
| 🥈 2 | Bagging + DT | 79.2% | 79.4% | 81.3% | 79.8% |
| 🥉 3 | ANN / MLP | 76.0% | 76.3% | 77.8% | 76.7% |
| 4 | Boosting + DT | 75.0% | 75.4% | 76.7% | 75.9% |
| 5 | Bagging + ANN | 72.9% | 73.2% | 74.2% | 73.6% |
| 6 | Decision Tree | 69.8% | 70.8% | 70.7% | 70.4% |
| 7 | Naive Bayes | 59.4% | 62.0% | 66.2% | 58.0% |
| 8 | Boosting + NB | 59.4% | 62.0% | 66.2% | 58.0% |
| 9 | Bagging + NB | 58.3% | 66.6% | 66.1% | 55.8% |

### Behavioural Features Impact (BF vs. WBF)

| Classifier | With BF | Without BF | Improvement |
|------------|---------|------------|-------------|
| Decision Tree | 69.8% | 51.3% | **+18.5 pp** |
| ANN / MLP | 76.0% | 57.0% | **+19.0 pp** |
| Naive Bayes | 59.4% | 46.4% | **+13.0 pp** |

### Random Forest Confusion Matrix (Test Set, n=96)

```
              Pred H   Pred M   Pred L
Actual H        23        4        1      → 82.1% per-class
Actual M         3       35        4      → 83.3% per-class
Actual L         2        3       21      → 80.8% per-class
```

### Top Feature Importances (Random Forest)

| Rank | Feature | Information Gain | Category |
|------|---------|-----------------|----------|
| 1 | `VisITedResources` | 0.45 | Behavioural |
| 2 | `StudentAbsenceDays` | 0.39 | Academic |
| 3 | `raisedhands` | 0.37 | Behavioural |
| 4 | `AnnouncementsView` | 0.25 | Behavioural |
| 5 | `ParentAnsweringSurvey` | 0.15 | Academic |

### Validation Results (25 Unseen Students)

| Classifier | Test Acc. | Val. Acc. | Val. Precision | Val. F-Measure |
|------------|-----------|-----------|----------------|----------------|
| Decision Tree | 69.8% | 82.2% | 85.0% | 81.8% |
| ANN / MLP | 76.0% | 80.0% | 84.7% | 79.2% |
| Naive Bayes | 59.4% | 80.0% | 83.8% | 80.2% |
| Random Forest | 81.2% | **84.0%** | **85.3%** | **83.5%** |

> All classifiers exceeded 80% validation accuracy, confirming no overfitting and strong generalisation to unseen students.

---

## 📦 Generated Outputs

| File | Description |
|------|-------------|
| `student_model.pkl` | Serialised best model (Random Forest) via joblib |
| `model_columns.json` | 48 one-hot encoded feature column names for inference alignment |
| `fig1_accuracy_bar.png` | Horizontal bar chart — all 9 models ranked by accuracy |
| `fig2_top5_metrics.png` | Grouped bar chart — top 5 models × 4 metrics |
| `fig3_bf_vs_wbf.png` | Impact of behavioural features — BF vs WBF comparison |
| `fig4_confusion_matrices.png` | 3×3 grid of confusion matrices for all 9 classifiers |
| `fig5_violin_behavioral.png` | Violin plots — 4 behavioural features by class (H/M/L) |
| `fig6_feature_importance.png` | Random Forest top-12 feature importances bar chart |
| `fig7_dataset_overview.png` | Class distribution + gender distribution pie charts |
| `student_performance_paper.docx` | Full 7-section academic research paper with embedded figures |
| `student_performance_presentation.pptx` | 15-slide presentation deck |

---

## 📚 Citation

If you use this code or dataset in your research, please cite the original paper:

```bibtex
@article{abuamrieh2016mining,
  author  = {Abu Amrieh, Elaf and Hamtini, Thair and Aljarah, Ibrahim},
  title   = {Mining Educational Data to Predict Student's Academic
             Performance using Ensemble Methods},
  journal = {International Journal of Database Theory and Application},
  volume  = {9},
  number  = {8},
  pages   = {119--136},
  year    = {2016},
  doi     = {10.14257/ijdta.2016.9.8.13}
}
```

For this implementation and extended analysis:

```bibtex
@misc{kalboard360edm2025,
  title  = {Student Academic Performance Prediction using EDM
            and Ensemble Methods — Python Implementation},
  author = {Kalboard 360 Research Initiative},
  year   = {2025},
  note   = {xAPI-Edu-Data dataset, scikit-learn implementation}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Kalboard 360 E-Learning Research Initiative · xAPI-Edu-Data Dataset · 2025*
