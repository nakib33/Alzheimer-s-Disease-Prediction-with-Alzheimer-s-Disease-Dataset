# 🧠 Alzheimer's Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Enabled-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

Alzheimer's disease is a progressive neurological disorder and the most common form of dementia, affecting millions of people worldwide. Early and accurate diagnosis is critical — it enables timely intervention, better patient care planning, and improved outcomes.

This project builds an end-to-end **binary classification pipeline** to predict whether a patient has Alzheimer's disease based on clinical, demographic, lifestyle, and cognitive assessment features. The goal is to compare a wide breadth of machine learning algorithms and identify the most effective model for this healthcare classification task.

---

## 🎯 Objectives

- Perform thorough exploratory data analysis (EDA) on a structured clinical dataset
- Apply feature engineering and scaling techniques to prepare data for modeling
- Train and evaluate **18 machine learning classifiers** under identical conditions
- Compare models using multiple performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Extract actionable insights from model performance for real-world clinical relevance

---

## 📁 Repository Structure

```
alzheimers-disease-prediction/
│
├── alzheimer_s_disease_prediction.ipynb   # Main Jupyter Notebook (full pipeline)
├── alzheimers_disease_data.csv            # Dataset (2,149 patients, 35 features)
└── README.md                              # Project documentation (this file)
```

---

## 📊 Dataset Description

**Source:** [Kaggle – Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

| Property | Detail |
|---|---|
| Total Records | 2,149 patients |
| Total Features | 35 columns (33 features + 1 target + 1 ID) |
| Target Variable | `Diagnosis` (0 = No Alzheimer's, 1 = Alzheimer's) |
| Class Distribution | ~63% Negative / ~37% Positive |
| Missing Values | None |

### Feature Categories

**Demographics**
- `Age`, `Gender`, `Ethnicity`, `EducationLevel`

**Lifestyle Factors**
- `BMI`, `Smoking`, `AlcoholConsumption`, `PhysicalActivity`, `DietQuality`, `SleepQuality`

**Medical History**
- `FamilyHistoryAlzheimers`, `CardiovascularDisease`, `Diabetes`, `Depression`, `HeadInjury`, `Hypertension`

**Clinical Measurements**
- `SystolicBP`, `DiastolicBP`, `CholesterolTotal`, `CholesterolLDL`, `CholesterolHDL`, `CholesterolTriglycerides`

**Cognitive & Functional Assessments**
- `MMSE` (Mini-Mental State Examination), `FunctionalAssessment`, `ADL` (Activities of Daily Living)

**Symptoms**
- `MemoryComplaints`, `BehavioralProblems`, `Confusion`, `Disorientation`, `PersonalityChanges`, `DifficultyCompletingTasks`, `Forgetfulness`

> **Dropped Columns:** `PatientID` (identifier) and `DoctorInCharge` (non-informative string) were removed before modeling.

---

## 🛠️ Tech Stack & Libraries

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `scipy` |
| Preprocessing | `sklearn.preprocessing` (MinMaxScaler, StandardScaler) |
| Machine Learning | `scikit-learn`, `xgboost`, `catboost` |
| Model Selection | `GridSearchCV`, `train_test_split` |
| Metrics | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score` |

---

## 🔄 Project Pipeline

```
Raw Data (CSV)
     │
     ▼
1. Data Loading & Inspection
     │   → df.info(), df.describe(), shape check
     ▼
2. Exploratory Data Analysis (EDA)
     │   → Categorical distribution bar charts (with % annotations)
     │   → Numerical distributions (Histogram + KDE + Rug plots)
     │   → Correlation heatmaps (Clustered, Triangular, Bubble)
     │   → Target class distribution (Pie chart)
     ▼
3. Data Preprocessing
     │   → Drop non-feature columns (PatientID, DoctorInCharge)
     │   → MinMax Normalization → StandardScaler Standardization
     ▼
4. Train / Test Split
     │   → 80% Train / 20% Test | random_state=40 | stratified shuffle
     ▼
5. Model Training & Evaluation (18 Models)
     │   → Classification Report (Precision, Recall, F1 per class)
     │   → Confusion Matrix
     │   → Accuracy, Precision, Recall, F1, ROC-AUC
     ▼
6. Model Comparison & Best Model Selection
```

---

## 🤖 Models Evaluated

The following 18 classifiers were trained and compared under identical data conditions:

| # | Model | Category |
|---|---|---|
| 1 | Decision Tree | Tree-based |
| 2 | Random Forest | Ensemble (Bagging) |
| 3 | Extra Trees | Ensemble (Bagging) |
| 4 | Bagging Classifier | Ensemble (Bagging) |
| 5 | Gradient Boosting | Ensemble (Boosting) |
| 6 | AdaBoost | Ensemble (Boosting) |
| 7 | XGBoost | Ensemble (Boosting) |
| 8 | K-Nearest Neighbors | Instance-based |
| 9 | Logistic Regression | Linear |
| 10 | Ridge Classifier | Linear |
| 11 | SGD Classifier | Linear |
| 12 | Support Vector Machine (SVC) | Kernel-based |
| 13 | Linear SVC | Linear / SVM |
| 14 | Gaussian Naive Bayes | Probabilistic |
| 15 | Bernoulli Naive Bayes | Probabilistic |
| 16 | Linear Discriminant Analysis | Dimensionality Reduction |
| 17 | Quadratic Discriminant Analysis | Dimensionality Reduction |
| 18 | MLP Classifier | Neural Network |

---

## 📈 Results Summary

The **Decision Tree** classifier achieved strong baseline performance, demonstrating that structured clinical data contains clear decision boundaries:

| Metric | Decision Tree |
|---|---|
| Accuracy | **93.49%** |
| Precision | **93.58%** |
| Recall | **93.02%** (Class 1) |
| F1-Score | **91%** (Class 1) |
| Confusion Matrix | TP: 147 / FP: 17 / FN: 11 / TN: 255 |

> Full comparative results for all 18 models are available in the notebook.

---

## 📉 Visualizations Included

The notebook features rich, multi-design visualizations:

- **Categorical Feature Distributions** — Gradient bar plots with count + percentage annotations per category (Gender, Smoking, Diabetes, etc.)
- **Numerical Feature Distributions** — Histograms with KDE overlay, rug plots, mean/median reference lines; 4-in-1 view (Histogram, KDE, Boxplot, Violin)
- **Correlation Analysis** — Three designs: Clustered heatmap with dendrogram, triangular heatmap with strong-correlation highlighting (|r| > 0.7), and bubble correlation matrix
- **Target Class Distribution** — Pie chart of Diagnosis balance (No Alzheimer's vs Alzheimer's)
- **Per-Model Reports** — Classification report + confusion matrix per classifier

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/alzheimers-disease-prediction.git
cd alzheimers-disease-prediction
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost scipy
```

### 3. Launch the Notebook
```bash
jupyter notebook alzheimer_s_disease_prediction.ipynb
```

Or run on **Kaggle** directly — the notebook was originally designed for the Kaggle environment.

---

## 💡 Key Takeaways & Insights

- **Cognitive and functional assessments** (MMSE, FunctionalAssessment, ADL) are among the strongest predictors of Alzheimer's diagnosis — even before applying ML, these clinical measurements show clear distributional separations between diagnosed and non-diagnosed patients.
- **Symptomatic features** such as `MemoryComplaints`, `Forgetfulness`, `Confusion`, and `BehavioralProblems` carry strong predictive signal.
- **Ensemble methods** (Random Forest, XGBoost, Extra Trees) generally outperform single classifiers in healthcare classification due to their robustness to feature noise.
- **Dual scaling** (MinMaxNormalization → StandardScaler) ensures that distance-based models (KNN, SVM) and gradient-based models (Logistic Regression, MLP) operate on properly normalized input.
- A **93%+ accuracy** on an 80/20 split with no hyperparameter tuning shows the strength of the raw feature set.

---

## 🔮 Future Work

- [ ] Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV` on top-performing models
- [ ] Cross-validation (k-fold) for more robust generalization estimates
- [ ] Feature importance analysis (SHAP values, permutation importance)
- [ ] Address class imbalance with SMOTE or class-weight adjustments
- [ ] Deploy best model as a REST API using Flask or FastAPI
- [ ] Add interactive dashboard using Streamlit or Gradio

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or raise an issue.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


> ⭐ If you found this project useful, please consider giving it a star on GitHub — it helps others discover it!
