# 🤖 Applied Machine Learning Portfolio: From Foundational Algorithms to Advanced Imbalanced Learning

Welcome to the **Applied Machine Learning Projects Suite**. This repository contains five comprehensive, end-to-end machine learning projects designed and implemented using Python, Scikit-Learn, and Jupyter Notebooks.

Each project walks through the complete data science and machine learning lifecycle:
1. **Exploratory Data Analysis (EDA) & Data Storytelling**
2. **Data Cleaning & Missing Value Imputation**
3. **Feature Engineering, Encoding (One-Hot / Label), and Scaling (MinMax / Standard Scaler)**
4. **Addressing Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)**
5. **Model Benchmarking across Classification and Regression tasks**
6. **In-depth Performance Evaluation & Business / Clinical Interpretations**

---

## 📂 Projects Overview

| # | Project Name | Domain | Core Challenge / Focus | Top Models & Key Metrics |
|---|--------------|--------|------------------------|---------------------------|
| **01** | **[Adult Census Income](./01-Adult-Census-Income)** | Socio-Economic / Census | Binary Income Classification (`<=50K` vs `>50K`) + Linear Probability Modeling | **KNN (82.81%)**, **LinearSVC (82.31%)**, **Logistic Regression (82.28%)** |
| **02** | **[Bank Marketing Campaign](./02-Bank-Marketing)** | Finance / Direct Marketing | Predicting Term Deposit Subscriptions | **SVM (80.85%)**, **Logistic Regression (79.28%)**, **KNN (76.53%)** |
| **03** | **[House Rent Prediction](./03-House-Rent-Prediction)** | Real Estate | Dual Problem: Continuous Rent Valuation vs Price Tier Categorization | **Linear Regression**, **SVR**, **KNN Regressor / Classifier** |
| **04** | **[Student Academic Performance](./04-Student-Performance)** | Education Analytics | Dual Problem: Continuous Index Prediction vs Binary Pass/Fail Decision | **Linear Regression (R² = 98.7%, RMSE = 2.15)**, **SVM (97.8% Accuracy)** |
| **05** | **[Stroke Prediction & SMOTE Balancing](./05-Stroke-Prediction)** | Healthcare / Clinical AI | Resolving the **Accuracy Paradox** on Heavily Imbalanced Clinical Data | **SMOTE Resampling: Recall boosted from 0% → 80%+**, **Logistic Regression**, **SVM** |

---

## 🔍 Deep-Dive into the Projects

### 1️⃣ [Stroke Prediction & Class Imbalance Handling](./05-Stroke-Prediction)
- **Problem**: Detecting early stroke risks from patient physiological attributes (glucose, BMI, hypertension, heart disease).
- **The Core Challenge — The Accuracy Paradox**: Prior to handling class imbalance, baseline models exhibited deceptive **~95.1% accuracy** while scoring a **0% Recall** on stroke-positive patients (missing every single patient at risk!).
- **The Solution**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly on the training partition.
- **The Outcome**: Model sensitivity (Recall) escalated from **0% to over 80%**, creating a clinically viable screening tool prioritizing the minimization of False Negatives.

### 2️⃣ [Student Academic Performance Analytics](./04-Student-Performance)
- **Dual Formulation**: 
  - **Regression**: Predicting exact continuous `Performance Index` based on study hours, previous scores, extracurriculars, sleep, and practice questions.
  - **Classification**: Categorizing students into Pass / Fail tiers.
- **Results**:
  - **Linear Regression** demonstrated outstanding fit: **$R^2 \approx 98.7\%$** and **RMSE = 2.15**, validating strong linear relationships.
  - **SVM (SVC)** and **Logistic Regression** achieved **~97.8% accuracy** on student classification.

### 3️⃣ [Adult Census Income Prediction](./01-Adult-Census-Income)
- **Problem**: Predicting whether individual annual earnings exceed \$50,000 based on census demographics (education, age, occupation, capital gain/loss).
- **Comparative Analysis**: Explored both non-parametric (**KNN**) and parametric (**Logistic Regression**, **LinearSVC**, **GaussianNB**) classifiers.
- **Linear Probability Modeling (LPM)**: Tested Ordinary Least Squares on binary outcome; thresholding at 0.5 reproduced **82.1% accuracy**, matching the logistic regression benchmark.

### 4️⃣ [Bank Direct Marketing Campaign](./02-Bank-Marketing)
- **Problem**: Predicting customer propensity to subscribe to long-term bank deposits from telemarketing campaigns.
- **Key Focus**: Importance of **StandardScaler** pipelines when comparing distance-based algorithms (**KNN**, **SVM**) against probabilistic benchmarks (**Naive Bayes**).
- **Results**: **SVM** achieved the highest accuracy (**80.85%**), demonstrating strong capacity for non-linear margin separation.

### 5️⃣ [House Rent Valuation & Categorization](./03-House-Rent-Prediction)
- **Problem**: Modeling rental pricing dynamics across Indian metropolitan areas based on size, furnishing status, floor level, and locality.
- **Tasks**:
  - Continuous valuation via **Linear Regression**, **KNN Regressor**, and **Support Vector Regressor (SVR)**.
  - Rental category tier classification via **Logistic Regression**, **KNN Classifier**, **Naive Bayes**, and **SVC**.

---

## 🛠️ Tech Stack & Algorithms Implemented

- **Languages & Frameworks**: Python 3, Jupyter Notebook
- **Core Libraries**: NumPy, Pandas, Scikit-Learn, Imbalanced-Learn (`imblearn.over_sampling.SMOTE`), Matplotlib, Seaborn
- **Algorithms Evaluated**:
  - **Regression**: Ordinary Least Squares (Linear Regression), K-Nearest Neighbors Regressor (KNN), Support Vector Regressor (SVR)
  - **Classification**: Logistic Regression, K-Nearest Neighbors Classifier (KNN), Support Vector Classifier (SVC / LinearSVC), Gaussian Naive Bayes
  - **Resampling**: SMOTE (Synthetic Minority Over-sampling Technique)
  - **Feature Preprocessing**: `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `LabelEncoder`, `SimpleImputer`

---

## 📁 Repository Structure

```text
Applied-Machine-Learning-Projects/
│
├── 01-Adult-Census-Income/
│   ├── Adult Census Income.csv
│   └── Adult_Census_Income.ipynb
│
├── 02-Bank-Marketing/
│   ├── bank.csv
│   └── bank.ipynb
│
├── 03-House-Rent-Prediction/
│   ├── House_Rent_Dataset.csv
│   └── house_rent_ml_models.ipynb
│
├── 04-Student-Performance/
│   ├── Student_Performance.csv
│   └── Student_Performance_New.ipynb
│
├── 05-Stroke-Prediction/
│   ├── healthcare-dataset-stroke-data.csv
│   ├── stroke_processed.csv
│   └── Stroke_Classification_Balanced.ipynb
│
├── .gitignore
└── README.md
```

---

## 👤 Author

**Naser Ashraf**  
- **GitHub**: [@naserashraf-alt](https://github.com/naserashraf-alt)
- **Project Repository**: [Applied-Machine-Learning-Projects](https://github.com/naserashraf-alt/Applied-Machine-Learning-Projects)
