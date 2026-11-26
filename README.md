# Machine Learning Assignment 2 — Adult Census Income Classification

## 1. Project Overview

This project implements a binary classification workflow using the **Adult Census Income** dataset from the UCI Machine Learning Repository. The objective is to predict whether an individual's annual income falls into the `<=50K` or `>50K` category using demographic and employment-related attributes.

The notebook covers dataset selection and justification, data inspection and preprocessing, training and evaluation of five classification algorithms, model comparison, visualizations, and performance observations.

## 2. Dataset Details and Selection Justification

- **Dataset:** Adult (Census Income)
- **Repository:** UCI Machine Learning Repository
- **UCI Dataset ID:** `2`
- **Dataset URL:** https://archive.ics.uci.edu/dataset/2/adult
- **Notebook retrieval:** `fetch_ucirepo(id=2)` from the `ucimlrepo` package
- **Target column:** `income`
- **Problem type:** Binary classification
- **Target classes:** `<=50K` and `>50K`

The dataset was selected because it is a publicly available, real-world classification dataset and exceeds the assignment minimum of 1,000 instances and 12 input features. It includes both numerical and categorical attributes, missing values, duplicate records, and an imbalanced target distribution. These characteristics make it suitable for demonstrating data cleaning, preprocessing, and comparison of multiple classification algorithms.

### Dataset dimensions and features

The fetched dataset contains **48,842 records** and **14 input features** before duplicate removal. The notebook groups the input features as follows:

**Numerical features (6):**

- `age`
- `fnlwgt`
- `education-num`
- `capital-gain`
- `capital-loss`
- `hours-per-week`

**Categorical features (8):**

- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

The target variable is `income` and is not counted as an input feature.

## 3. Data Cleaning and Preprocessing

The notebook performs the following preprocessing steps:

1. **Data retrieval and formatting:** Fetches the dataset from UCI, trims whitespace from column names and string values, and cleans the target labels.
2. **Missing-value normalization:** Converts `?` markers and whitespace-only values to `NaN`. The initial feature data contains **6,465 missing values**.
3. **Duplicate removal:** Identifies and removes **52 duplicate records**, leaving **48,790 records**.
4. **Feature type handling:** Explicitly identifies six numerical and eight categorical input features and converts them to appropriate data types.
5. **Train-test split:** Uses an **80:20** split with stratification and `random_state=42`. The resulting sets contain **39,032 training records** and **9,758 test records**.
6. **Numerical preprocessing:** Uses median imputation followed by `StandardScaler`.
7. **Categorical preprocessing:** Uses most-frequent-value imputation followed by `OneHotEncoder(handle_unknown="ignore")`.
8. **Data leakage prevention:** Fits the preprocessing pipeline only on the training data and uses the fitted pipeline to transform the test data.

After preprocessing, the notebook reports **105 transformed features** and verifies that the processed training and test matrices contain no missing values.

## 4. Classification Algorithms

The notebook trains and compares these five classifiers using the same preprocessed training and held-out test data:

1. **Logistic Regression** — `max_iter=1000`, balanced class weights, `lbfgs` solver, and random state 42.
2. **Decision Tree** — balanced class weights and random state 42.
3. **k-Nearest Neighbors (kNN)** — 5 neighbors, distance-based weighting, and parallel processing.
4. **Gaussian Naive Bayes** — Gaussian Naive Bayes classifier. The notebook converts the one-hot encoded matrices to dense arrays for this model.
5. **Random Forest** — 200 estimators, balanced subsample class weights, random state 42, and parallel processing.

Class weighting is enabled for Logistic Regression, Decision Tree, and Random Forest to help address the class imbalance. kNN and Gaussian Naive Bayes are included as comparison baselines.

## 5. Evaluation Metrics

Each classifier is evaluated on the same held-out test set using the following metrics:

- **Accuracy**
- **ROC-AUC (AUC)**
- **Precision**
- **Recall**
- **F1-score**
- **Matthews Correlation Coefficient (MCC)**

The positive class for Precision, Recall, F1-score, and AUC is explicitly set to **`>50K`**. The notebook also generates a model comparison table, metric comparison charts, ROC curves, confusion matrices, and automatically generated model-performance observations.

### Best model in the recorded notebook execution

The notebook selects the overall model using the highest F1-score for the positive `>50K` class. In the recorded execution, **Logistic Regression** achieved the highest F1-score:

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 0.8082 |
| AUC | 0.9080 |
| Precision | 0.5660 |
| Recall | 0.8519 |
| F1-score | 0.6801 |
| MCC | 0.5744 |

These are the values recorded in the uploaded notebook execution. Re-running the notebook may produce different timings or results depending on the software environment and execution conditions.

## 6. Repository Structure

The GitHub repository should be organized as follows. Keep the assignment notebook and/or individual model implementation files inside the `model/` directory as required by the submission instructions.

```text
ML-Assignment-2/
│
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── Assignment2_Updated_with_Dataset_Justification.ipynb
    ├── logistic_regression.py       # if maintained as a separate implementation
    ├── decision_tree.py             # if maintained as a separate implementation
    ├── knn.py                       # if maintained as a separate implementation
    ├── naive_bayes.py               # if maintained as a separate implementation
    └── random_forest.py             # if maintained as a separate implementation
```

The notebook is the source of the complete preprocessing and five-model workflow. Add separate `.py` files if your submission requires each model implementation to be provided individually. Do not include placeholder files as though they contain completed implementations.

## 7. Test Data

The notebook exports `test_data.csv` from the held-out, stratified 20% test partition. It contains **9,758 rows and 15 columns**: the 14 original input features plus the true `income` target label. The CSV is exported before imputation and encoding so that it retains the original feature columns. The preprocessing pipeline is still fitted only on the training data.

## 8. Installation and Execution

### Prerequisites

- Python 3.9 or later is recommended.
- Jupyter Notebook or JupyterLab.
- Internet access when fetching the dataset from UCI for the first time.

### Install dependencies

From the repository root, run:

```bash
pip install -r requirements.txt
```

The notebook uses these Python packages:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
ucimlrepo
jupyter
```

### Run the notebook

1. Clone or download this repository.
2. Install the dependencies.
3. Open the notebook in `model/` using Jupyter Notebook, JupyterLab, or Google Colab.
4. Run all cells from top to bottom.
5. The notebook fetches the Adult dataset, performs preprocessing, trains the five classifiers, evaluates them, and generates result CSV files and visualizations.

The notebook generates the following evaluation files when the relevant cells are executed:

- `classification_model_results.csv`
- `classification_model_observations.csv`
- `classification_graph_observations.csv`
- `classification_model_final_report.csv`

It also exports `test_data.csv` from the held-out test split. If running in a hosted notebook environment, download the generated files and upload them to the appropriate location in the GitHub repository.

## 9. Reproducibility

The train-test split and supported classifiers use `random_state=42` where applicable. The same stratified test set is used to evaluate all five classifiers. The preprocessing pipeline is fitted on training data only to prevent information leakage from the test set.

## 10. Assignment Scope

This repository documents the dataset selection, preprocessing workflow, five required classification algorithms, and their evaluation using the six specified metrics. The model comparison and best-model statement are based on the notebook's recorded execution, not on an assumed result.
