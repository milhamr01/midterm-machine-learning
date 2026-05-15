# Midterm Machine Learning Repository

## Hands-On End-to-End Machine Learning Models

This repository contains the Machine Learning implementation for the Midterm Individual Task. The project focuses on building end-to-end machine learning pipelines using the provided datasets for three different tasks:

1. Fraud detection classification using `train_transaction.csv`
2. Regression using `midterm-regresi-dataset.csv`
3. Customer clustering using `clusteringmidterm.csv`

Each notebook includes clear explanations, code comments, preprocessing steps, model training, evaluation, and result interpretation.

---

## Student Identification

- **Name:** MUHAMMAD ILHAM RAYANDA
- **Class:** [TK-46-GAB]
- **NIM:** [1103223199]
- **Program:** S1 Teknik Komputer

---

## Repository Purpose

The purpose of this repository is to demonstrate practical knowledge of Machine Learning through complete end-to-end workflows. Each task is implemented in a separate Jupyter Notebook and covers the essential machine learning pipeline stages, including:

- Dataset loading
- Exploratory Data Analysis (EDA)
- Data cleaning
- Missing value handling
- Outlier handling when relevant
- Feature preparation or feature selection
- Data preprocessing
- Model training
- Hyperparameter tuning using Optuna
- Experiment tracking using MLflow
- Model evaluation using appropriate metrics
- Result interpretation and conclusion

---

## Repository Structure

```text
midterm-machine-learning/
├── 01_train_transaction.ipynb
├── 02_regression.ipynb
├── 03_clustering.ipynb
├── README.md
└── requirements.txt (optional)
```

> Note: The README.md file explains the whole Machine Learning repository, not each notebook separately.

---

## Notebook Overview

### 1. `01_train_transaction.ipynb` — Fraud Detection Classification

This notebook builds an end-to-end machine learning pipeline for online transaction fraud detection.

#### Dataset

- **File:** `train_transaction.csv`
- **Target column:** `isFraud`
- **Task type:** Binary classification
- **Goal:** Predict whether an online transaction is fraudulent or legitimate.

#### Main Workflow

- Load the transaction dataset from Google Drive
- Perform Exploratory Data Analysis
- Analyze missing values
- Clean and prepare features
- Handle class imbalance using `class_weight="balanced"`
- Apply preprocessing pipeline
- Train a scalable logistic-regression-style model
- Tune hyperparameters using Optuna
- Track experiment parameters and metrics using MLflow
- Evaluate the model using classification metrics

#### Model Used

- `SGDClassifier(loss="log_loss")`

This model is used because it is scalable for large datasets and represents a logistic-regression-style classifier suitable for binary classification.

#### Metrics Used

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

#### Key Interpretation

Fraud detection is an imbalanced classification problem, so accuracy alone is not sufficient. ROC-AUC and PR-AUC are used to evaluate how well the model separates fraudulent and legitimate transactions, while recall is important for measuring how many fraud cases are detected.

---

### 2. `02_regression.ipynb` — Regression Pipeline

This notebook builds an end-to-end regression pipeline to predict a continuous target value from numerical audio features.

#### Dataset

- **File:** `midterm-regresi-dataset.csv`
- **Target column:** First value in each row, renamed as `target_year`
- **Feature columns:** All remaining numerical values, renamed as `feature_1`, `feature_2`, and so on
- **Task type:** Regression
- **Goal:** Predict the release year of a song based on audio-related numerical features.

#### Main Workflow

- Load the dataset from Google Drive using `header=None`
- Validate dataset structure
- Rename columns into `target_year` and `feature_n`
- Perform EDA
- Handle missing target values
- Remove constant features
- Handle outliers using quantile clipping
- Apply median imputation and standard scaling
- Train a baseline Ridge Regression model
- Tune the Ridge regularization parameter using Optuna
- Track experiment using MLflow
- Evaluate the model using regression metrics
- Interpret one prediction using LIME

#### Model Used

- `Ridge Regression`

Ridge Regression is used because it is simple, stable, efficient, and suitable as a baseline model for numerical tabular regression data.

#### Metrics Used

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

#### Interpretation Method

- LIME (`LimeTabularExplainer`)

LIME is used to explain an individual prediction by showing which numerical features contribute most strongly to the predicted target year.

---

### 3. `03_clustering.ipynb` — Customer Clustering

This notebook builds an end-to-end customer clustering pipeline using credit card usage and payment behavior data.

#### Dataset

- **File:** `clusteringmidterm.csv`
- **Task type:** Unsupervised learning / clustering
- **Goal:** Group customers based on credit card spending, cash advance, payment behavior, balance, credit limit, and tenure.

#### Main Workflow

- Load the customer dataset from Google Drive
- Perform EDA
- Analyze missing values
- Remove identifier column `CUST_ID`
- Prepare numerical features
- Handle missing values using median imputation
- Handle outliers using quantile clipping
- Scale features using StandardScaler
- Choose the number of clusters using clustering evaluation metrics
- Tune the number of clusters using Optuna
- Train the final K-Means model
- Compare with Agglomerative Clustering and DBSCAN
- Visualize clusters using PCA
- Build cluster profiles
- Interpret each cluster based on customer behavior
- Track clustering metrics using MLflow

#### Models Used

- K-Means Clustering
- Agglomerative Clustering
- DBSCAN

K-Means is selected as the final model because it is simple, stable, and interpretable for customer segmentation.

#### Metrics Used

- Inertia
- Silhouette Score
- Davies-Bouldin Score
- Calinski-Harabasz Score

#### Cluster Interpretation

Clusters are interpreted by comparing average customer behavior across important features such as:

- `BALANCE`
- `PURCHASES`
- `CASH_ADVANCE`
- `CREDIT_LIMIT`
- `PAYMENTS`
- `MINIMUM_PAYMENTS`
- `PRC_FULL_PAYMENT`
- `TENURE`

The final interpretation explains what each cluster represents in terms of spending level, cash advance usage, payment behavior, and credit usage.

---

## Dataset Paths Used in Google Colab

The notebooks use the following dataset paths:

```python
/content/drive/MyDrive/train_transaction.csv
/content/drive/MyDrive/midterm-regresi-dataset.csv
/content/drive/MyDrive/clusteringmidterm.csv
```

Before running the notebooks, make sure all datasets are uploaded to the correct Google Drive location.

---

## How to Run the Notebooks

1. Open the notebook in Google Colab.
2. Mount Google Drive when prompted.
3. Make sure the required CSV dataset is available in `/content/drive/MyDrive/`.
4. Run all cells from top to bottom.
5. Check that all outputs are generated, including tables, plots, metrics, and interpretations.
6. If you are not using Colab Pro, it is recommended to disconnect and delete the runtime after finishing each notebook. Please run the notebooks one by one, and only move to the next notebook after the current one has finished running.

Recommended execution order:

```text
1. 01_train_transaction.ipynb
2. 02_regression.ipynb
3. 03_clustering.ipynb
```

---

## Main Libraries

The notebooks use the following main Python libraries:

```text
pandas
numpy
matplotlib
scikit-learn
optuna
mlflow
lime
```

If needed, install the dependencies manually using:

```bash
pip install pandas numpy matplotlib scikit-learn optuna mlflow lime
```

---

## Experiment Tracking

MLflow is used in the notebooks to track:

- Dataset information
- Model parameters
- Hyperparameters
- Evaluation metrics
- Trained model artifacts when applicable

MLflow runs are stored in the local `mlruns` directory during notebook execution.

---

## Notes on Model Selection

The models used in this repository are intentionally simple and stable because the datasets can be large and computationally expensive. The focus is on building complete, explainable, and reproducible machine learning workflows rather than directly jumping to complex models.

---

## Final Summary

This repository contains three complete Machine Learning pipelines:

1. **Fraud Detection Classification** using transaction data
2. **Regression** for predicting song release year
3. **Customer Clustering** using credit card behavior data

Each notebook follows the required end-to-end workflow, including preprocessing, model training, hyperparameter tuning with Optuna, evaluation, MLflow tracking, and interpretation.
