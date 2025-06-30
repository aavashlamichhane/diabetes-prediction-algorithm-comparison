# Diabetes Prediction Algorithm Comparison

## Overview

This project compares various machine learning algorithms for predicting hospital readmission within 30 days for diabetic patients, using the [UCI Diabetes 130-US hospitals dataset (1999–2008)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008). The workflow includes data collection, preprocessing, exploratory analysis, supervised and unsupervised learning, model evaluation, and visualization.

## Project Structure

- `data-collection.py`: Downloads the raw dataset from UCI ML Repo.
- `data-preprocessing.ipynb`: Cleans, encodes, and engineers features for modeling.
- `data-exploration.ipynb`: Basic data exploration and understanding.
- `data-plots.ipynb`: Visualizations for feature relationships and outcome analysis.
- `supervised-learning.ipynb`: Trains and compares multiple supervised ML models (Logistic Regression, Decision Tree, Random Forest, KNN, Gradient Boosting, MLP, XGBoost, LightGBM).
- `ensemble-learning.ipynb`: Builds and evaluates ensemble models (Voting, Stacking).
- `unsupervised-learning.ipynb`: Clusters patients using K-Means and analyzes cluster profiles.
- `models/`: Saved best models for each algorithm.
- `plots/`: Feature importance and ROC curve plots for each model.
- `output.txt`: Model evaluation results and metrics.
- `requirements.txt` / `environment.yml`: Python dependencies and environment setup.

## Workflow

1. **Data Collection**
   - Run `data-collection.py` to fetch the dataset and save as `diabetes_data.csv`.

2. **Data Preprocessing**
   - Clean missing/invalid values, drop irrelevant columns, encode categorical variables, engineer features (e.g., comorbidity score, prior utilization, hba1c_attention).
   - Save the processed data as `processed_diabetes_data.csv`.

3. **Exploratory Data Analysis**
   - Use `data-exploration.ipynb` and `data-plots.ipynb` for summary statistics and visualizations (demographics, readmission rates, feature distributions).

4. **Supervised Learning**
   - Train/test split, SMOTE balancing, feature scaling, and one-hot encoding.
   - Hyperparameter tuning and evaluation for:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - KNN
     - Gradient Boosting
     - MLP Neural Network
     - XGBoost
     - LightGBM
   - Metrics: Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix.
   - Save best models and plots to `models/` and `plots/`.

5. **Ensemble Learning**
   - Build Voting and Stacking ensembles using top-performing models.
   - Evaluate and compare ensemble performance.

6. **Unsupervised Learning**
   - K-Means clustering on clinical/utilization features.
   - Visualize clusters (PCA, radar, heatmap) and analyze readmission rates by cluster.

## Results

- All model metrics and ROC curves are saved in `output.txt` and `plots/`.
- Ensemble models (Voting, Stacking) generally outperform individual models.
- Feature importance plots highlight key predictors (e.g., prior utilization, comorbidity score, A1C result).
- Unsupervised clusters reveal patient profiles with distinct readmission risks.

## Setup & Requirements

- Python 3.12+
- Recommended: Create a conda environment using `environment.yml`:
  ```sh
  conda env create -f environment.yml
  conda activate assign-dm
  ```
- Or install dependencies with pip:
  ```sh
  pip install -r requirements.txt
  ```

## Usage

1. Download the dataset:
   ```sh
   python data-collection.py
   ```
2. Run preprocessing and analysis notebooks in order:
   - `data-preprocessing.ipynb`
   - `data-exploration.ipynb`
   - `data-plots.ipynb`
   - `supervised-learning.ipynb`
   - `ensemble-learning.ipynb`
   - `unsupervised-learning.ipynb`

## Authors
- Prayash
- Aavash

## References
- [UCI ML Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- [Scikit-learn](https://scikit-learn.org/)
- [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
