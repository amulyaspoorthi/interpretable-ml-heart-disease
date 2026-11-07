# Interpretable Machine Learning for Heart Disease Prediction

Do SHAP and LIME agree on which features matter, and how stable are their explanations? This project trains three classifiers on the Cleveland heart disease dataset and compares SHAP and LIME feature rankings across models and random splits.

**Paper:** [link to paper, add when available]
**Author:** Amulya Spoorthi S., RV Institute of Technology and Management

## Dataset
UCI Heart Disease dataset, processed Cleveland subset: 303 patients, 13 features, binary target (0 = no disease, 1 = disease). The data is not included in this repository.
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ (file `processed.cleveland.data`)

## Methods
- Models: Logistic Regression, Random Forest (200 trees), XGBoost (default settings), no hyperparameter tuning
- Preprocessing: median imputation (and scaling for Logistic Regression) inside a scikit-learn pipeline to avoid data leakage
- Evaluation: 5-fold CV on an 80% training split, a held-out 61-patient test set, and 10x repeated 5-fold CV on all 303 patients
- Explanations: SHAP (linear and tree explainers) and LIME, with global importance defined as the mean absolute attribution
- Agreement and stability: Spearman rank correlation, top-5 overlap, and 5 random 80/20 splits

## Key results
- Repeated CV accuracy: Logistic Regression 0.835, Random Forest 0.831, XGBoost 0.801 (ROC-AUC 0.903, 0.902, 0.877)
- SHAP and LIME consistently ranked `ca`, `thal` and `cp` among the top features
- SHAP-LIME agreement across 5 splits (mean Spearman rho): Logistic Regression 0.736, Random Forest 0.924, XGBoost 0.810
- Mid-ranked features such as `sex` and `age` changed rank noticeably between models

## Repository contents
- `heart_disease_xai.ipynb`: full experiment code
- `outputs/`: result tables (CSV) and figures (PNG)

## How to run
1. Open the notebook in Kaggle or Google Colab and add the Cleveland heart disease data (or load it from the UCI link above).
2. Install the libraries: `pip install -r requirements.txt`
3. Run all cells. The random seeds are fixed, so results should reproduce.
4. Note: the notebook reads the data from a Kaggle input path. Change the `path` line if you run it elsewhere.

## Limitations
Small single-source dataset, only 5 splits for the stability analysis, no hyperparameter tuning, and no
