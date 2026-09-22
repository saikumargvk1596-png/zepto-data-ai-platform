# Analytics Pipeline

Run `python 01_eda.py` with internet access the first time. It calls `sns.load_dataset("titanic")` once and creates `titanic.csv`; commit that CSV. Then run `python 02_modeling.py`.

The modeling stage performs a stratified split before preprocessing, uses a train-only `ColumnTransformer`, trains Logistic Regression/Decision Tree/Random Forest, evaluates the required metrics, compares baseline/class-weight/SMOTE, tunes Random Forest with OOB, performs fare regression, creates plots, and saves a complete end-to-end `joblib` pipeline.
