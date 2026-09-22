# Zepto Data & AI Platform

Single repository containing the three required modules: `data_pipeline`, `analytics`, and `support_assistant`.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

## Module 1
```bash
cd data_pipeline
python run_pipeline.py
```
Uses the required fixed conversion `1 GBP = 105.50 INR` and creates the SQLite database and query outputs.

## Module 2
Run `python 01_eda.py` from `analytics` once with internet access. It loads `sns.load_dataset("titanic")` once and creates `titanic.csv`. Commit that CSV. Then run `python 02_modeling.py`.

Outputs include EDA plots, classification metrics, imbalance comparison, GridSearchCV/OOB results, regression metrics, residual plot, and `models/best_pipeline.joblib`.

## Module 3
```bash
cd support_assistant
python ingest.py
uvicorn main:app --reload
```
The graded default is `MOCK_LLM=1`/unset and requires no API key.

Docker:
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

## Required Git history
```bash
git init
git add .
git commit -m "Initial project structure"
git checkout -b feature/analytics
git add analytics/
git commit -m "Add analytics pipeline"
git add support_assistant/
git commit -m "Add support assistant"
git checkout main
git merge --no-ff feature/analytics -m "Merge project modules"
git log --graph --all --oneline
```
