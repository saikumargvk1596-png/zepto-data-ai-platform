# Data Pipeline

Run `pip install -r requirements.txt` then `python run_pipeline.py`.

The pipeline scrapes at least 60 books across at least 3 categories, uses median imputation for numeric parse failures, converts with the required fixed `1 GBP = 105.50 INR`, loads a normalized two-table SQLite schema, runs the required SQL clauses plus JOIN, and compares `pd.read_sql` with `pd.merge`.
