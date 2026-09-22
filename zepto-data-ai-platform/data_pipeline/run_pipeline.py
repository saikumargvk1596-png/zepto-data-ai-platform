from pathlib import Path
from scraper import scrape_books
from database import load_dataframe
from queries import execute_and_save
Path("data").mkdir(exist_ok=True)
df=scrape_books(3,60); df.to_csv("data/clean_books.csv",index=False)
load_dataframe(df); execute_and_save(); print("Pipeline completed successfully.")
