import sqlite3
from pathlib import Path
import pandas as pd
DB_PATH=Path("data/zepto_books.db")

def create_schema():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(DB_PATH); conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
    DROP TABLE IF EXISTS books;
    DROP TABLE IF EXISTS categories;
    CREATE TABLE categories(category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT NOT NULL UNIQUE);
    CREATE TABLE books(book_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, price_gbp REAL NOT NULL,
      price_inr REAL NOT NULL, rating INTEGER NOT NULL, in_stock INTEGER NOT NULL, availability TEXT NOT NULL,
      category_id INTEGER NOT NULL, FOREIGN KEY(category_id) REFERENCES categories(category_id));
    """)
    conn.commit(); return conn

def load_dataframe(df):
    conn=create_schema()
    pd.DataFrame({"category_name":sorted(df.category.unique())}).to_sql("categories",conn,if_exists="append",index=False)
    cmap=pd.read_sql("SELECT category_id,category_name FROM categories",conn)
    m=df.merge(cmap,left_on="category",right_on="category_name")
    books=m[["title","price_gbp","price_inr","rating","in_stock","availability","category_id"]].copy()
    books["in_stock"]=books.in_stock.astype(int)
    books.to_sql("books",conn,if_exists="append",index=False)
    conn.commit(); conn.close()
