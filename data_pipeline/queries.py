import sqlite3
from pathlib import Path
import pandas as pd
DB_PATH=Path("data/zepto_books.db"); OUT=Path("data/query_results")
QUERIES={
"01_select_where":"SELECT title,price_gbp,rating FROM books WHERE rating>=4 ORDER BY rating DESC",
"02_order_by":"SELECT title,price_inr FROM books ORDER BY price_inr DESC",
"03_limit":"SELECT title,rating FROM books LIMIT 10",
"04_distinct":"SELECT DISTINCT category_name FROM categories ORDER BY category_name",
"05_between":"SELECT title,price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30 ORDER BY price_gbp",
"06_join":"SELECT b.title,b.rating,c.category_name FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.rating DESC,b.title LIMIT 10"}

def execute_and_save():
    OUT.mkdir(parents=True,exist_ok=True); conn=sqlite3.connect(DB_PATH); results={}
    for name,sql in QUERIES.items():
        print("\\n",name,"\\n",sql); df=pd.read_sql(sql,conn); results[name]=df; print(df.to_string(index=False))
        (OUT/f"{name}.sql").write_text(sql,encoding="utf-8"); df.to_csv(OUT/f"{name}.csv",index=False)
    sql_join=pd.read_sql(QUERIES["06_join"],conn).reset_index(drop=True)
    books=pd.read_sql("SELECT * FROM books",conn); cats=pd.read_sql("SELECT * FROM categories",conn)
    pandas_join=books.merge(cats,on="category_id")[["title","rating","category_name"]].sort_values(["rating","title"],ascending=[False,True]).head(10).reset_index(drop=True)
    print("\\nEquivalent:",sql_join.equals(pandas_join)); sql_join.to_csv(OUT/"join_pd_read_sql.csv",index=False); pandas_join.to_csv(OUT/"join_pd_merge.csv",index=False)
    conn.close(); return results
if __name__=="__main__": execute_and_save()
