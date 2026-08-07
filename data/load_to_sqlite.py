"""
data/load_to_sqlite.py
------------------------
Loads the clean, joined Olist data into a SQLite database so our FastMCP
data-quality server (mcp_servers/dq_server.py) has real "tables" to run
checks against.
"""

import sys
sys.path.insert(0, ".")

import sqlite3
from agents.data_fetcher import fetch_and_clean

DB_PATH = "data/context_window.db"


def load():
    clean_df, quality_report = fetch_and_clean(
        orders_path="data/raw/olist_orders_dataset.csv",
        order_items_path="data/raw/olist_order_items_dataset.csv",
        products_path="data/raw/olist_products_dataset.csv",
        category_translation_path="data/raw/product_category_name_translation.csv",
    )

    conn = sqlite3.connect(DB_PATH)
    clean_df.to_sql("sales", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Loaded {len(clean_df)} rows into {DB_PATH}, table 'sales'.")
    print(quality_report.summary())


if __name__ == "__main__":
    load()
