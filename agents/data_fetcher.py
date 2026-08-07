"""
Agent 1: Data Fetcher
----------------------
Job: Load raw CSV data, join relevant tables, clean it, and validate it.
Hands off a clean DataFrame + a data quality report to the next agent.

Owner: [assign teammate name here]
"""

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class DataQualityReport:
    """Summary of what the Fetcher found/fixed while cleaning the data."""
    total_orders_loaded: int = 0
    total_order_items_loaded: int = 0
    rows_dropped_missing_price: int = 0
    rows_dropped_missing_date: int = 0
    rows_dropped_bad_status: int = 0
    duplicate_rows_removed: int = 0
    rows_dropped_incomplete_tail: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    categories_found: int = 0
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Data Quality Report ===",
            f"Orders loaded:            {self.total_orders_loaded}",
            f"Order items loaded:       {self.total_order_items_loaded}",
            f"Dropped (missing price):  {self.rows_dropped_missing_price}",
            f"Dropped (missing date):   {self.rows_dropped_missing_date}",
            f"Dropped (bad status):     {self.rows_dropped_bad_status}",
            f"Duplicates removed:       {self.duplicate_rows_removed}",
            f"Date range:               {self.date_range_start} to {self.date_range_end}",
            f"Categories found:         {self.categories_found}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def fetch_and_clean(
    orders_path: str,
    order_items_path: str,
    products_path: str,
    category_translation_path: str,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Load and join the core tables into one clean, analysis-ready DataFrame.

    Returns:
        (clean_df, quality_report)
    """
    report = DataQualityReport()

    # --- Load raw files ---
    orders = pd.read_csv(orders_path)
    order_items = pd.read_csv(order_items_path)
    products = pd.read_csv(products_path)
    category_translation = pd.read_csv(category_translation_path)

    report.total_orders_loaded = len(orders)
    report.total_order_items_loaded = len(order_items)

    # --- Clean orders: parse dates, keep only delivered orders ---
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    before = len(orders)
    orders = orders.dropna(subset=["order_purchase_timestamp"])
    report.rows_dropped_missing_date = before - len(orders)

    before = len(orders)
    valid_statuses = ["delivered", "shipped", "invoiced", "processing"]
    dropped_statuses = orders[~orders["order_status"].isin(valid_statuses)]["order_status"].unique().tolist()
    orders = orders[orders["order_status"].isin(valid_statuses)]
    report.rows_dropped_bad_status = before - len(orders)
    if dropped_statuses:
        report.warnings.append(f"Excluded order statuses: {dropped_statuses}")

    # --- Clean order_items: drop missing/invalid prices ---
    before = len(order_items)
    order_items = order_items.dropna(subset=["price"])
    order_items = order_items[order_items["price"] > 0]
    report.rows_dropped_missing_price = before - len(order_items)

    # --- Join: orders + order_items + products + category translation ---
    df = order_items.merge(orders, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(category_translation, on="product_category_name", how="left")

    # Fill missing English category names with the raw Portuguese name as fallback
    missing_translation = df["product_category_name_english"].isna().sum()
    if missing_translation > 0:
        report.warnings.append(
            f"{missing_translation} rows had no English category translation (kept original name)"
        )
    df["category"] = df["product_category_name_english"].fillna(
        df["product_category_name"]
    ).fillna("unknown")

    # --- Remove duplicates ---
    before = len(df)
    df = df.drop_duplicates(subset=["order_id", "order_item_id"])
    report.duplicate_rows_removed = before - len(df)

    # --- Drop incomplete trailing period ---
    # Bug found during testing: the dataset's final month has only a handful
    # of scattered orders (a few per day instead of dozens), which makes it
    # look like revenue "crashed" when really that month just isn't finished.
    # We detect this by finding the last month with reasonable order volume
    # and dropping anything after it.
    monthly_counts = df.groupby(
        df["order_purchase_timestamp"].dt.to_period("M")
    ).size()
    if len(monthly_counts) >= 2:
        avg_volume = monthly_counts.iloc[:-1].mean()
        last_month_volume = monthly_counts.iloc[-1]
        if last_month_volume < avg_volume * 0.5:  # last month has <50% of typical volume
            cutoff_period = monthly_counts.index[-2]  # keep up through the prior full month
            cutoff_date = cutoff_period.end_time
            before = len(df)
            df = df[df["order_purchase_timestamp"] <= cutoff_date]
            report.rows_dropped_incomplete_tail = before - len(df)
            report.warnings.append(
                f"Dropped {report.rows_dropped_incomplete_tail} rows from an incomplete "
                f"trailing period (last month had only {last_month_volume} orders vs. "
                f"~{avg_volume:.0f} average) to avoid distorting trend calculations"
            )

    # --- Derive useful columns for the Analyser ---
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["revenue"] = df["price"]  # per line item; sum for totals

    # --- Final report stats ---
    report.date_range_start = str(df["order_purchase_timestamp"].min().date())
    report.date_range_end = str(df["order_purchase_timestamp"].max().date())
    report.categories_found = df["category"].nunique()

    # --- Select clean, relevant columns only ---
    clean_df = df[[
        "order_id", "order_item_id", "product_id", "seller_id",
        "category", "price", "freight_value",
        "order_purchase_timestamp", "order_month", "order_status", "revenue",
    ]].reset_index(drop=True)

    return clean_df, report


if __name__ == "__main__":
    # Quick standalone test
    clean_df, report = fetch_and_clean(
        orders_path="data/raw/olist_orders_dataset.csv",
        order_items_path="data/raw/olist_order_items_dataset.csv",
        products_path="data/raw/olist_products_dataset.csv",
        category_translation_path="data/raw/product_category_name_translation.csv",
    )
    print(report.summary())
    print("\n--- Sample of clean data ---")
    print(clean_df.head())
    print(f"\nFinal shape: {clean_df.shape}")