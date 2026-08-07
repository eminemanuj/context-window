"""
mcp_servers/dq_server.py — Data Quality MCP Server
------------------------------------------------------
Same tool pattern learned in the DataPilot cohort workshop
(check_nulls, check_duplicates, check_freshness, count_rows),
rebuilt here as our own FastMCP server pointed at this project's
SQLite database (data/context_window.db).

This is what lets the Data Fetcher agent's quality checks be real
MCP tool calls instead of plain pandas code — the architectural
piece that matches "MCP-native" design from the workshop.

Run standalone to test:
    python mcp_servers/dq_server.py

Owner: [assign teammate name here]
"""

import sqlite3
from fastmcp import FastMCP

DB_PATH = "data/context_window.db"

mcp = FastMCP("data-quality-server")


def _connect():
    return sqlite3.connect(DB_PATH)


@mcp.tool()
def count_rows(table: str) -> str:
    """Return the row count of the given table."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    conn.close()
    return f"{table}: {count} rows"


@mcp.tool()
def check_nulls(table: str, column: str) -> str:
    """Count NULL values in a given column of a table."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
    null_count = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    total = cur.fetchone()[0]
    conn.close()
    pct = (null_count / total * 100) if total > 0 else 0
    return f"{table}.{column}: {null_count} nulls out of {total} rows ({pct:.2f}%)"


@mcp.tool()
def check_duplicates(table: str, key_columns: str) -> str:
    """
    Count duplicate rows in a table keyed on comma-separated key_columns.
    Example: key_columns="order_id,order_item_id"
    """
    conn = _connect()
    cur = conn.cursor()
    cols = key_columns
    cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT {cols}, COUNT(*) as cnt
            FROM {table}
            GROUP BY {cols}
            HAVING cnt > 1
        )
    """)
    dup_groups = cur.fetchone()[0]
    conn.close()
    return f"{table} keyed on ({cols}): {dup_groups} duplicate key group(s) found"


@mcp.tool()
def check_freshness(table: str, timestamp_column: str = "order_purchase_timestamp") -> str:
    """How fresh is the data in this table, based on its timestamp column?"""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT MIN({timestamp_column}), MAX({timestamp_column}) FROM {table}")
    min_date, max_date = cur.fetchone()
    conn.close()
    return f"{table}.{timestamp_column}: ranges from {min_date} to {max_date}"


if __name__ == "__main__":
    mcp.run()