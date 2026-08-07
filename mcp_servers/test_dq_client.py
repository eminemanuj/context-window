"""
mcp_servers/test_dq_client.py
--------------------------------
Standalone test: spins up the DQ server as a subprocess and calls its
tools as an MCP client would — this is the proof that the MCP wiring
actually works, not just that the server file exists.

Run this AFTER running data/load_to_sqlite.py at least once.
"""

import asyncio
from fastmcp import Client

SERVER_SCRIPT = "mcp_servers/dq_server.py"


async def main():
    client = Client(SERVER_SCRIPT)

    async with client:
        print("=== Connected to DQ MCP Server ===\n")

        result = await client.call_tool("count_rows", {"table": "sales"})
        print("count_rows:", result.content[0].text)

        result = await client.call_tool("check_nulls", {"table": "sales", "column": "category"})
        print("check_nulls:", result.content[0].text)

        result = await client.call_tool(
            "check_duplicates",
            {"table": "sales", "key_columns": "order_id,order_item_id"},
        )
        print("check_duplicates:", result.content[0].text)

        result = await client.call_tool(
            "check_freshness",
            {"table": "sales", "timestamp_column": "order_purchase_timestamp"},
        )
        print("check_freshness:", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())