"""
mcp_servers/dq_client.py
---------------------------
Reusable helper that connects to the DQ MCP server and runs the standard
set of checks against a table. Used by graph.py to make an MCP tool call
part of the actual pipeline (not just a standalone test).
"""

from fastmcp import Client

SERVER_SCRIPT = "mcp_servers/dq_server.py"


async def run_all_checks(table: str = "sales") -> dict:
    """Connects to the DQ MCP server and runs all 4 checks against a table.
    Returns a dict of tool_name -> result string."""
    client = Client(SERVER_SCRIPT)
    results = {}

    async with client:
        result = await client.call_tool("count_rows", {"table": table})
        results["count_rows"] = result.content[0].text

        result = await client.call_tool("check_nulls", {"table": table, "column": "category"})
        results["check_nulls"] = result.content[0].text

        result = await client.call_tool(
            "check_duplicates",
            {"table": table, "key_columns": "order_id,order_item_id"},
        )
        results["check_duplicates"] = result.content[0].text

        result = await client.call_tool(
            "check_freshness",
            {"table": table, "timestamp_column": "order_purchase_timestamp"},
        )
        results["check_freshness"] = result.content[0].text

    return results