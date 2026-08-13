"""
mcp_servers/dq_client.py
---------------------------
Reusable helper that connects to the DQ MCP server and runs the standard
set of checks against a table. Used by graph.py to make an MCP tool call
part of the actual pipeline (not just a standalone test).

Includes a simple circuit breaker: if 2+ consecutive tool calls fail,
we stop calling further tools rather than hammering a server that's
clearly down, and surface that degradation clearly to the caller.
"""

from fastmcp import Client

SERVER_SCRIPT = "mcp_servers/dq_server.py"
CIRCUIT_BREAKER_THRESHOLD = 2  # consecutive failures before we stop calling


async def run_all_checks(table: str = "sales") -> dict:
    """Connects to the DQ MCP server and runs all 4 checks against a table.
    Returns a dict of tool_name -> result string. If the circuit breaker
    trips, remaining checks are marked as skipped rather than attempted."""
    client = Client(SERVER_SCRIPT)
    results = {}
    consecutive_failures = 0
    circuit_open = False

    checks = [
        ("count_rows", {"table": table}),
        ("check_nulls", {"table": table, "column": "category"}),
        ("check_duplicates", {"table": table, "key_columns": "order_id,order_item_id"}),
        ("check_freshness", {"table": table, "timestamp_column": "order_purchase_timestamp"}),
    ]

    try:
        async with client:
            for tool_name, args in checks:
                if circuit_open:
                    results[tool_name] = "SKIPPED — circuit breaker open (too many consecutive failures)"
                    continue

                try:
                    result = await client.call_tool(tool_name, args)
                    text = result.content[0].text
                    results[tool_name] = text

                    if text.startswith("ERROR:"):
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0  # reset on success

                except Exception as e:
                    consecutive_failures += 1
                    results[tool_name] = f"ERROR: tool call failed — {e}"

                if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                    circuit_open = True

    except Exception as e:
        # Couldn't even connect to the server at all
        for tool_name, _ in checks:
            if tool_name not in results:
                results[tool_name] = f"ERROR: could not connect to MCP server — {e}"

    return results