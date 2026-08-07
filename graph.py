"""
graph.py — LangGraph Orchestrator
-----------------------------------
Wires the 3 agents into a single pipeline:

    Data Fetcher -> MCP Verify -> Insight Analyser -> [conditional] -> Report Writer
                                                             |
                                                    (if no findings) -> No-Findings Report

This is the "how do agents communicate" piece of the demo — state is
passed explicitly between nodes via a shared GraphState object, and you
can print/inspect that state at each step to show the handoff live.

Owner: [assign teammate name here]
"""

import sys
sys.path.insert(0, ".")

import sqlite3
import asyncio
from typing import TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph, END

from agents.data_fetcher import fetch_and_clean, DataQualityReport
from agents.insight_analyser import analyse
from agents.report_writer import write_report
from mcp_servers.dq_client import run_all_checks
from memory.memory_store import save_report, get_most_recent_report
DB_PATH = "data/context_window.db"


# --- Shared state passed between every node in the graph ---
class GraphState(TypedDict):
    # Inputs (set before the graph runs)
    orders_path: str
    order_items_path: str
    products_path: str
    category_translation_path: str

    # Populated by Data Fetcher
    clean_df: Optional[pd.DataFrame]
    quality_report: Optional[DataQualityReport]

    # Populated by MCP Verification step
    mcp_checks: Optional[dict]

    # Populated by Insight Analyser
    findings: Optional[list]

    # Populated by Report Writer
    report: Optional[str]
    validation: Optional[dict]


# --- Node 1: Data Fetcher ---
def fetcher_node(state: GraphState) -> GraphState:
    print("[Data Fetcher] Loading and cleaning data...")
    clean_df, quality_report = fetch_and_clean(
        orders_path=state["orders_path"],
        order_items_path=state["order_items_path"],
        products_path=state["products_path"],
        category_translation_path=state["category_translation_path"],
    )
    print(f"[Data Fetcher] Done. {len(clean_df)} clean rows handed off.")
    return {**state, "clean_df": clean_df, "quality_report": quality_report}


# --- Node 1.5: MCP Verification — real MCP tool calls against SQLite ---
async def mcp_verify_node(state: GraphState) -> GraphState:
    print("[MCP Verify] Loading clean data into SQLite for MCP server...")
    conn = sqlite3.connect(DB_PATH)
    state["clean_df"].to_sql("sales", conn, if_exists="replace", index=False)
    conn.close()

    print("[MCP Verify] Calling DQ MCP server tools...")
    mcp_checks = await run_all_checks(table="sales")
    for tool_name, result in mcp_checks.items():
        print(f"[MCP Verify]   {tool_name}: {result}")

    return {**state, "mcp_checks": mcp_checks}


# --- Node 2: Insight Analyser ---
def analyser_node(state: GraphState) -> GraphState:
    print("[Insight Analyser] Analysing clean data for trends/anomalies...")
    findings = analyse(state["clean_df"])
    print(f"[Insight Analyser] Done. {len(findings)} findings handed off.")
    return {**state, "findings": findings}


# --- Conditional routing: skip straight to a short report if nothing found ---
def route_after_analysis(state: GraphState) -> str:
    if not state["findings"]:
        return "no_findings"
    return "has_findings"


# --- Node 3a: Report Writer (normal path, findings exist) ---
def writer_node(state: GraphState) -> GraphState:
    print("[Report Writer] Checking memory for last week's report...")
    previous = get_most_recent_report()
    previous_context = previous["report"] if previous else None
    if previous:
        print(f"[Report Writer] Found previous report from {previous['run_date']}")
    else:
        print("[Report Writer] No previous report in memory (first run).")

    print("[Report Writer] Generating report from findings...")
    result = write_report(state["findings"], previous_context=previous_context)
    print(f"[Report Writer] Done. Guardrail passed: {result['validation']['passed']}")

    print("[Report Writer] Saving this report to memory...")
    save_report(result["report"], state["findings"])

    return {**state, "report": result["report"], "validation": result["validation"]}


# --- Node 3b: No-findings path (guardrail — don't force the LLM to invent content) ---
def no_findings_node(state: GraphState) -> GraphState:
    print("[Report Writer] No significant findings — skipping LLM call.")
    report_text = (
        "## Weekly Business Report\n\n"
        "### Summary\n"
        "No significant changes or anomalies were detected this period. "
        "All metrics remained within normal historical ranges.\n"
    )
    return {**state, "report": report_text, "validation": {"passed": True, "suspicious_numbers": []}}


def build_graph():
    """Constructs and compiles the LangGraph pipeline."""
    graph = StateGraph(GraphState)

    graph.add_node("fetcher", fetcher_node)
    graph.add_node("mcp_verify", mcp_verify_node)
    graph.add_node("analyser", analyser_node)
    graph.add_node("writer", writer_node)
    graph.add_node("no_findings", no_findings_node)

    graph.set_entry_point("fetcher")
    graph.add_edge("fetcher", "mcp_verify")
    graph.add_edge("mcp_verify", "analyser")

    # Conditional edge — the actual guardrail logic
    graph.add_conditional_edges(
        "analyser",
        route_after_analysis,
        {
            "has_findings": "writer",
            "no_findings": "no_findings",
        },
    )

    graph.add_edge("writer", END)
    graph.add_edge("no_findings", END)

    return graph.compile()


if __name__ == "__main__":
    async def main():
        app = build_graph()

        initial_state = {
            "orders_path": "data/raw/olist_orders_dataset.csv",
            "order_items_path": "data/raw/olist_order_items_dataset.csv",
            "products_path": "data/raw/olist_products_dataset.csv",
            "category_translation_path": "data/raw/product_category_name_translation.csv",
            "clean_df": None,
            "quality_report": None,
            "mcp_checks": None,
            "findings": None,
            "report": None,
            "validation": None,
        }

        print("=" * 50)
        print("RUNNING THE CONTEXT WINDOW PIPELINE")
        print("=" * 50)

        final_state = await app.ainvoke(initial_state)

        print("\n" + "=" * 50)
        print("FINAL REPORT")
        print("=" * 50)
        print(final_state["report"])

        print("\n" + "=" * 50)
        print("DATA QUALITY REPORT (pandas-based, from Data Fetcher)")
        print("=" * 50)
        print(final_state["quality_report"].summary())

        print("\n" + "=" * 50)
        print("MCP-VERIFIED QUALITY CHECKS (from DQ MCP Server)")
        print("=" * 50)
        for tool_name, result in final_state["mcp_checks"].items():
            print(f"{tool_name}: {result}")

    asyncio.run(main())