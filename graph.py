"""
graph.py — LangGraph Orchestrator
-----------------------------------
Wires the agents into a single pipeline:

    Data Fetcher -> MCP Verify -> [PARALLEL FAN-OUT]
                                       |-- Trend Analysis -----|
                                       |-- Category Analysis --|--> Combine Findings -> [conditional] -> Report Writer
                                       |-- Anomaly Detection --|                              |
                                                                                    (if no findings) -> No-Findings Report

ORCHESTRATION PATTERN — deliberately mixed:
    The overall pipeline is SEQUENTIAL, because each stage has a genuine
    data dependency on the previous one (Analyser needs Fetcher's clean
    data; Writer needs the Analyser's findings). There's no way around
    that, so it stays sequential.

    But INSIDE the analysis stage, the 3 analysis functions
    (trend, category performance, anomaly detection) have NO dependency
    on each other — they only depend on the same clean_df. So we run
    them as a genuine parallel fan-out/fan-in branch in the graph,
    rather than sequentially, since there's no reason to make one wait
    for another. This is "pattern matches the use case, not just
    convenience" — sequential where there's a real dependency, parallel
    where there isn't.

This is the "how do agents communicate" piece of the demo — state is
passed explicitly between nodes via a shared GraphState object, and you
can print/inspect that state at each step to show the handoff live.

Owner: [assign teammate name here]
"""

import sys
sys.path.insert(0, ".")

import sqlite3
import asyncio
import time
from typing import TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph, END

from agents.data_fetcher import fetch_and_clean, DataQualityReport
from agents.insight_analyser import (
    analyse_monthly_trend,
    analyse_category_performance,
    detect_anomalies,
)
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

    # Populated by the 3 parallel analysis branches
    trend_findings: Optional[list]
    category_findings: Optional[list]
    anomaly_findings: Optional[list]

    # Populated by Combine Findings (merges the 3 branches above)
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


# --- Nodes 2a/2b/2c: PARALLEL analysis branches ---
# Each is async and wraps its (CPU-bound, pandas-based) work in
# asyncio.to_thread so LangGraph can genuinely run all 3 concurrently
# in the same "superstep" instead of one blocking the others.

async def trend_analysis_node(state: GraphState) -> dict:
    start = time.time()
    print("[Trend Analysis] Started (parallel branch 1/3)...")
    findings = await asyncio.to_thread(analyse_monthly_trend, state["clean_df"])
    print(f"[Trend Analysis] Done in {time.time() - start:.2f}s. {len(findings)} findings.")
    # IMPORTANT: return ONLY the key this node updates, not the full state.
    # LangGraph runs all 3 branches concurrently in the same "superstep" —
    # if every branch returned {**state, ...}, all 3 would be writing to
    # every shared key (orders_path, clean_df, etc.) at once, which
    # LangGraph correctly rejects as an unresolvable concurrent write.
    return {"trend_findings": [f.to_dict() for f in findings]}


async def category_analysis_node(state: GraphState) -> dict:
    start = time.time()
    print("[Category Analysis] Started (parallel branch 2/3)...")
    findings = await asyncio.to_thread(analyse_category_performance, state["clean_df"])
    print(f"[Category Analysis] Done in {time.time() - start:.2f}s. {len(findings)} findings.")
    return {"category_findings": [f.to_dict() for f in findings]}


async def anomaly_analysis_node(state: GraphState) -> dict:
    start = time.time()
    print("[Anomaly Detection] Started (parallel branch 3/3)...")
    findings = await asyncio.to_thread(detect_anomalies, state["clean_df"])
    print(f"[Anomaly Detection] Done in {time.time() - start:.2f}s. {len(findings)} findings.")
    return {"anomaly_findings": [f.to_dict() for f in findings]}


# --- Node 2d: Combine Findings — fan-in point, waits for all 3 branches ---
def combine_findings_node(state: GraphState) -> dict:
    combined = []
    combined.extend(state.get("trend_findings") or [])
    combined.extend(state.get("category_findings") or [])
    combined.extend(state.get("anomaly_findings") or [])
    print(f"[Combine Findings] Merged {len(combined)} findings from 3 parallel branches.")
    return {"findings": combined}


# --- Conditional routing: skip straight to a short report if nothing found ---
def route_after_analysis(state: GraphState) -> str:
    if not state["findings"]:
        return "no_findings"
    return "has_findings"


# --- Node 3a: Report Writer (normal path, findings exist) ---
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
    total_revenue = float(state["clean_df"]["revenue"].sum())
    save_report(
        result["report"],
        state["findings"],
        usage=result.get("usage"),
        guardrail_passed=result["validation"]["passed"],
        used_fallback=result.get("used_fallback", False),
        total_revenue=total_revenue,
    )

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
    graph.add_node("trend_analysis", trend_analysis_node)
    graph.add_node("category_analysis", category_analysis_node)
    graph.add_node("anomaly_analysis", anomaly_analysis_node)
    graph.add_node("combine_findings", combine_findings_node)
    graph.add_node("writer", writer_node)
    graph.add_node("no_findings", no_findings_node)

    graph.set_entry_point("fetcher")
    graph.add_edge("fetcher", "mcp_verify")

    # --- Fan-out: mcp_verify feeds all 3 analysis branches simultaneously ---
    graph.add_edge("mcp_verify", "trend_analysis")
    graph.add_edge("mcp_verify", "category_analysis")
    graph.add_edge("mcp_verify", "anomaly_analysis")

    # --- Fan-in: combine_findings waits for all 3 branches to complete ---
    graph.add_edge("trend_analysis", "combine_findings")
    graph.add_edge("category_analysis", "combine_findings")
    graph.add_edge("anomaly_analysis", "combine_findings")

    # Conditional edge — the actual guardrail logic
    graph.add_conditional_edges(
        "combine_findings",
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
            "trend_findings": None,
            "category_findings": None,
            "anomaly_findings": None,
            "findings": None,
            "report": None,
            "validation": None,
        }

        print("=" * 50)
        print("RUNNING THE CONTEXT WINDOW PIPELINE")
        print("=" * 50)

        pipeline_start = time.time()
        final_state = await app.ainvoke(initial_state)
        total_time = time.time() - pipeline_start

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

        print(f"\nTotal pipeline time: {total_time:.2f}s")

    asyncio.run(main())