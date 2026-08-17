"""
app/streamlit_app.py — The Context Window UI
------------------------------------------------
Run with: streamlit run app/streamlit_app.py

Shows the full pipeline running live: Data Fetcher -> MCP Verify ->
Insight Analyser -> Report Writer, with each step visible as it happens.
This is what you'll actually click through on Demo Day.

Owner: [assign teammate name here]
"""

import sys
sys.path.insert(0, ".")

import asyncio
import time
import streamlit as st
import pandas as pd

from agents.data_fetcher import fetch_and_clean
from agents.insight_analyser import analyse
from agents.report_writer import write_report
from agents.qa_agent import answer_question
from mcp_servers.dq_client import run_all_checks
from memory.memory_store import save_report, get_most_recent_report, memory_stats, get_all_reports, find_similar_reports
from integrations.slack_export import send_report_to_slack

st.set_page_config(page_title="The Context Window", page_icon="📊", layout="wide")

st.title("📊 The Context Window")
st.caption("A multi-agent system that turns raw sales data into a weekly business report — automatically.")

with st.sidebar:
    st.header("About")
    st.write(
        "**3 agents, one pipeline:**\n\n"
        "1. **Data Fetcher** — loads & cleans data\n"
        "2. **Insight Analyser** — finds trends & anomalies\n"
        "3. **Report Writer** — writes the report (Groq)\n\n"
        "Plus: MCP data-quality checks and ChromaDB memory for "
        "week-over-week comparison."
    )
    stats = memory_stats()
    st.metric("Reports in memory", stats["total_reports_stored"])

DATA_PATHS = {
    "orders_path": "data/raw/olist_orders_dataset.csv",
    "order_items_path": "data/raw/olist_order_items_dataset.csv",
    "products_path": "data/raw/olist_products_dataset.csv",
    "category_translation_path": "data/raw/product_category_name_translation.csv",
}


def run_pipeline():
    """Runs all 3 agents + MCP verify + memory, updating the UI live at each step."""
    pipeline_start = time.time()

    # --- Agent 1: Data Fetcher ---
    with st.status("🔍 Data Fetcher — loading and cleaning data...", expanded=True) as status:
        clean_df, quality_report = fetch_and_clean(**DATA_PATHS)
        st.write(f"Loaded and cleaned **{len(clean_df):,}** rows.")
        st.code(quality_report.summary(), language=None)
        status.update(label="✅ Data Fetcher — done", state="complete")

    # --- MCP Verify ---
    with st.status("🔧 MCP Verify — running data-quality checks via MCP server...", expanded=True) as status:
        import sqlite3
        conn = sqlite3.connect("data/context_window.db")
        clean_df.to_sql("sales", conn, if_exists="replace", index=False)
        conn.close()

        mcp_checks = asyncio.run(run_all_checks(table="sales"))
        for tool_name, result in mcp_checks.items():
            st.write(f"**{tool_name}**: {result}")
        status.update(label="✅ MCP Verify — done", state="complete")

    # --- Agent 2: Insight Analyser ---
    with st.status("📈 Insight Analyser — finding trends and anomalies...", expanded=True) as status:
        findings = analyse(clean_df)
        st.write(f"Found **{len(findings)}** significant findings.")
        if findings:
            st.dataframe(findings, use_container_width=True)
        status.update(label="✅ Insight Analyser — done", state="complete")

    # --- Agent 3: Report Writer ---
    with st.status("✍️ Report Writer — checking memory and generating report...", expanded=True) as status:
        previous = get_most_recent_report()
        previous_context = previous["report"] if previous else None
        if previous:
            st.write(f"Found previous report from **{previous['run_date']}** — will reference it for comparison.")
        else:
            st.write("No previous report in memory (first run).")

        result = write_report(findings, previous_context=previous_context)
        st.write(f"Guardrail check: {'✅ Passed' if result['validation']['passed'] else '⚠️ Flagged'}")
        if result.get("used_fallback"):
            st.warning(f"⚠️ Groq API unavailable after retries ({result.get('error')}) — showing raw findings as fallback.")
        if not result["validation"]["passed"]:
            st.warning(f"Suspicious numbers: {result['validation']['suspicious_numbers']}")
        if result.get("usage"):
            u = result["usage"]
            st.caption(f"⏱️ {u['latency_seconds']}s · {u['total_tokens']} tokens ({u['prompt_tokens']} in / {u['completion_tokens']} out)")

        save_report(
            result["report"],
            findings,
            usage=result.get("usage"),
            guardrail_passed=result["validation"]["passed"],
            used_fallback=result.get("used_fallback", False),
            total_revenue=float(clean_df["revenue"].sum()),
        )
        st.write("Report saved to memory for next run's comparison.")
        status.update(label="✅ Report Writer — done", state="complete")

    total_time = time.time() - pipeline_start
    st.caption(f"Total pipeline time: {total_time:.2f}s")

    return result["report"], quality_report, mcp_checks, findings, result["validation"]


tab_pipeline, tab_monitoring, tab_history = st.tabs(["🚀 Pipeline", "📡 Monitoring", "📅 Report History"])

with tab_pipeline:
    if st.button("▶️ Run Pipeline", type="primary", use_container_width=True):
        report, quality_report, mcp_checks, findings, validation = run_pipeline()
        # Store in session_state so the Q&A section below survives reruns
        st.session_state["report"] = report
        st.session_state["quality_report"] = quality_report
        st.session_state["mcp_checks"] = mcp_checks
        st.session_state["findings"] = findings
        st.session_state["validation"] = validation

    if "report" in st.session_state:
        st.divider()
        st.subheader("📄 Final Report")
        st.markdown(st.session_state["report"].replace("$", "\\$"))

        st.divider()
        st.subheader("📋 Findings at a Glance")
        findings_df = pd.DataFrame(st.session_state["findings"])
        findings_df = findings_df.rename(columns={
            "finding_type": "Type", "metric": "Metric", "finding": "Finding",
            "magnitude": "Magnitude", "confidence": "Confidence",
        })
        findings_df["Type"] = findings_df["Type"].str.capitalize()
        findings_df = findings_df[["Type", "Metric", "Finding", "Magnitude", "Confidence"]]
        st.dataframe(findings_df, use_container_width=True, hide_index=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Quality")
            st.code(st.session_state["quality_report"].summary(), language=None)
        with col2:
            st.subheader("Guardrail Check")
            st.json(st.session_state["validation"])
            st.divider()
        st.subheader("📤 Export & Share")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                label="⬇️ Download Report (.md)",
                data=st.session_state["report"],
                file_name=f"context_window_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with export_col2:
            if st.button("📨 Send to Slack", use_container_width=True):
                with st.spinner("Sending to Slack..."):
                    slack_result = send_report_to_slack(st.session_state["report"])
                if slack_result["success"]:
                    st.success("Sent to Slack ✅")
                else:
                    st.warning(f"Couldn't send to Slack: {slack_result['error']}")



        st.divider()
        st.subheader("💬 Ask a Question")
        st.caption("Ask anything about this report — answers are grounded strictly in the findings above.")

        question = st.text_input("Your question", placeholder="e.g. Which category should I be most worried about?")
        if st.button("Ask"):
            if question.strip():
                with st.spinner("Thinking..."):
                    qa_result = answer_question(
                        question,
                        st.session_state["findings"],
                        st.session_state["report"],
                    )
                st.markdown(f"**Answer:** {qa_result['answer'].replace('$', chr(92)+'$')}")
                if qa_result.get("used_fallback"):
                    st.warning("⚠️ AI service was unavailable for this question.")
                if qa_result.get("usage"):
                    u = qa_result["usage"]
                    st.caption(f"⏱️ {u['latency_seconds']}s · {u['total_tokens']} tokens ({u['prompt_tokens']} in / {u['completion_tokens']} out)")
                if qa_result["used_history"] and qa_result.get("history_sources"):
                    sources_str = ", ".join(s["run_id"] for s in qa_result["history_sources"])
                    st.caption(f"🧠 Sourced from memory: {sources_str}")
                if not qa_result["validation"]["passed"]:
                    st.warning(f"Guardrail flagged possible unsupported numbers: {qa_result['validation']['suspicious_numbers']}")
            else:
                st.warning("Type a question first.")
    else:
        st.info("Click **Run Pipeline** to run all 3 agents end-to-end on the Olist dataset.")


with tab_monitoring:
    st.subheader("📡 Live Run Monitoring")
    st.caption("Cost and latency tracking across ALL runs ever (persisted in ChromaDB) — not just this browser session.")

    all_reports = get_all_reports()
    # Only reports that have performance data attached (older reports saved
    # before this feature existed won't have it — filter those out cleanly
    # rather than showing broken/missing metrics).
    perf_reports = [r for r in all_reports if r.get("latency_seconds") is not None]

    if not perf_reports:
        st.info("No performance data yet. Run the pipeline at least once (in the Pipeline tab) to see monitoring data here.")
    else:
        perf_df = pd.DataFrame(perf_reports)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total runs tracked", len(perf_df))
        col1.metric("Avg. latency", f"{perf_df['latency_seconds'].mean():.2f}s")
        col2.metric("Guardrail pass rate", f"{(perf_df['guardrail_passed'].fillna(True).sum() / len(perf_df) * 100):.0f}%")
        col2.metric("Fallback triggered", int(perf_df["used_fallback"].fillna(False).sum()))
        col3.metric("Avg. tokens/run", f"{perf_df['total_tokens'].mean():.0f}")
        col3.metric("Total findings (all runs)", int(perf_df["num_findings"].sum()))

        st.divider()
        st.write("**Latency per run, over time**")
        st.line_chart(perf_df.set_index("run_date")["latency_seconds"])

        st.write("**Token usage per run, over time**")
        st.line_chart(perf_df.set_index("run_date")["total_tokens"])

        st.write("**Run log**")
        st.dataframe(
            perf_df[["run_date", "latency_seconds", "total_tokens", "num_findings", "guardrail_passed", "used_fallback"]],
            use_container_width=True,
        )


with tab_history:
    st.subheader("📅 Reports Over Time")
    st.caption("Every report ever generated (from ChromaDB memory) — proof that memory persists across runs, not just within one session.")

    all_reports = get_all_reports()

    if not all_reports:
        st.info("No reports in memory yet. Run the pipeline at least once to populate this view.")
    else:
        history_df = pd.DataFrame([
            {
                "run_date": r["run_date"],
                "num_findings": r["num_findings"],
                "run_id": r["run_id"],
                "total_revenue": r.get("total_revenue"),
            }
            for r in all_reports
        ])

        # Revenue trend — only shown if at least one report has this data
        # (older reports saved before this feature existed won't have it)
        if history_df["total_revenue"].notna().any():
            st.write("**Total revenue per report, over time**")
            st.line_chart(history_df.dropna(subset=["total_revenue"]).set_index("run_date")["total_revenue"])
            st.divider()

        st.write("**Findings per report over time**")
        st.bar_chart(history_df.set_index("run_date")["num_findings"])

        st.divider()
        st.subheader("🔎 Search Past Reports")
        st.caption(
            "Semantic search across all stored reports — finds reports by MEANING, "
            "not just exact keyword matches. This is the same RAG retrieval used "
            "automatically by the Q&A tab when you ask about \"last week\" — exposed "
            "here directly so you can search any past report."
        )
        search_query = st.text_input("Search query", placeholder="e.g. food category anomaly")
        if st.button("🔎 Search"):
            if search_query.strip():
                search_results = find_similar_reports(search_query, n_results=5)
                if not search_results:
                    st.info("No stored reports to search yet.")
                else:
                    for r in search_results:
                        st.markdown(f"**{r['run_date']}** — similarity distance: `{r['distance']:.3f}` (lower = more similar)")
                        st.markdown(r["report"][:300].replace("$", "\\$") + "...")
                        st.divider()
            else:
                st.warning("Type a search query first.")

        st.divider()
        st.write(f"**All {len(all_reports)} stored reports** (most recent first)")
        for r in reversed(all_reports):
            with st.expander(f"{r['run_date']} — {r['num_findings']} findings ({r['run_id']})"):
                st.markdown(r["report"].replace("$", "\\$"))