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
import streamlit as st

from agents.data_fetcher import fetch_and_clean
from agents.insight_analyser import analyse
from agents.report_writer import write_report
from agents.qa_agent import answer_question
from mcp_servers.dq_client import run_all_checks
from memory.memory_store import save_report, get_most_recent_report, memory_stats

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

        save_report(result["report"], findings)
        st.write("Report saved to memory for next run's comparison.")
        status.update(label="✅ Report Writer — done", state="complete")

    return result["report"], quality_report, mcp_checks, findings, result["validation"]


st.divider()

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
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Quality")
        st.code(st.session_state["quality_report"].summary(), language=None)
    with col2:
        st.subheader("Guardrail Check")
        st.json(st.session_state["validation"])

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