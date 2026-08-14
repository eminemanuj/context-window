# 📊 The Context Window

A multi-agent Business Intelligence system that turns raw e-commerce sales data into an automated weekly report — no manual analysis required.

Built as a Cohort 2 Capstone Project.

---

## What It Does

Every week, someone normally has to manually dig through sales data to answer: *"How is the business actually doing?"* That takes about an hour of spreadsheet work.

**The Context Window automates this end-to-end.** Feed it raw sales data, and three AI agents work together to load it, analyze it, and write a plain-English report — complete with trend detection, anomaly flagging, and week-over-week memory. You can then ask it follow-up questions, all grounded strictly in real data with zero hallucination.

Data Fetcher → MCP Verify → Insight Analyser → [conditional] → Report Writer → Q&A Layer

---

## Architecture

| Component | Role |
|---|---|
| **Data Fetcher** (`agents/data_fetcher.py`) | Loads and joins raw CSVs, cleans bad data, validates quality |
| **MCP Verify** (`mcp_servers/`) | A FastMCP data-quality server — real MCP client/server calls checking nulls, duplicates, freshness. Every tool call is error-wrapped, and the client has a circuit breaker that stops calling further tools after repeated failures |
| **Insight Analyser** (`agents/insight_analyser.py`) | Finds statistically significant trends, category performance shifts, and anomalies (z-score based) |
| **Report Writer** (`agents/report_writer.py`) | Generates a narrative report via Groq (Llama 3.3), with retry logic, a graceful fallback (raw findings) if the API is unavailable, and a hallucination guardrail that verifies every number against source data |
| **Q&A Layer** (`agents/qa_agent.py`) | Answers follow-up questions, grounded in the same findings, memory-aware for historical questions with explicit source citations, same retry/fallback reliability as the Report Writer |
| **Memory** (`memory/memory_store.py`) | ChromaDB — stores every report as an embedding for week-over-week comparison and semantic search, with metadata filtering (by date) to avoid stale matches |
| **Orchestration** (`graph.py`) | LangGraph — wires all agents into one pipeline with a conditional edge (skips the LLM call if nothing significant is found) |
| **UI** (`app/streamlit_app.py`) | Streamlit app — click through the full pipeline live, then ask questions |

---

## Tech Stack

`LangGraph` · `FastMCP` · `ChromaDB` · `Groq (Llama 3.3 70B)` · `SQLite` · `Streamlit` · `pandas`

---

## Guardrails

- **Significance thresholds** — tiny, noisy category swings are filtered out before they ever reach the report
- **Skip-if-nothing-found** — if the Analyser finds nothing significant, the LLM is never called; avoids the model inventing content
- **Hallucination check** — every number in the generated report is verified against the source findings after generation
- **Q&A grounding** — follow-up answers reuse the same number-verification guardrail as the main report

---

## Reliability & Error Handling

Built against the "fail loudly, don't guess quietly" principle:

- **Retry with backoff** — Groq API calls (Report Writer, Q&A Agent) retry up to 3 times with exponential backoff before giving up
- **Graceful fallback, not a crash** — if Groq is genuinely unavailable after retries, the Report Writer falls back to a plain listing of the raw findings (clearly labeled "Fallback Mode") instead of crashing the pipeline; the Q&A Agent returns a clear "service unavailable" message instead of an unhandled exception
- **MCP tool error handling** — every tool in the data-quality MCP server (`count_rows`, `check_nulls`, `check_duplicates`, `check_freshness`) is wrapped in try/except and returns a clean `ERROR: ...` string rather than crashing the server on a bad table/column name
- **Circuit breaker** — the MCP client stops calling further tools after 2 consecutive failures, marking the rest as `SKIPPED` rather than continuing to hammer a server that's clearly down
- **RAG source citations** — when the Q&A Agent uses memory to answer a historical question, it cites the exact `run_id` and date of the report(s) it used, not just "memory" vaguely
- **Metadata filtering** — `find_similar_reports()` supports an `after_date` filter, so semantic search can be constrained to a recent time window instead of surfacing stale matches purely by similarity

---

## What Broke (and How We Fixed It)

Seven real bugs were found and fixed during development:

1. **Phantom -100% crash** — an incomplete trailing month in the raw data made revenue look like it had crashed. Fixed by detecting and dropping partial reporting periods.
2. **Noisy findings** — the first working version produced 39 findings per run, mostly meaningless swings in tiny categories. Tuned thresholds down to a focused, meaningful 11.
3. **Hidden crashes** — the eval suite caught a filter that was accidentally hiding genuine crash-to-near-zero findings because it required *both* the before and after values to clear a minimum threshold. Fixed the logic to only check the prior value.
4. **False-positive hallucination flags** — the number-checker was flagging real numbers as "hallucinated" because `+18.4%` in the data and `18.4%` in the prose weren't recognized as the same number. Fixed by normalizing signs before comparison.
5. **Dates misread as numbers** — the same checker was grabbing fragments like `-08` out of dates like `2018-08` and treating them as invented numbers. Fixed by excluding date patterns before number extraction.
6. **Orphaned function after a refactor** — while adding retry logic to the Report Writer, a mid-file edit accidentally deleted the `def extract_numbers(...)` signature, leaving its body attached to nothing. The eval suite caught this immediately on the next run (`NameError: extract_numbers is not defined`) — direct proof the eval suite earns its keep beyond the initial build.
7. **No error boundaries on external calls** — an early version had no retry/fallback logic around the Groq API or MCP tool calls; a single failed request would crash the entire pipeline. Added retry-with-backoff, a fallback report path, MCP-side error wrapping, and a circuit breaker on repeated tool failures.

---

---

## Screenshots

The full pipeline running end-to-end in the Streamlit UI:

| | |
|---|---|
| ![Data Fetcher](Screenshots/1.png) | ![Data Fetcher output](Screenshots/2.png) |
| ![MCP Verify](Screenshots/3.png) | ![MCP Verify output](Screenshots/4.png) |
| ![Insight Analyser](Screenshots/5.png) | ![Findings table](Screenshots/6.png) |
| ![Report Writer](Screenshots/7.png) | ![Generated report](Screenshots/8.png) |
| ![Data quality + guardrail check](Screenshots/9.png) | ![Q&A section](Screenshots/10.png) |
| ![Q&A answer with source citation](Screenshots/11.png) | ![Full report view](Screenshots/12.png) |

---

## Eval Suite — 9/9

A synthetic dataset with deliberately planted, known outcomes proves the system behaves correctly:

- Planted revenue spike → correctly detected as both a trend and a statistical anomaly
- Planted revenue crash → correctly detected
- Noisy, tiny category → correctly ignored
- Perfectly stable category → correctly ignored
- Hand-crafted hallucinated report → correctly flagged by the guardrail
- Clean, accurate report → correctly passes the guardrail

Run it yourself:
```bash
python tests/eval_suite.py
```

---

## Setup

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/eminemanuj/context-window.git
cd context-window
uv venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
```

### 2. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a `.env` file in the project root:

GROQ_API_KEY=your_key_here

### 4. Add the dataset
Place the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) CSVs in `data/raw/`:
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `product_category_name_translation.csv`

### 5. Load data into SQLite (for the MCP data-quality server)
```bash
python data/load_to_sqlite.py
```

---

## Running It

**Full pipeline, terminal:**
```bash
python graph.py
```

**Full pipeline, UI (recommended):**
```bash
streamlit run app/streamlit_app.py
```

**Just the eval suite:**
```bash
python tests/eval_suite.py
```

**Test the MCP server standalone:**
```bash
python mcp_servers/test_dq_client.py
```

---

## Project Structure

context-window/
├── agents/
│ ├── data_fetcher.py # Agent 1: load, join, clean, validate
│ ├── insight_analyser.py # Agent 2: trends, category performance, anomalies
│ ├── report_writer.py # Agent 3: Groq-powered report + guardrail + retry/fallback
│ └── qa_agent.py # Q&A layer, memory-aware, cites sources, retry/fallback
├── mcp_servers/
│ ├── dq_server.py # FastMCP data-quality server, error-wrapped tools
│ ├── dq_client.py # Client used inside the pipeline, circuit breaker
│ └── test_dq_client.py # Standalone test client
├── memory/
│ └── memory_store.py # ChromaDB long-term memory, metadata filtering
├── data/
│ ├── raw/ # Source CSVs (not committed)
│ └── load_to_sqlite.py # Loads clean data into SQLite
├── tests/
│ └── eval_suite.py # Synthetic-data eval suite (9/9)
├── app/
│ └── streamlit_app.py # Demo UI
├── graph.py # LangGraph orchestrator
├── requirements.txt
└── README.md

---

## Team

| Member | Agent Owned |
|---|---|
| [Name] | Data Fetcher |
| [Name] | Insight Analyser |
| [Name] | Report Writer / Q&A |
