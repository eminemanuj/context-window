"""
agents/qa_agent.py — Q&A Layer
----------------------------------
Job: Let the user ask follow-up questions about the generated report and
findings (e.g. "why did food revenue spike?", "which category is riskiest?").

Uses the SAME guardrail approach as the Report Writer: the model is only
allowed to answer using the findings it's given, and we run the same
hallucination check on its answer afterward.

Also memory-aware: if the question seems to reference history ("compared
to last week", "has this happened before"), it pulls relevant past
reports from ChromaDB to ground the answer — and cites the specific
source (run_id + date) it used, not just "memory" vaguely.

RELIABILITY: Groq calls use retry with backoff, and fail loudly with a
clear message rather than crashing if the service is unavailable.

Owner: [assign teammate name here]
"""

import sys
sys.path.insert(0, ".")

import os
import json
import time
from groq import Groq, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

from agents.report_writer import validate_report
from memory.memory_store import find_similar_reports, get_most_recent_report

load_dotenv()

MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

HISTORY_KEYWORDS = ["last week", "before", "previous", "compared", "history", "past", "again", "trend over time"]


def _needs_history(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in HISTORY_KEYWORDS)


def build_qa_prompt(question: str, findings: list[dict], report_text: str, history_context: str = None) -> str:
    findings_text = json.dumps(findings, indent=2)

    history_block = ""
    if history_context:
        history_block = f"""
RELEVANT PAST REPORTS (for historical context only — each is tagged with a
Source ID and date; cite the Source ID if you reference it):
---
{history_context}
---
"""

    prompt = f"""You are a business analyst answering a follow-up question about a report
you already wrote. You must stay strictly grounded in the data provided below.

STRICT RULES:
- Only use numbers, percentages, and categories that appear in the findings or report below.
- Do NOT invent, estimate, or guess numbers that aren't given.
- If the question asks something the data genuinely can't answer, say so clearly
  instead of guessing (e.g. "The data doesn't show the cause, only that revenue
  changed by X%").
- Keep the answer concise — 2-4 sentences, conversational, not a full report.

THE REPORT YOU WROTE:
{report_text}

THE UNDERLYING FINDINGS (structured facts the report was based on):
{findings_text}
{history_block}
USER'S QUESTION:
{question}

Answer the question now.
"""
    return prompt


def answer_question(question: str, findings: list[dict], report_text: str) -> dict:
    """
    Main entry point. Answers a user question grounded in the findings,
    optionally pulling historical context from memory if relevant.

    Returns:
        {
            "answer": str,
            "validation": dict,
            "used_history": bool,
            "history_sources": list,
            "used_fallback": bool,
            "usage": dict | None,
        }
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure you have a .env file with "
            "GROQ_API_KEY=your_key_here in the project root."
        )

    client = Groq(api_key=api_key)

    history_context = None
    history_sources = []
    used_history = False
    if _needs_history(question):
        similar = find_similar_reports(question, n_results=2)
        if similar:
            history_context = "\n\n".join(
                f"[Source: {r['run_id']}, dated {r['run_date']}]: {r['report']}" for r in similar
            )
            history_sources = [{"run_id": r["run_id"], "run_date": r["run_date"]} for r in similar]
            used_history = True

    prompt = build_qa_prompt(question, findings, report_text, history_context)

    answer_text = None
    error = None
    usage_stats = None
    for attempt in range(1, MAX_RETRIES + 1):
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            latency = time.time() - start_time
            usage = response.usage
            usage_stats = {
                "latency_seconds": round(latency, 2),
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            }
            answer_text = response.choices[0].message.content
            error = None
            break
        except RateLimitError as e:
            error = f"rate limited ({e})"
        except APIConnectionError as e:
            error = f"connection error ({e})"
        except APIError as e:
            error = f"API error ({e})"
        except Exception as e:
            error = f"unexpected error ({e})"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    if answer_text is None:
        return {
            "answer": f"⚠️ Sorry, I can't answer right now — the AI service is unavailable ({error}). "
                      f"Please try again in a moment.",
            "validation": {"passed": True, "suspicious_numbers": []},
            "used_history": used_history,
            "history_sources": history_sources,
            "used_fallback": True,
            "usage": None,
        }

    validation = validate_report(answer_text, findings)

    return {
        "answer": answer_text,
        "validation": validation,
        "used_history": used_history,
        "history_sources": history_sources,
        "used_fallback": False,
        "usage": usage_stats,
    }


if __name__ == "__main__":
    from agents.data_fetcher import fetch_and_clean
    from agents.insight_analyser import analyse
    from agents.report_writer import write_report

    clean_df, dq_report = fetch_and_clean(
        orders_path="data/raw/olist_orders_dataset.csv",
        order_items_path="data/raw/olist_order_items_dataset.csv",
        products_path="data/raw/olist_products_dataset.csv",
        category_translation_path="data/raw/product_category_name_translation.csv",
    )
    findings = analyse(clean_df)
    result = write_report(findings)

    test_question = "Which category should I be most worried about?"
    qa_result = answer_question(test_question, findings, result["report"])

    print(f"Q: {test_question}")
    print(f"A: {qa_result['answer']}")
    print(f"\nGuardrail passed: {qa_result['validation']['passed']}")
    print(f"Used history: {qa_result['used_history']}")
    if qa_result.get("usage"):
        print(f"Usage: {qa_result['usage']}")