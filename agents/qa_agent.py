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
reports from ChromaDB to ground the answer.

Owner: [assign teammate name here]
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

from agents.report_writer import validate_report
from memory.memory_store import find_similar_reports, get_most_recent_report

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

HISTORY_KEYWORDS = ["last week", "before", "previous", "compared", "history", "past", "again", "trend over time"]


def _needs_history(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in HISTORY_KEYWORDS)


def build_qa_prompt(question: str, findings: list[dict], report_text: str, history_context: str = None) -> str:
    findings_text = json.dumps(findings, indent=2)

    history_block = ""
    if history_context:
        history_block = f"""
RELEVANT PAST REPORTS (for historical context only):
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
            "validation": dict,       # same guardrail check as Report Writer
            "used_history": bool,
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
    used_history = False
    if _needs_history(question):
        similar = find_similar_reports(question, n_results=2)
        if similar:
            history_context = "\n\n".join(
                f"[{r['run_date']}]: {r['report']}" for r in similar
            )
            used_history = True

    prompt = build_qa_prompt(question, findings, report_text, history_context)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    answer_text = response.choices[0].message.content

    # Reuse the same hallucination guardrail from the Report Writer
    validation = validate_report(answer_text, findings)

    return {
        "answer": answer_text,
        "validation": validation,
        "used_history": used_history,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
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