"""
Agent 3: Report Writer
------------------------
Job: Take the structured findings list from the Insight Analyser and write
a plain-English business report using an LLM (Groq).

GUARDRAIL: This agent is only allowed to talk about numbers that exist in
the findings list. We enforce this two ways:
    1. The prompt explicitly instructs the model not to invent numbers.
    2. After generation, we run a lightweight check comparing numbers in
       the report against numbers in the findings — flagging anything
       that doesn't match (a simple hallucination check).

Owner: [assign teammate name here]
"""

import os
import re
import json
import time
from groq import Groq, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"  # good balance of quality/speed on Groq
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def build_prompt(findings: list[dict], previous_context: str = None) -> str:
    """Turn the structured findings list into a prompt for the LLM.
    If previous_context is given (last week's report text), the model is
    asked to reference it for a "compared to last week" style comparison."""
    findings_text = json.dumps(findings, indent=2)

    comparison_instruction = ""
    if previous_context:
        comparison_instruction = f"""
For context, here is LAST WEEK'S report (for comparison only — do not
re-report its numbers as if they were this week's):
---
{previous_context}
---
If relevant, briefly note how this week compares to last week's report
(e.g. "unlike last week's flat revenue, this week saw..."). Keep this
comparison brief — one sentence at most. If it's not clearly relevant, skip it.
"""

    prompt = f"""You are a business analyst writing a weekly performance report.

Below is a list of findings that were already calculated by a data analysis system.
Your ONLY job is to turn these facts into a clear, readable report.

STRICT RULES:
- Only mention numbers, percentages, and categories that appear in the findings below.
- Do NOT invent, estimate, or round numbers beyond what's given.
- Do NOT add findings that aren't in the list.
- If the findings list is empty, say clearly that no significant changes were found this period.
- Write in plain, professional English. No bullet-point dumps — write it as a short narrative report.
- Structure it with these sections: Summary, Key Trends, Anomalies to Watch, Recommendations.
- Keep it concise — this is a weekly report, not an essay. Aim for 200-350 words.
{comparison_instruction}
FINDINGS:
{findings_text}

Write the report now.
"""
    return prompt


def build_fallback_report(findings: list[dict], error_reason: str) -> str:
    """
    If Groq is unavailable, we still hand back something useful instead of
    crashing — a plain listing of the raw findings. This is the "fail
    loudly, don't guess quietly" principle: we clearly say the AI writer
    is down, rather than silently returning nothing or a fake report.
    """
    lines = [
        "## Weekly Business Report (Fallback Mode)",
        "",
        f"**Note:** The AI report writer is temporarily unavailable ({error_reason}). "
        "Showing raw findings instead so nothing is lost.",
        "",
        "### Raw Findings",
    ]
    if not findings:
        lines.append("No significant findings this period.")
    else:
        for f in findings:
            lines.append(f"- **{f['metric']}**: {f['finding']} ({f['magnitude']}, confidence: {f['confidence']})")
    return "\n".join(lines)


def call_groq_with_retry(client: Groq, prompt: str) -> tuple[str | None, str | None, dict | None]:
    """
    Calls Groq with retry + exponential backoff. Returns
    (response_text, error_message, usage_stats).
    If it succeeds, error_message is None and usage_stats has token/latency
    info. If all retries fail, response_text and usage_stats are None.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            latency = time.time() - start_time
            usage = response.usage
            usage_stats = {
                "latency_seconds": round(latency, 2),
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            }
            return response.choices[0].message.content, None, usage_stats
        except RateLimitError as e:
            last_error = f"rate limited ({e})"
        except APIConnectionError as e:
            last_error = f"connection error ({e})"
        except APIError as e:
            last_error = f"API error ({e})"
        except Exception as e:
            last_error = f"unexpected error ({e})"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)  # simple exponential backoff

    return None, last_error, None


def extract_numbers(text: str, exclude_dates: bool = True) -> set[str]:
    """Extract numeric tokens (percentages, currency figures) from text,
    used for a lightweight hallucination check. Normalizes by stripping
    leading +/- signs so '+144.4%' and '144.4%' are treated as the same
    number (report prose often drops the explicit + sign)."""
    if exclude_dates:
        # Remove date-like patterns (e.g. 2018-08, 2018-07) before number
        # extraction so we don't grab fragments like "-08" out of a date.
        text = re.sub(r"\b\d{4}-\d{2}\b", "", text)

    pattern = r"[+-]?\d[\d,]*\.?\d*%?"
    raw_numbers = re.findall(pattern, text)

    normalized = set()
    for n in raw_numbers:
        cleaned = n.lstrip("+-")  # normalize sign so +144.4%, -144.4%, 144.4% all match
        if cleaned.strip(".,%") == "":  # skip empty/junk matches
            continue
        normalized.add(cleaned)

    return normalized


def validate_report(report_text: str, findings: list[dict]) -> dict:
    """
    Lightweight guardrail check: flags numbers in the report that don't
    appear anywhere in the source findings. Not perfect (numbers can be
    formatted differently), but catches obvious hallucinations.
    """
    findings_text = json.dumps(findings)
    findings_numbers = extract_numbers(findings_text)
    report_numbers = extract_numbers(report_text)

    # Ignore tiny/common numbers likely to be incidental (section numbering etc.)
    suspicious = {
        n for n in report_numbers
        if n not in findings_numbers and len(n.replace(",", "").replace(".", "").replace("-", "")) > 1
    }

    return {
        "passed": len(suspicious) == 0,
        "suspicious_numbers": list(suspicious),
    }


def write_report(findings: list[dict], previous_context: str = None) -> dict:
    """
    Main entry point. Calls Groq to generate the report, then validates it.

    Args:
        findings: structured facts from the Insight Analyser
        previous_context: optional text of last week's report, for
            week-over-week comparison (pulled from ChromaDB memory)

    Returns:
        {
            "report": str,
            "validation": dict,
            "used_fallback": bool,
            "error": str | None,
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

    if not findings:
        # Guardrail: don't even call the LLM if there's nothing to report —
        # avoids the model inventing content to fill space.
        report_text = (
            "## Weekly Business Report\n\n"
            "### Summary\n"
            "No significant changes or anomalies were detected this period. "
            "All metrics remained within normal historical ranges.\n"
        )
        return {"report": report_text, "validation": {"passed": True, "suspicious_numbers": []}, "used_fallback": False, "error": None, "usage": None}

    prompt = build_prompt(findings, previous_context=previous_context)

    report_text, error, usage_stats = call_groq_with_retry(client, prompt)

    if report_text is None:
        # All retries failed — fail loudly with a usable fallback, not a crash
        # and not a silent wrong answer.
        fallback = build_fallback_report(findings, error)
        return {
            "report": fallback,
            "validation": {"passed": True, "suspicious_numbers": []},
            "used_fallback": True,
            "error": error,
            "usage": None,
        }

    validation = validate_report(report_text, findings)

    return {"report": report_text, "validation": validation, "used_fallback": False, "error": None, "usage": usage_stats}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from agents.data_fetcher import fetch_and_clean
    from agents.insight_analyser import analyse

    clean_df, dq_report = fetch_and_clean(
        orders_path="data/raw/olist_orders_dataset.csv",
        order_items_path="data/raw/olist_order_items_dataset.csv",
        products_path="data/raw/olist_products_dataset.csv",
        category_translation_path="data/raw/product_category_name_translation.csv",
    )

    findings = analyse(clean_df)
    result = write_report(findings)

    print("=== GENERATED REPORT ===\n")
    print(result["report"])
    print("\n=== GUARDRAIL CHECK ===")
    print(f"Passed: {result['validation']['passed']}")
    if result["validation"]["suspicious_numbers"]:
        print(f"Suspicious numbers found: {result['validation']['suspicious_numbers']}")