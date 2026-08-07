"""
tests/eval_suite.py — Eval Suite for The Context Window
-----------------------------------------------------------
This is our "report card" — proof the system actually works, not just
vibes. We build a small SYNTHETIC dataset where we know the exact answer
in advance (we planted it ourselves), then check whether:

    PART A: Insight Analyser correctly detects the planted trends/anomalies
             and does NOT flag random noise as significant
    PART B: Report Writer's output stays grounded in the findings
             (no hallucinated numbers) — reuses the same guardrail check
             from report_writer.py

Run this before demo day and drop the final score directly into your
slides / talking points, same idea as the 41/41 eval you built for the
RAG project.

Owner: [assign teammate name here]
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
from datetime import datetime
from agents.insight_analyser import analyse
from agents.report_writer import write_report, validate_report


# ============================================================
# PART A: Build synthetic data with KNOWN planted patterns
# ============================================================

def build_synthetic_data() -> pd.DataFrame:
    """
    Builds a small fake dataset with deliberately planted patterns so we
    know exactly what the Analyser SHOULD find:

    - 'gadgets' category: steady ~1000/month for 4 months, then spikes to
      5000 in the last month -> should be flagged as BOTH a category trend
      AND a z-score anomaly.
    - 'furniture' category: steady ~2000/month, then drops to 400 in the
      last month -> should be flagged as a category trend (sharp drop).
    - 'stationery' category: tiny, noisy amounts every month (50-150) ->
      should NOT be flagged (too small / within normal noise).
    - 'books' category: perfectly steady 800/month every month -> should
      NOT be flagged (no meaningful change).
    """
    rows = []
    months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]

    # Planted anomaly: gadgets spikes hard in the last month
    gadgets_revenue = [1000, 1050, 980, 1020, 5000]
    # Planted trend: furniture drops hard in the last month
    furniture_revenue = [2000, 1950, 2050, 2000, 400]
    # Noise: should NOT be flagged
    stationery_revenue = [50, 120, 80, 150, 90]
    # Stable: should NOT be flagged
    books_revenue = [800, 800, 800, 800, 800]

    def add_rows(category, monthly_values):
        for i, month in enumerate(months):
            rows.append({
                "order_id": f"{category}_{i}",
                "order_item_id": 1,
                "product_id": f"prod_{category}",
                "seller_id": "seller_1",
                "category": category,
                "price": monthly_values[i],
                "freight_value": 10.0,
                "order_purchase_timestamp": pd.Timestamp(f"{month}-15"),
                "order_month": month,
                "order_status": "delivered",
                "revenue": monthly_values[i],
            })

    add_rows("gadgets", gadgets_revenue)
    add_rows("furniture", furniture_revenue)
    add_rows("stationery", stationery_revenue)
    add_rows("books", books_revenue)

    return pd.DataFrame(rows)


# ============================================================
# PART A tests: does the Analyser find what we planted?
# ============================================================

def run_analyser_evals() -> tuple[int, int, list[str]]:
    """Returns (passed, total, list of failure descriptions)."""
    df = build_synthetic_data()
    findings = analyse(df)

    findings_text = str(findings).lower()
    checks = []

    # Check 1: gadgets spike should be caught as a category trend
    checks.append((
        "Gadgets spike detected as category trend",
        "gadgets" in findings_text and (
            any(f["metric"] == "Category Trend: gadgets" for f in findings)
        ),
    ))

    # Check 2: gadgets spike should ALSO be caught as a statistical anomaly
    checks.append((
        "Gadgets spike detected as anomaly (z-score)",
        any(f["metric"] == "Anomaly: gadgets" for f in findings),
    ))

    # Check 3: furniture drop should be caught as a category trend
    checks.append((
        "Furniture drop detected as category trend",
        any(f["metric"] == "Category Trend: furniture" for f in findings),
    ))

    # Check 4: stationery (tiny, noisy) should NOT be flagged at all
    checks.append((
        "Stationery noise correctly ignored",
        not any("stationery" in f["metric"].lower() for f in findings),
    ))

    # Check 5: books (perfectly stable) should NOT be flagged at all
    checks.append((
        "Books (stable) correctly ignored",
        not any("books" in f["metric"].lower() for f in findings),
    ))

    passed = sum(1 for _, ok in checks if ok)
    failures = [name for name, ok in checks if not ok]
    return passed, len(checks), failures


# ============================================================
# PART B tests: does the guardrail catch a DELIBERATELY BROKEN report?
# ============================================================

def run_guardrail_evals() -> tuple[int, int, list[str]]:
    """
    Tests the hallucination-check logic itself using hand-crafted report
    text (not a live LLM call) — this makes the test fast, free, and
    reproducible, while still proving the guardrail logic works.
    """
    findings = [
        {"metric": "Test Metric", "finding": "Revenue grew", "magnitude": "+25.0%",
         "confidence": "high", "finding_type": "trend"},
    ]

    checks = []

    # Check 1: a report that ONLY uses real numbers should pass
    clean_report = "Revenue grew by 25.0% this period, driven by strong demand."
    result = validate_report(clean_report, findings)
    checks.append(("Clean report passes guardrail", result["passed"]))

    # Check 2: a report with an INVENTED number should be caught
    hallucinated_report = "Revenue grew by 25.0% this period, and next month we expect 47.0% growth."
    result = validate_report(hallucinated_report, findings)
    checks.append(("Hallucinated number is caught", not result["passed"]))

    # Check 3: sign formatting shouldn't cause false positives (regression test
    # for the bug we found during manual testing)
    natural_phrasing = "Revenue dropped by 25.0% this period."  # findings say +25.0%, opposite direction in prose
    result = validate_report(natural_phrasing, findings)
    checks.append(("Sign-normalization doesn't false-positive", result["passed"]))

    # Check 4: dates shouldn't be misread as numbers (regression test for
    # the other bug we found during manual testing)
    date_containing_report = "In 2024-05, revenue grew by 25.0%."
    result = validate_report(date_containing_report, findings)
    checks.append(("Dates aren't misread as numbers", result["passed"]))

    passed = sum(1 for _, ok in checks if ok)
    failures = [name for name, ok in checks if not ok]
    return passed, len(checks), failures


# ============================================================
# Run everything and print a scorecard
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("THE CONTEXT WINDOW — EVAL SUITE")
    print("=" * 55)

    a_passed, a_total, a_failures = run_analyser_evals()
    print(f"\nPART A — Insight Analyser: {a_passed}/{a_total} passed")
    for f in a_failures:
        print(f"  FAILED: {f}")

    b_passed, b_total, b_failures = run_guardrail_evals()
    print(f"\nPART B — Report Guardrail: {b_passed}/{b_total} passed")
    for f in b_failures:
        print(f"  FAILED: {f}")

    total_passed = a_passed + b_passed
    total_checks = a_total + b_total
    print("\n" + "=" * 55)
    print(f"OVERALL SCORE: {total_passed}/{total_checks}")
    print("=" * 55)

    if total_passed == total_checks:
        print("\nAll checks passed. System behaves as expected on known inputs.")
    else:
        print(f"\n{total_checks - total_passed} check(s) failed — review before demo day.")