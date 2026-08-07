"""
Agent 2: Insight Analyser
--------------------------
Job: Take the clean DataFrame from the Fetcher and find what's actually
interesting — revenue trends over time, category performance, and anomalies.

Does NOT write prose. Outputs structured facts only:
    {metric, finding, magnitude, confidence}

This structure is what stops Agent 3 (Report Writer) from hallucinating —
it can only talk about what's in this list.

Owner: [assign teammate name here]
"""

import pandas as pd
from dataclasses import dataclass, asdict


# Guardrail thresholds — an "insight" must clear these bars to be reported.
# Tune these based on your eval suite results.
MIN_PCT_CHANGE_TO_FLAG = 10.0   # ignore trend changes smaller than this (%)
MIN_CATEGORY_REVENUE_SHARE = 1.0  # ignore categories under this % of total revenue
ANOMALY_ZSCORE_THRESHOLD = 2.5  # standard deviations from the mean to count as anomaly
MIN_CATEGORY_REVENUE_FOR_TREND = 500  # ignore categories with less than this in revenue (too noisy)
MIN_CATEGORY_PCT_CHANGE = 60.0  # bigger bar for category-level swings (noisier data)


@dataclass
class Finding:
    metric: str          # e.g. "Total Revenue", "Electronics Category Revenue"
    finding: str          # short factual description, no fluff
    magnitude: str        # e.g. "+18.4%", "R$45,200"
    confidence: str        # "high" / "medium" / "low"
    finding_type: str     # "trend" | "category" | "anomaly"

    def to_dict(self):
        return asdict(self)


def analyse_monthly_trend(df: pd.DataFrame) -> list[Finding]:
    """Find month-over-month revenue trend changes worth reporting."""
    findings = []

    monthly = df.groupby("order_month")["revenue"].sum().sort_index()
    if len(monthly) < 2:
        return findings

    latest_month = monthly.index[-1]
    prev_month = monthly.index[-2]
    latest_val = monthly.iloc[-1]
    prev_val = monthly.iloc[-2]

    if prev_val == 0:
        return findings

    pct_change = ((latest_val - prev_val) / prev_val) * 100

    if abs(pct_change) >= MIN_PCT_CHANGE_TO_FLAG:
        direction = "increased" if pct_change > 0 else "decreased"
        findings.append(Finding(
            metric="Total Revenue (Month-over-Month)",
            finding=f"Revenue {direction} from {prev_month} to {latest_month} "
                     f"(R${prev_val:,.2f} -> R${latest_val:,.2f})",
            magnitude=f"{pct_change:+.1f}%",
            confidence="high" if abs(pct_change) >= 20 else "medium",
            finding_type="trend",
        ))

    return findings


def analyse_category_performance(df: pd.DataFrame) -> list[Finding]:
    """Find which categories drive revenue, and which are underperforming."""
    findings = []

    total_revenue = df["revenue"].sum()
    if total_revenue == 0:
        return findings

    category_revenue = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    category_share = (category_revenue / total_revenue) * 100

    if len(category_revenue) > 0:
        top_cat = category_revenue.index[0]
        top_share = category_share.iloc[0]
        if top_share >= MIN_CATEGORY_REVENUE_SHARE:
            findings.append(Finding(
                metric="Top Category by Revenue",
                finding=f"'{top_cat}' is the highest-revenue category",
                magnitude=f"{top_share:.1f}% of total revenue (R${category_revenue.iloc[0]:,.2f})",
                confidence="high",
                finding_type="category",
            ))

    monthly_cat = df.groupby(["order_month", "category"])["revenue"].sum().reset_index()
    months = sorted(monthly_cat["order_month"].unique())
    category_trend_findings = []
    if len(months) >= 2:
        latest, prev = months[-1], months[-2]
        latest_data = monthly_cat[monthly_cat["order_month"] == latest].set_index("category")["revenue"]
        prev_data = monthly_cat[monthly_cat["order_month"] == prev].set_index("category")["revenue"]

        common_categories = latest_data.index.intersection(prev_data.index)
        for cat in common_categories:
            prev_val = prev_data[cat]
            latest_val = latest_data[cat]
            # Skip only if the category was NEVER significant (prior value
            # too small to matter). Do NOT also require latest_val to be
            # large — a category crashing toward zero is exactly the kind
            # of finding we want to catch, not noise to filter out.
            # (Bug found via eval suite: original check required BOTH
            # values to clear the threshold, which silently hid genuine
            # crashes where the category's value dropped below the bar.)
            if prev_val < MIN_CATEGORY_REVENUE_FOR_TREND:
                continue
            pct_change = ((latest_val - prev_val) / prev_val) * 100
            if abs(pct_change) >= MIN_CATEGORY_PCT_CHANGE:
                direction = "grew" if pct_change > 0 else "dropped"
                category_trend_findings.append((abs(pct_change), Finding(
                    metric=f"Category Trend: {cat}",
                    finding=f"'{cat}' revenue {direction} sharply from {prev} to {latest}",
                    magnitude=f"{pct_change:+.1f}%",
                    confidence="medium",
                    finding_type="category",
                )))

        # Only keep the top 5 most significant category swings — avoids
        # flooding the report with every minor category fluctuation
        category_trend_findings.sort(key=lambda x: x[0], reverse=True)
        findings.extend([f for _, f in category_trend_findings[:5]])

    return findings


def detect_anomalies(df: pd.DataFrame) -> list[Finding]:
    """Flag categories whose latest-month revenue is a statistical outlier
    vs. their own historical monthly average (z-score based)."""
    findings = []
    scored_findings = []

    monthly_cat = df.groupby(["order_month", "category"])["revenue"].sum().reset_index()
    months = sorted(monthly_cat["order_month"].unique())
    if len(months) < 4:
        return findings

    latest_month = months[-1]

    for cat in monthly_cat["category"].unique():
        cat_history = monthly_cat[
            (monthly_cat["category"] == cat) & (monthly_cat["order_month"] != latest_month)
        ]["revenue"]
        latest_val_row = monthly_cat[
            (monthly_cat["category"] == cat) & (monthly_cat["order_month"] == latest_month)
        ]["revenue"]

        if len(cat_history) < 3 or len(latest_val_row) == 0:
            continue

        mean = cat_history.mean()
        std = cat_history.std()
        if std == 0 or pd.isna(std) or mean < MIN_CATEGORY_REVENUE_FOR_TREND:
            continue

        latest_val = latest_val_row.values[0]
        z_score = (latest_val - mean) / std

        if abs(z_score) >= ANOMALY_ZSCORE_THRESHOLD:
            direction = "spiked above" if z_score > 0 else "dropped below"
            scored_findings.append((abs(z_score), Finding(
                metric=f"Anomaly: {cat}",
                finding=f"'{cat}' revenue {direction} its historical average in {latest_month}",
                magnitude=f"z-score {z_score:+.2f} (avg R${mean:,.2f}, latest R${latest_val:,.2f})",
                confidence="high" if abs(z_score) >= 3 else "medium",
                finding_type="anomaly",
            )))

    scored_findings.sort(key=lambda x: x[0], reverse=True)
    findings = [f for _, f in scored_findings[:5]]

    return findings


def analyse(df: pd.DataFrame) -> list[dict]:
    """
    Main entry point. Runs all analysis functions and returns a combined
    list of findings as plain dicts (ready for JSON / LLM prompt use).
    """
    findings: list[Finding] = []
    findings.extend(analyse_monthly_trend(df))
    findings.extend(analyse_category_performance(df))
    findings.extend(detect_anomalies(df))

    return [f.to_dict() for f in findings]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from agents.data_fetcher import fetch_and_clean
    import json

    clean_df, report = fetch_and_clean(
        orders_path="data/raw/olist_orders_dataset.csv",
        order_items_path="data/raw/olist_order_items_dataset.csv",
        products_path="data/raw/olist_products_dataset.csv",
        category_translation_path="data/raw/product_category_name_translation.csv",
    )

    findings = analyse(clean_df)
    print(f"Found {len(findings)} findings:\n")
    print(json.dumps(findings, indent=2))