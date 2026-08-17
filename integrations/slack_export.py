"""
integrations/slack_export.py — Send a report to Slack
----------------------------------------------------------
Scoped-down version of "scheduled autorun + Slack/email export": rather
than building a fragile scheduler service days before demo day, we expose
a manual "send to Slack" action. The underlying send logic here is the
same piece that a real scheduler would call automatically — this proves
the export mechanism works without adding scheduling infrastructure risk
this close to the deadline.

Uses a Slack Incoming Webhook (https://api.slack.com/messaging/webhooks) —
no bot token needed, just a webhook URL configured per-channel in Slack.

Owner: [assign teammate name here]
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def send_report_to_slack(report_text: str, webhook_url: str = None) -> dict:
    """
    Sends a report to Slack via an Incoming Webhook.

    Args:
        report_text: the report content to send
        webhook_url: Slack webhook URL. If not provided, reads from the
            SLACK_WEBHOOK_URL environment variable.

    Returns:
        {"success": bool, "error": str | None}
    """
    webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    if not webhook_url:
        return {
            "success": False,
            "error": "No Slack webhook URL configured. Set SLACK_WEBHOOK_URL "
                      "in .env, or pass one directly.",
        }

    # Slack has a practical message size limit — truncate very long reports
    # rather than risk the request failing.
    MAX_SLACK_CHARS = 3000
    display_text = report_text
    if len(display_text) > MAX_SLACK_CHARS:
        display_text = display_text[:MAX_SLACK_CHARS] + "\n\n_...truncated for Slack..._"

    payload = {
        "text": f"📊 *The Context Window — Weekly Report*\n\n{display_text}"
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            return {"success": True, "error": None}
        else:
            return {"success": False, "error": f"Slack returned status {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}


if __name__ == "__main__":
    # Standalone test
    test_report = "## Test Report\n\nThis is a test message from The Context Window."
    result = send_report_to_slack(test_report)
    print(result)