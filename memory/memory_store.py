"""
memory/memory_store.py — Long-Term Memory (ChromaDB)
--------------------------------------------------------
Stores every generated report + its findings as embeddings in ChromaDB.
This is what gives the system real long-term memory instead of forgetting
everything after each run.

Three things this enables:
    1. get_most_recent_report() — literal "what did we say last time"
       comparison, used to make the new report say things like
       "compared to last week's report..."
    2. find_similar_reports(query) — semantic search across ALL past
       reports, e.g. "find past reports that also mentioned anomalies
       in the food category"
    3. Metadata filtering (after_date) — constrain semantic search to a
       recent time window, so stale matches from months ago don't get
       pulled in just because they're semantically similar.

Owner: [assign teammate name here]
"""

import chromadb
from datetime import datetime
import json

DB_PATH = "memory/chroma_db"
COLLECTION_NAME = "weekly_reports"


def _to_epoch(date_str: str) -> float:
    """Converts a 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD' string to a Unix
    timestamp. ChromaDB's numeric comparison filters ($gte etc.) only work
    on numbers, not date strings, so we store this alongside the
    human-readable run_date for filtering purposes."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {date_str}")


def _get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def save_report(report_text: str, findings: list[dict], run_date: str = None) -> str:
    """
    Saves a generated report + its findings to memory.
    Returns the run_id used to store it.
    """
    collection = _get_collection()
    run_date = run_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"run_{run_date.replace(' ', '_').replace(':', '-')}"

    collection.add(
        documents=[report_text],
        metadatas=[{
            "run_date": run_date,
            "run_epoch": _to_epoch(run_date),
            "findings_json": json.dumps(findings),
            "num_findings": len(findings),
        }],
        ids=[run_id],
    )
    return run_id


def get_most_recent_report() -> dict | None:
    """
    Returns the most recently saved report (by run_date), or None if
    memory is empty. Used for literal week-over-week comparison.
    """
    collection = _get_collection()
    all_records = collection.get(include=["documents", "metadatas"])

    if not all_records["ids"]:
        return None

    # Sort by run_date metadata, most recent first
    records = list(zip(all_records["ids"], all_records["documents"], all_records["metadatas"]))
    records.sort(key=lambda r: r[2]["run_date"], reverse=True)

    run_id, document, metadata = records[0]
    return {
        "run_id": run_id,
        "report": document,
        "run_date": metadata["run_date"],
        "findings": json.loads(metadata["findings_json"]),
    }


def find_similar_reports(query: str, n_results: int = 3, after_date: str = None) -> list[dict]:
    """
    Semantic search across all past reports. Example query:
    "anomalies in the food category" — returns past reports whose
    content is semantically similar, even if the wording differs.

    Args:
        query: natural language search query
        n_results: max results to return
        after_date: optional metadata filter — only return reports with
            run_date >= this value (e.g. "2026-08-01"). This is the
            metadata-filtering piece of our RAG pipeline: semantic search
            alone can surface stale, irrelevant matches from months ago,
            so callers who only care about recent history can constrain
            the search window explicitly.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    n_results = min(n_results, collection.count())

    where_filter = None
    if after_date:
        where_filter = {"run_epoch": {"$gte": _to_epoch(after_date)}}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    similar = []
    for i in range(len(results["ids"][0])):
        similar.append({
            "run_id": results["ids"][0][i],
            "report": results["documents"][0][i],
            "run_date": results["metadatas"][0][i]["run_date"],
            "distance": results["distances"][0][i],  # lower = more similar
        })
    return similar


def memory_stats() -> dict:
    """Quick summary of what's in memory — useful for demo/debugging."""
    collection = _get_collection()
    return {"total_reports_stored": collection.count()}


if __name__ == "__main__":
    # Standalone test: save a couple of fake reports, then test retrieval
    print("=== Testing Memory Store ===\n")

    print(f"Before: {memory_stats()}")

    save_report(
        report_text="Revenue grew 20% driven by electronics. Food category showed a mild anomaly.",
        findings=[{"metric": "Test", "finding": "test", "magnitude": "+20%", "confidence": "high", "finding_type": "trend"}],
        run_date="2026-07-30 10:00:00",
    )
    save_report(
        report_text="Revenue was flat this week. No major anomalies detected.",
        findings=[],
        run_date="2026-08-06 10:00:00",
    )

    print(f"After saving 2 reports: {memory_stats()}\n")

    recent = get_most_recent_report()
    print("Most recent report:")
    print(f"  Date: {recent['run_date']}")
    print(f"  Report: {recent['report']}\n")

    print("Semantic search for 'food category anomaly':")
    results = find_similar_reports("food category anomaly", n_results=2)
    for r in results:
        print(f"  [{r['run_date']}] (distance={r['distance']:.3f}) {r['report'][:60]}...")

    print("\nSemantic search with metadata filter (after_date='2026-08-01'):")
    results = find_similar_reports("food category anomaly", n_results=2, after_date="2026-08-01")
    for r in results:
        print(f"  [{r['run_date']}] (distance={r['distance']:.3f}) {r['report'][:60]}...")