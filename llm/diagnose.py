"""
diagnose.py
-----------
For each cluster found by cluster_engine.py, sends the cluster's log
lines to Groq's free LLM API and asks for a short, plain-English
root-cause explanation + recommended first action.
"""

import os
import sqlite3
from collections import defaultdict
from groq import Groq

DB_PATH = "logs.db"
MODEL = "openai/gpt-oss-20b"  # current free-tier model as of late 2026, replaces decommissioned llama-3.3-70b-versatile

PROMPT_TEMPLATE = """You are an SRE/DevOps assistant. Below are {n} related log lines \
from the same incident, grouped automatically by a clustering algorithm.

Log lines:
{log_lines}

In 2-3 sentences, explain the most likely root cause of this incident, \
then recommend ONE specific first action an on-call engineer should take. \
Be specific and concise. Do not repeat the raw log lines back."""


def get_client():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment.")
    return Groq(api_key=api_key)


def load_clusters(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT cluster_id, message, service_name, log_level FROM logs "
                "WHERE cluster_id IS NOT NULL ORDER BY cluster_id")
    rows = cur.fetchall()
    conn.close()

    clusters = defaultdict(list)
    for cluster_id, message, service_name, log_level in rows:
        clusters[cluster_id].append(f"[{log_level}] {service_name}: {message}")
    return clusters


def diagnose_cluster(client, cluster_id, log_lines, max_lines=10):
    if cluster_id == -1:
        return "Unclustered/isolated alert -- likely unrelated noise, not a single incident."

    sample_lines = log_lines[:max_lines]
    prompt = PROMPT_TEMPLATE.format(
        n=len(log_lines),
        log_lines="\n".join(f"- {line}" for line in sample_lines),
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        reasoning_effort="low",  # gpt-oss burns tokens on hidden reasoning by default; "low" leaves room for the actual answer
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def save_all_diagnoses(results, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            cluster_id INTEGER PRIMARY KEY,
            diagnosis TEXT,
            log_count INTEGER
        )
    """)
    for cluster_id, (diagnosis, count) in results.items():
        cur.execute(
            "INSERT OR REPLACE INTO diagnoses (cluster_id, diagnosis, log_count) VALUES (?, ?, ?)",
            (cluster_id, diagnosis, count),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    client = get_client()
    clusters = load_clusters()

    if not clusters:
        print("No clusters found. Run cluster_engine.py first.")
        exit(1)

    print(f"Found {len(clusters)} clusters (including noise/-1 if present). Diagnosing each...\n")

    results = {}
    for cluster_id, log_lines in sorted(clusters.items()):
        label = "NOISE/UNCLUSTERED" if cluster_id == -1 else f"INCIDENT #{cluster_id}"
        print(f"--- {label} ({len(log_lines)} log lines) ---")

        diagnosis = diagnose_cluster(client, cluster_id, log_lines)
        print(diagnosis)
        print()

        results[cluster_id] = (diagnosis, len(log_lines))

    save_all_diagnoses(results)
    print(f"Saved {len(results)} diagnoses to {DB_PATH} (table: diagnoses)")
