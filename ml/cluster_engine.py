"""
cluster_engine.py
------------------
Reads logs from logs.db, converts log messages into vector embeddings,
and groups related logs into incidents using DBSCAN (unsupervised
clustering). Evaluates clustering quality against the hidden
scenario_label ground truth (which is NEVER fed into the clustering
algorithm itself -- it's only used afterwards to check our work).

Embedding strategy:
  - Preferred: sentence-transformers (all-MiniLM-L6-v2) -- captures real
    semantic meaning. Use this when running in an environment with
    internet access (e.g. Google Colab).
  - Fallback: TF-IDF vectors (scikit-learn, no internet needed) -- used
    automatically if sentence-transformers isn't installed, so this
    script still runs anywhere for testing/demo purposes.
"""

import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

DB_PATH = "logs.db"


def load_logs(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT log_id, timestamp, service_name, message, scenario_label FROM logs ORDER BY timestamp")
    rows = cur.fetchall()
    conn.close()
    return rows


def embed_messages(messages):
    """Try real sentence embeddings first; fall back to TF-IDF if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(messages)
        method = "sentence-transformers (all-MiniLM-L6-v2)"
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        embeddings = vectorizer.fit_transform(messages).toarray()
        method = "TF-IDF (fallback -- install sentence-transformers for better results)"
    return np.array(embeddings), method


def recommended_eps(method):
    if "fallback" in method:
        return 1.36
    return 0.4


def cluster_logs(embeddings, eps=0.6, min_samples=2):
    normalized = normalize(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(normalized)
    return labels, normalized


def evaluate(labels, scenario_labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\n--- Clustering Results ---")
    print(f"Clusters found: {n_clusters}  |  Points marked as noise/outlier: {n_noise}")

    unique_labels = set(labels)
    if len(unique_labels) > 1 and n_noise < len(labels):
        try:
            sil = silhouette_score(embeddings_global, labels)
            print(f"Silhouette score: {sil:.3f}  (closer to 1 = better separated clusters)")
        except Exception:
            print("Silhouette score: not computable for this label distribution")

    print("\n--- Ground Truth vs. Cluster Assignment ---")
    from collections import defaultdict
    scenario_to_clusters = defaultdict(list)
    for scenario, cluster_id in zip(scenario_labels, labels):
        scenario_to_clusters[scenario].append(cluster_id)

    for scenario, cluster_ids in scenario_to_clusters.items():
        from collections import Counter
        counts = Counter(cluster_ids)
        dominant_cluster, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / len(cluster_ids) * 100
        print(f"  {scenario:28s} -> mostly cluster {dominant_cluster:>3}  "
              f"({dominant_count}/{len(cluster_ids)} = {purity:.0f}% grouped together)")


if __name__ == "__main__":
    rows = load_logs()
    log_ids = [r[0] for r in rows]
    messages = [r[3] for r in rows]
    scenario_labels = [r[4] for r in rows]

    embeddings, method = embed_messages(messages)
    embeddings_global = embeddings
    print(f"Embedding method used: {method}")
    print(f"Embedded {len(messages)} log messages -> shape {embeddings.shape}")

    eps = recommended_eps(method)
    print(f"Using eps={eps} (tuned for this embedding method)")
    labels, normalized_embeddings = cluster_logs(embeddings, eps=eps, min_samples=2)

    n_total = len(labels)
    n_incidents = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"\nNoise reduction: {n_total} raw log lines -> {n_incidents} distinct incidents "
          f"(+ noise/outliers handled separately)")

    evaluate(labels, scenario_labels)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE logs ADD COLUMN cluster_id INTEGER")
    except Exception:
        pass
    for log_id, cluster_id in zip(log_ids, labels):
        cur.execute("UPDATE logs SET cluster_id = ? WHERE log_id = ?", (int(cluster_id), log_id))
    conn.commit()
    conn.close()
    print(f"\nSaved cluster assignments back to {DB_PATH} (new column: cluster_id)")
