import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict

conn = sqlite3.connect("logs.db")
cur = conn.cursor()
cur.execute("SELECT message, scenario_label FROM logs ORDER BY timestamp")
rows = cur.fetchall()
messages = [r[0] for r in rows]
labels_true = [r[1] for r in rows]

model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode(messages)
Xn = normalize(X)

for eps in [0.95, 1.0, 1.02, 1.05, 1.08, 1.1, 1.12, 1.15]:
    db = DBSCAN(eps=eps, min_samples=2, metric="euclidean")
    pred = db.fit_predict(Xn)
    n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    n_noise = list(pred).count(-1)

    scenario_to_clusters = defaultdict(list)
    for s, c in zip(labels_true, pred):
        if s != "noise":
            scenario_to_clusters[s].append(c)

    print(f"\n=== eps={eps}  (clusters={n_clusters}, noise={n_noise}) ===")
    for s, cids in scenario_to_clusters.items():
        top, cnt = Counter(cids).most_common(1)[0]
        print(f"  {s:28s} -> dominant cluster {top:>3}  purity={cnt/len(cids):.2f}")
