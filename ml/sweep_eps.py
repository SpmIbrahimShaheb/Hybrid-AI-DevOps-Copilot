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

print("Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode(messages)
Xn = normalize(X)

for eps in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
    db = DBSCAN(eps=eps, min_samples=2, metric="euclidean")
    pred = db.fit_predict(Xn)
    n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    n_noise = list(pred).count(-1)

    scenario_to_clusters = defaultdict(list)
    for s, c in zip(labels_true, pred):
        if s != "noise":
            scenario_to_clusters[s].append(c)

    dominants = {}
    for s, cids in scenario_to_clusters.items():
        top, cnt = Counter(cids).most_common(1)[0]
        dominants[s] = (top, round(cnt/len(cids), 2))
    unique_clusters = len(set(d[0] for d in dominants.values()))

    print(f"eps={eps}: clusters={n_clusters} noise={n_noise} unique_incident_clusters={unique_clusters}/4  purity={ {k: v[1] for k,v in dominants.items()} }")
