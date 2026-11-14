"""
make_demo_data.py -- export embeddings + metadata to docs/demo_data.json
for the static GitHub Pages retrieval demo.

Run in the same environment as the pipeline (needs the frozen jsonl and the
saved .npy embeddings). Output: one JSON the demo page fetches.
"""

import os
import json
import numpy as np

SEQ_PATH = "/content/census_out/swara_sequences.jsonl"
OUT_DIR = "/content/census_out"
OUT_JSON = os.path.join(OUT_DIR, "demo_data.json")

POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
IDX = {s: i for i, s in enumerate(POS12)}
SIL = "_"


def load_records():
    recs = []
    with open(SEQ_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("raga") and r["raga"].lower() not in ("unknown", "ragamalika"):
                recs.append(r)
    return recs


def bag_of_notes(seq):
    v = np.zeros(12)
    for s in seq:
        if s in IDX:
            v[IDX[s]] += 1
    t = v.sum()
    return v / t if t else v


def transition_matrix(seq):
    M = np.zeros((12, 12))
    prev = None
    for s in seq:
        if s == SIL or s not in IDX:
            prev = None
            continue
        c = IDX[s]
        if prev is not None:
            M[prev, c] += 1
        prev = c
    rs = M.sum(1, keepdims=True); rs[rs == 0] = 1
    return (M / rs).flatten()


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


recs = load_records()
print(f"{len(recs)} recordings")

reps = {
    "bag_of_notes": np.array([bag_of_notes(r["collapsed_seq"]) for r in recs]),
    "transitions":  np.array([transition_matrix(r["collapsed_seq"]) for r in recs]),
}
for name, path in [("lstm_lm", "stage4a_lstm_embeddings.npy"),
                   ("contrastive", "stage5_contrastive_embeddings.npy")]:
    p = os.path.join(OUT_DIR, path)
    if os.path.exists(p):
        X = np.load(p)
        if len(X) == len(recs):
            reps[name] = X
        else:
            print(f"[skip {name}] count mismatch")

data = {
    "recordings": [
        {"title": r["stem"], "raga": r["raga"], "artist": r.get("artist", "?")}
        for r in recs
    ],
    # store L2-normalized, rounded to 4dp to keep the JSON small
    "embeddings": {
        name: np.round(l2norm(X), 4).tolist() for name, X in reps.items()
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(data, f)
size_mb = os.path.getsize(OUT_JSON) / 1e6
print(f"Wrote {OUT_JSON}  ({size_mb:.1f} MB) -> copy to repo docs/demo_data.json")
