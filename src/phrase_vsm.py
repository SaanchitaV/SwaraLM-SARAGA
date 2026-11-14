"""
Phrase sub-study, Step 2 -- PATH 1: exact-phrase TF-IDF VSM retrieval
=============================================================
Each recording -> a bag of its expert-annotated phrase strings. Build a TF-IDF
vector-space model over the phrase vocabulary (the Gulati-style phrase VSM),
retrieve by cosine, evaluate with the SAME protocol as the main paper.

Why this could beat swara-counting: unlike the 12-swara vocabulary (where every
symbol occurs in every recording and IDF is degenerate), phrases are RARE and
raga-specific (probe: same-raga vs diff-raga Jaccard = 57x). So IDF actually
weights here -- discriminative phrases get high weight.

Evaluated on the 100 phrase-annotated tracks (22 evaluable ragas). This is a
SEPARATE sub-study from the main 43-raga comparison; report in its own table.
"""

import os
import random
from collections import Counter, defaultdict

import numpy as np
import mirdata

DATA_HOME = "/content/saraga_carnatic"
OUT_DIR = "/content/census_out"
os.makedirs(OUT_DIR, exist_ok=True)
ds = mirdata.initialize("saraga_carnatic", data_home=DATA_HOME, version="1.5")

EXCLUDE = ("Rāgamālika", "Ragamalika", "?", "")


def phrase_labels(t):
    ph = t.phrases
    if ph is None:
        return []
    ev = getattr(ph, "events", None)
    return [str(e) for e in ev] if ev is not None else []


# ---- collect phrase-annotated, labelled tracks ----
recs = []  # (track_id, [phrases], raga)
for tid in ds.track_ids:
    t = ds.track(tid)
    labs = phrase_labels(t)
    if not labs:
        continue
    raga = "?"
    md = t.metadata or {}
    raagas = md.get("raaga") or []
    if raagas and isinstance(raagas[0], dict):
        raga = raagas[0].get("name", "?")
    if raga in EXCLUDE:
        continue
    recs.append((tid, labs, raga))

ragas = [r for _, _, r in recs]
raga_counts = Counter(ragas)
evaluable = {r for r, c in raga_counts.items() if c >= 2}
print(f"Phrase-annotated labelled tracks: {len(recs)}")
print(f"Evaluable ragas (>=2 tracks): {len(evaluable)}")
print(f"Evaluable queries: {sum(1 for r in ragas if r in evaluable)}")


# ---- build TF-IDF over phrase vocabulary ----
def build_tfidf(docs):
    """docs: list of lists of phrase-strings -> (matrix, vocab)."""
    df = Counter()
    for d in docs:
        for p in set(d):
            df[p] += 1
    vocab = {p: i for i, p in enumerate(sorted(df))}
    N = len(docs)
    idf = np.zeros(len(vocab))
    for p, i in vocab.items():
        idf[i] = np.log((N + 1) / (df[p] + 1)) + 1     # smoothed idf
    X = np.zeros((N, len(vocab)))
    for r, d in enumerate(docs):
        tf = Counter(d)
        for p, c in tf.items():
            X[r, vocab[p]] = c * idf[vocab[p]]
    # L2 normalize
    norm = np.linalg.norm(X, axis=1, keepdims=True); norm[norm == 0] = 1
    return X / norm, vocab


docs = [labs for _, labs, _ in recs]
X, vocab = build_tfidf(docs)
print(f"Phrase vocabulary size: {len(vocab)}")


# ---- retrieval eval (same protocol as main paper) ----
def evaluate(X, ragas, evaluable, name, ks=(1, 3, 5)):
    sim = X @ X.T
    counts = Counter(ragas)
    precisions = {k: [] for k in ks}
    aps, rrs = [], []
    n_eval = 0
    for i in range(len(ragas)):
        if ragas[i] not in evaluable:
            continue
        n_eval += 1
        order = np.argsort(-sim[i]); order = order[order != i]
        rel = np.array([ragas[j] == ragas[i] for j in order], dtype=int)
        for k in ks:
            precisions[k].append(rel[:k].sum() / k)
        if rel.sum():
            cum = np.cumsum(rel)
            prec_at = cum / (np.arange(len(rel)) + 1)
            aps.append((prec_at * rel).sum() / rel.sum())
            rrs.append(1.0 / (np.argmax(rel) + 1))
    row = {"representation": name, "evaluable_queries": n_eval}
    for k in ks:
        row[f"P@{k}"] = round(float(np.mean(precisions[k])), 3)
    row["MAP"] = round(float(np.mean(aps)), 3)
    row["MRR"] = round(float(np.mean(rrs)), 3)
    return row


# ---- also compute bag-of-notes ON THE SAME 100-track set for a fair comparison ----
# (phrases are strings of solfege chars; derive a swara-count vector from them
#  so both representations use the SAME underlying data)
SWARA_CHARS = "SsRrGgMmPpDdNn"
def bag_of_notes_from_phrases(labs):
    v = Counter()
    for p in labs:
        for ch in p:
            if ch in SWARA_CHARS:
                v[ch] += 1
    vec = np.array([v[c] for c in SWARA_CHARS], dtype=float)
    s = vec.sum()
    return vec / s if s else vec

Xbon = np.array([bag_of_notes_from_phrases(labs) for _, labs, _ in recs])
nb = np.linalg.norm(Xbon, axis=1, keepdims=True); nb[nb == 0] = 1
Xbon = Xbon / nb

import pandas as pd
results = [
    evaluate(Xbon, ragas, evaluable, "bag_of_notes (from phrase chars)"),
    evaluate(X,    ragas, evaluable, "phrase_tfidf_vsm (exact)"),
]
df = pd.DataFrame(results)
print("\n=== PHRASE SUB-STUDY: retrieval on 100 phrase-annotated tracks ===")
print(df.to_string(index=False))
print("\nKey comparison: does phrase_tfidf_vsm beat bag_of_notes ON THE SAME tracks?")
print("(This is the fair test -- both built from the same phrase-annotated data.)")
df.to_csv(os.path.join(OUT_DIR, "phrase_vsm_results.csv"), index=False)
print(f"Saved -> {OUT_DIR}/phrase_vsm_results.csv")
