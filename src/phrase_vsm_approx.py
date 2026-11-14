"""
Phrase sub-study, Step 3 -- PATH 2: approximate phrase matching, then VSM
=============================================================
Exact-match VSM (path 1) treats 'ndnsndpm' and 'ndnsndppm' as different terms,
so transcription variance suppresses same-raga overlap. Path 2 first CLUSTERS
near-duplicate phrases into canonical terms (edit-distance / normalized
similarity -- the same fuzzy-matching idea used for BOM record linkage), then
rebuilds the TF-IDF VSM over clustered terms.

The delta (path2 - path1) quantifies how much transcription variance was
costing the phrase representation. If path2 >> path1: phrases are bottlenecked
by transcription consistency, not by being uninformative.

Compares three on the SAME 94-track / 60-query set:
  bag_of_notes (chars) | phrase_vsm_exact | phrase_vsm_approx
"""

import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import mirdata

DATA_HOME = "/content/saraga_carnatic"
OUT_DIR = "/content/census_out"
ds = mirdata.initialize("saraga_carnatic", data_home=DATA_HOME, version="1.5")

EXCLUDE = ("Rāgamālika", "Ragamalika", "?", "")
SWARA_CHARS = "SsRrGgMmPpDdNn"
SIM_THRESHOLD = 0.85     # phrases with normalized similarity >= this merge


def phrase_labels(t):
    ph = t.phrases
    if ph is None:
        return []
    ev = getattr(ph, "events", None)
    return [str(e) for e in ev] if ev is not None else []


# ---- collect ----
recs = []
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
print(f"Tracks: {len(recs)}   Evaluable ragas: {len(evaluable)}   "
      f"Queries: {sum(1 for r in ragas if r in evaluable)}")


# ============ approximate matching: cluster the phrase vocabulary ============
def norm_sim(a, b):
    """Normalized similarity via Levenshtein distance (1 - dist/maxlen).
    Simple DP edit distance; vocab is small enough (~535)."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # early length-ratio gate (cheap): very different lengths can't be >=0.85
    if min(la, lb) / max(la, lb) < SIM_THRESHOLD:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def cluster_vocabulary(all_phrases):
    """Greedy single-linkage clustering: map each phrase to a canonical form.
    Sort by frequency so the most common variant becomes the cluster head."""
    freq = Counter(all_phrases)
    uniq = [p for p, _ in freq.most_common()]   # frequent first
    canon = {}                                   # phrase -> canonical head
    heads = []
    for p in uniq:
        placed = False
        # only compare against existing heads of similar length (speed)
        for h in heads:
            if abs(len(h) - len(p)) / max(len(h), len(p)) > (1 - SIM_THRESHOLD):
                continue
            if norm_sim(p, h) >= SIM_THRESHOLD:
                canon[p] = h
                placed = True
                break
        if not placed:
            heads.append(p)
            canon[p] = p
    return canon


all_phrases = [p for _, labs, _ in recs for p in labs]
print(f"Distinct phrases (exact): {len(set(all_phrases))}")
canon = cluster_vocabulary(all_phrases)
print(f"Canonical phrases (after approx merge @ {SIM_THRESHOLD}): "
      f"{len(set(canon.values()))}")


# ============ TF-IDF builder (shared) ============
def build_tfidf(docs):
    df = Counter()
    for d in docs:
        for p in set(d):
            df[p] += 1
    vocab = {p: i for i, p in enumerate(sorted(df))}
    N = len(docs)
    idf = np.zeros(len(vocab))
    for p, i in vocab.items():
        idf[i] = np.log((N + 1) / (df[p] + 1)) + 1
    X = np.zeros((N, len(vocab)))
    for r, d in enumerate(docs):
        for p, c in Counter(d).items():
            X[r, vocab[p]] = c * idf[vocab[p]]
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return X / n


docs_exact = [labs for _, labs, _ in recs]
docs_approx = [[canon[p] for p in labs] for _, labs, _ in recs]

X_exact = build_tfidf(docs_exact)
X_approx = build_tfidf(docs_approx)

# bag of notes from chars
def bon(labs):
    v = Counter(ch for p in labs for ch in p if ch in SWARA_CHARS)
    vec = np.array([v[c] for c in SWARA_CHARS], dtype=float)
    s = vec.sum()
    return vec / s if s else vec
Xbon = np.array([bon(labs) for _, labs, _ in recs])
nb = np.linalg.norm(Xbon, axis=1, keepdims=True); nb[nb == 0] = 1
Xbon = Xbon / nb


# ============ eval ============
def evaluate(X, name, ks=(1, 3, 5)):
    sim = X @ X.T
    precisions = {k: [] for k in ks}
    aps, rrs = [], []
    ne = 0
    for i in range(len(ragas)):
        if ragas[i] not in evaluable:
            continue
        ne += 1
        order = np.argsort(-sim[i]); order = order[order != i]
        rel = np.array([ragas[j] == ragas[i] for j in order], dtype=int)
        for k in ks:
            precisions[k].append(rel[:k].sum() / k)
        if rel.sum():
            cum = np.cumsum(rel)
            pa = cum / (np.arange(len(rel)) + 1)
            aps.append((pa * rel).sum() / rel.sum())
            rrs.append(1.0 / (np.argmax(rel) + 1))
    row = {"representation": name, "evaluable_queries": ne}
    for k in ks:
        row[f"P@{k}"] = round(float(np.mean(precisions[k])), 3)
    row["MAP"] = round(float(np.mean(aps)), 3)
    row["MRR"] = round(float(np.mean(rrs)), 3)
    return row


results = [
    evaluate(Xbon, "bag_of_notes (chars)"),
    evaluate(X_exact, "phrase_vsm_exact"),
    evaluate(X_approx, f"phrase_vsm_approx (>={SIM_THRESHOLD})"),
]
df = pd.DataFrame(results)
print("\n=== PHRASE SUB-STUDY: exact vs approximate matching ===")
print(df.to_string(index=False))
print(f"\nvocab: {len(set(all_phrases))} exact -> {len(set(canon.values()))} clustered")
print("delta (approx - exact) P@1 quantifies transcription-variance cost.")
df.to_csv(os.path.join(OUT_DIR, "phrase_vsm_approx_results.csv"), index=False)
print(f"Saved -> {OUT_DIR}/phrase_vsm_approx_results.csv")
