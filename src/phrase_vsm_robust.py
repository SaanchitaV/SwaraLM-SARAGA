"""
Phrase sub-study, Step 4 -- robustness: bootstrap CIs + threshold sweep
=============================================================
Converts the single-split phrase result into a defensible one:

  (A) BOOTSTRAP confidence intervals on P@1/MAP/MRR by resampling the 60
      queries with replacement -> is the phrase advantage real given small N?
  (B) THRESHOLD SWEEP for approximate matching (0.80..0.92) -> is the
      phrase_vsm_approx advantage stable, or an artifact of one threshold?

Reports bag_of_notes | phrase_vsm_exact | phrase_vsm_approx(each threshold)
with bootstrap mean +/- and 95% CI.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import mirdata

DATA_HOME = "/content/saraga_carnatic"
OUT_DIR = "/content/census_out"
ds = mirdata.initialize("saraga_carnatic", data_home=DATA_HOME, version="1.5")

EXCLUDE = ("Rāgamālika", "Ragamalika", "?", "")
SWARA_CHARS = "SsRrGgMmPpDdNn"
THRESHOLDS = [0.80, 0.85, 0.90, 0.92]
N_BOOT = 1000
RNG = np.random.default_rng(42)


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
    rr = md.get("raaga") or []
    if rr and isinstance(rr[0], dict):
        raga = rr[0].get("name", "?")
    if raga in EXCLUDE:
        continue
    recs.append((tid, labs, raga))

ragas = np.array([r for _, _, r in recs])
raga_counts = Counter(ragas)
evaluable = {r for r, c in raga_counts.items() if c >= 2}
query_idx = [i for i in range(len(recs)) if ragas[i] in evaluable]
print(f"Tracks: {len(recs)}   Evaluable ragas: {len(evaluable)}   "
      f"Queries: {len(query_idx)}")


# ---- edit-distance similarity + clustering ----
def norm_sim(a, b):
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    if min(la, lb) / max(la, lb) < 0.75:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i-1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def cluster_vocab(all_phrases, thr):
    freq = Counter(all_phrases)
    heads = []
    canon = {}
    for p in (x for x, _ in freq.most_common()):
        placed = False
        for h in heads:
            if abs(len(h) - len(p)) / max(len(h), len(p)) > (1 - thr):
                continue
            if norm_sim(p, h) >= thr:
                canon[p] = h; placed = True; break
        if not placed:
            heads.append(p); canon[p] = p
    return canon


# ---- TF-IDF ----
def build_tfidf(docs):
    df = Counter()
    for d in docs:
        for p in set(d):
            df[p] += 1
    vocab = {p: i for i, p in enumerate(sorted(df))}
    N = len(docs)
    idf = np.array([np.log((N + 1) / (df[p] + 1)) + 1 for p in vocab])
    X = np.zeros((N, len(vocab)))
    for r, d in enumerate(docs):
        for p, c in Counter(d).items():
            X[r, vocab[p]] = c * idf[vocab[p]]
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return X / n


# ---- per-query metric arrays (so we can bootstrap over queries) ----
def per_query_metrics(X):
    sim = X @ X.T
    p1, ap, rr = [], [], []
    for i in query_idx:
        order = np.argsort(-sim[i]); order = order[order != i]
        rel = (ragas[order] == ragas[i]).astype(int)
        p1.append(rel[:1].sum())
        if rel.sum():
            cum = np.cumsum(rel)
            pa = cum / (np.arange(len(rel)) + 1)
            ap.append((pa * rel).sum() / rel.sum())
            rr.append(1.0 / (np.argmax(rel) + 1))
        else:
            ap.append(0.0); rr.append(0.0)
    return np.array(p1, float), np.array(ap, float), np.array(rr, float)


def boot_ci(arr):
    idx = RNG.integers(0, len(arr), size=(N_BOOT, len(arr)))
    means = arr[idx].mean(axis=1)
    return arr.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


docs_exact = [labs for _, labs, _ in recs]
all_phrases = [p for d in docs_exact for p in d]

def bon(labs):
    v = Counter(ch for p in labs for ch in p if ch in SWARA_CHARS)
    vec = np.array([v[c] for c in SWARA_CHARS], float)
    s = vec.sum(); return vec / s if s else vec
Xbon = np.array([bon(l) for _, l, _ in recs])
nb = np.linalg.norm(Xbon, axis=1, keepdims=True); nb[nb == 0] = 1
Xbon = Xbon / nb

reps = {"bag_of_notes": Xbon,
        "phrase_vsm_exact": build_tfidf(docs_exact)}
vocab_sizes = {"phrase_vsm_exact": len(set(all_phrases))}
for thr in THRESHOLDS:
    canon = cluster_vocab(all_phrases, thr)
    docs_a = [[canon[p] for p in labs] for _, labs, _ in recs]
    reps[f"phrase_approx@{thr}"] = build_tfidf(docs_a)
    vocab_sizes[f"phrase_approx@{thr}"] = len(set(canon.values()))

rows = []
for name, X in reps.items():
    p1, ap, rr = per_query_metrics(X)
    m1, lo1, hi1 = boot_ci(p1)
    ma, loa, hia = boot_ci(ap)
    mr, lor, hir = boot_ci(rr)
    rows.append({
        "representation": name,
        "vocab": vocab_sizes.get(name, "-"),
        "P@1": f"{m1:.3f} [{lo1:.3f},{hi1:.3f}]",
        "MAP": f"{ma:.3f} [{loa:.3f},{hia:.3f}]",
        "MRR": f"{mr:.3f} [{lor:.3f},{hir:.3f}]",
    })

df = pd.DataFrame(rows)
print(f"\n=== PHRASE SUB-STUDY robustness ({N_BOOT} bootstrap resamples, "
      f"95% CI over {len(query_idx)} queries) ===")
print(df.to_string(index=False))
print("\nRead:")
print(" - if phrase_vsm_exact P@1 CI lower bound > bag_of_notes P@1 upper bound,")
print("   the phrase advantage is significant despite small N.")
print(" - if phrase_approx P@1 stays high & similar across thresholds, the")
print("   approximate-matching gain is robust, not a threshold artifact.")
df.to_csv(os.path.join(OUT_DIR, "phrase_vsm_robust.csv"), index=False)
print(f"Saved -> {OUT_DIR}/phrase_vsm_robust.csv")
