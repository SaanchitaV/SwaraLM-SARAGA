"""
Stage 2 -- Baselines: bag-of-notes & swara-transition retrieval
=============================================================
Builds the two DELIBERATELY-SIMPLE representations the learned model (Stage 4)
must beat, and runs the first real IR evaluation on them.

  Baseline 1  BAG-OF-NOTES  (swara-presence vector)
              12-d: how much of each swara a recording contains.
              Captures WHAT notes, not MOVEMENT. Confuses allied ragas. Strawman.

  Baseline 2  TRANSITION MATRIX  (first-order Markov)
              12x12 -> 144-d flattened: P(next swara | current swara).
              Captures MOVEMENT (which swara follows which) -> encodes early
              sanchara. The "floor" the learned model must exceed.

EVALUATION: same-raga retrieval (query-by-example).
  For each recording, rank all others by similarity; check whether same-raga
  recordings come back on top. Metrics: Precision@k, MAP, MRR.

  Split note: for a fair test we evaluate on the well-populated ragas only, and
  report per-raga so single-recording ragas (no same-raga target) are excluded.

Requires: swara_sequences.jsonl (Stage 1 v2).
-------------------------------------------------------------
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd

SEQ_PATH = "/content/census_out/swara_sequences.jsonl"
OUT_DIR = "/content/census_out"

POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
IDX = {s: i for i, s in enumerate(POS12)}
SIL = "_"


# =============================================================
# LOAD
# =============================================================
def load_records():
    recs = []
    with open(SEQ_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("raga") and (r["raga"].lower() not in ("unknown", "ragamalika")):
                recs.append(r)
    return recs


# =============================================================
# REPRESENTATIONS
# =============================================================
def bag_of_notes(seq):
    """12-d normalized count of swaras (silence ignored)."""
    v = np.zeros(12)
    for s in seq:
        if s in IDX:
            v[IDX[s]] += 1
    tot = v.sum()
    return v / tot if tot > 0 else v


def transition_matrix(seq):
    """12x12 row-normalized first-order transition probs, flattened to 144-d.
    Silence breaks continuity (no transition across a phrase boundary)."""
    M = np.zeros((12, 12))
    prev = None
    for s in seq:
        if s == SIL or s not in IDX:
            prev = None
            continue
        cur = IDX[s]
        if prev is not None:
            M[prev, cur] += 1
        prev = cur
    # row-normalize
    rowsum = M.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1
    M = M / rowsum
    return M.flatten()


def build_representations(recs):
    bon = np.array([bag_of_notes(r["collapsed_seq"]) for r in recs])
    trans = np.array([transition_matrix(r["collapsed_seq"]) for r in recs])
    ragas = [r["raga"].lower() for r in recs]
    artists = [r.get("artist", "UNKNOWN") for r in recs]
    return bon, trans, ragas, artists


# =============================================================
# RETRIEVAL EVAL
# =============================================================
def cosine_sim_matrix(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1
    Xn = X / norm
    return Xn @ Xn.T


def evaluate(X, ragas, name, ks=(1, 3, 5)):
    """Query-by-example: each recording queries the rest, ranked by cosine.
    Relevant = same raga. Only queries whose raga has >=2 recordings count
    (otherwise there is no correct answer to retrieve)."""
    sim = cosine_sim_matrix(X)
    n = len(ragas)
    raga_counts = defaultdict(int)
    for r in ragas:
        raga_counts[r] += 1

    precisions = {k: [] for k in ks}
    aps, rrs = [], []
    evaluable = 0

    for i in range(n):
        if raga_counts[ragas[i]] < 2:
            continue  # no same-raga target exists
        evaluable += 1
        order = np.argsort(-sim[i])
        order = order[order != i]  # drop self
        rel = np.array([ragas[j] == ragas[i] for j in order], dtype=int)

        for k in ks:
            precisions[k].append(rel[:k].sum() / k)

        # average precision
        if rel.sum() > 0:
            cum = np.cumsum(rel)
            prec_at = cum / (np.arange(len(rel)) + 1)
            aps.append((prec_at * rel).sum() / rel.sum())
            # reciprocal rank
            first = np.argmax(rel) + 1
            rrs.append(1.0 / first)

    row = {"representation": name, "evaluable_queries": evaluable}
    for k in ks:
        row[f"P@{k}"] = round(np.mean(precisions[k]), 3)
    row["MAP"] = round(np.mean(aps), 3)
    row["MRR"] = round(np.mean(rrs), 3)
    return row


# =============================================================
# MAIN
# =============================================================
def main():
    recs = load_records()
    print(f"Loaded {len(recs)} labelled recordings "
          f"({len(set(r['raga'].lower() for r in recs))} ragas).")

    bon, trans, ragas, artists = build_representations(recs)

    results = [
        evaluate(bon,   ragas, "bag_of_notes (strawman)"),
        evaluate(trans, ragas, "transition_matrix (floor)"),
    ]
    df = pd.DataFrame(results)
    print("\n=== Same-raga retrieval (higher = better) ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, "stage2_baseline_results.csv"), index=False)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'stage2_baseline_results.csv')}")

    print("\nInterpretation:")
    print("- If transition_matrix beats bag_of_notes, MOVEMENT already helps over")
    print("  note-content alone -- the core premise, shown at baseline level.")
    print("- These are the numbers the learned model (Stage 4) must exceed.")


if __name__ == "__main__":
    main()