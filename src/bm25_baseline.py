"""
BM25 baseline -- sparse lexical scoring over swara sequences
=============================================================
Adds one row to the results table: BM25 (Okapi) scoring with each recording as
a "document" (its swara token sequence) and each query recording represented
by its DISTINCT swara tokens (short-query analogue).

Why this is not the default baseline (state this in the paper): BM25 assumes
short queries against longer documents; our task is symmetric
(recording-vs-recording), where cosine over normalized count vectors is the
standard choice. This run answers the inevitable "why not BM25" question with
a number, and probes whether IDF is nearly degenerate on a 12-symbol
vocabulary (every swara appears in almost every recording).

Install:  pip install rank_bm25
Requires: the frozen swara_sequences jsonl.
Output:   printed metrics row + bm25_baseline_results.csv
-------------------------------------------------------------
"""

import os
import json
from collections import Counter

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

SEQ_PATH = "/content/census_out/swara_sequences.jsonl"
OUT_DIR = "/content/census_out"

POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
SIL = "_"

# BM25 parameters -- report defaults; optionally grid a couple of values.
K1_VALUES = [1.2]        # extend to [0.5, 1.2, 2.0] for a mini sensitivity check
B_VALUES = [0.75]        # extend to [0.0, 0.75] likewise


def load_records():
    recs = []
    with open(SEQ_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("raga") and r["raga"].lower() not in ("unknown", "ragamalika"):
                recs.append(r)
    return recs


def tokens(seq):
    """Document tokens: the swara sequence with silences removed."""
    return [s for s in seq if s != SIL and s in POS12]


def query_terms(seq):
    """Query representation: DISTINCT swaras, ordered by frequency (the
    short-query analogue of a whole-recording query)."""
    c = Counter(tokens(seq))
    return [s for s, _ in c.most_common()]


def evaluate_bm25(recs, k1, b, ks=(1, 3, 5)):
    docs = [tokens(r["collapsed_seq"]) for r in recs]
    ragas = [r["raga"].lower() for r in recs]
    bm25 = BM25Okapi(docs, k1=k1, b=b)
    counts = Counter(ragas)

    precisions = {k: [] for k in ks}
    aps, rrs = [], []
    evaluable = 0
    for i in range(len(recs)):
        if counts[ragas[i]] < 2:
            continue
        evaluable += 1
        scores = bm25.get_scores(query_terms(recs[i]["collapsed_seq"]))
        scores[i] = -np.inf                      # exclude self
        order = np.argsort(-scores)
        rel = np.array([ragas[j] == ragas[i] for j in order], dtype=int)
        for k in ks:
            precisions[k].append(rel[:k].sum() / k)
        if rel.sum():
            cum = np.cumsum(rel)
            prec_at = cum / (np.arange(len(rel)) + 1)
            aps.append((prec_at * rel).sum() / rel.sum())
            rrs.append(1.0 / (np.argmax(rel) + 1))

    row = {"representation": f"bm25 (k1={k1}, b={b})",
           "evaluable_queries": evaluable}
    for k in ks:
        row[f"P@{k}"] = round(float(np.mean(precisions[k])), 3)
    row["MAP"] = round(float(np.mean(aps)), 3)
    row["MRR"] = round(float(np.mean(rrs)), 3)
    return row


def idf_diagnostics(recs):
    """How degenerate is IDF on a 12-symbol vocabulary? Report per-swara
    document frequency -- if every swara appears in ~every recording, IDF is
    near-constant and BM25 reduces to saturated TF + length norm."""
    n = len(recs)
    df = Counter()
    for r in recs:
        for s in set(tokens(r["collapsed_seq"])):
            df[s] += 1
    print("\n--- IDF diagnostics (document frequency per swara) ---")
    for s in POS12:
        print(f"  {s:3s}: appears in {df[s]}/{n} recordings "
              f"({100*df[s]/n:.0f}%)")
    print("If most are ~100%, IDF is near-degenerate -- state this in the paper.")


def main():
    recs = load_records()
    print(f"Loaded {len(recs)} recordings.")

    rows = []
    for k1 in K1_VALUES:
        for b in B_VALUES:
            rows.append(evaluate_bm25(recs, k1, b))
    df = pd.DataFrame(rows)
    print("\n=== BM25 baseline ===")
    print(df.to_string(index=False))
    print("\nReference (same harness): bag_of_notes 0.423 | transitions 0.431 | "
          "lstm_lm 0.358 | contrastive 0.423 (single-split)")
    df.to_csv(os.path.join(OUT_DIR, "bm25_baseline_results.csv"), index=False)

    idf_diagnostics(recs)


if __name__ == "__main__":
    main()
