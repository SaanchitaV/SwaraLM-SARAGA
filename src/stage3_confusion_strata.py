"""
Stage 3 -- Confusion analysis: where does movement beat note-content?
=============================================================
Stage 2 showed bag-of-notes and transition-matrix tie ON AVERAGE. That average is
dominated by ragas with distinct note-sets, where note-content already suffices.
The premise of the project -- movement (sanchara) carries identity beyond notes --
should show up specifically on ALLIED ragas (same/similar scale, different
movement). This stage isolates those cases.

Three analyses:
  (A) Confusion pairs: which ragas get retrieved for which? Surfaces the ragas
      each representation actually confuses.
  (B) Scale-overlap stratification: split raga PAIRS by how much their note-sets
      overlap (Jaccard). Compare bag-of-notes vs transition ON HIGH-OVERLAP pairs
      (the allied cases) vs low-overlap pairs. Movement should help more on high
      overlap.
  (C) Same-melakarta / same-scale clusters: retrieval error rate within known
      allied clusters, per representation.

Requires: swara_sequences.jsonl, and the raga theory loader (for scale overlap).
Uses the SAME representations as Stage 2 (imported/redefined here).
-------------------------------------------------------------
"""

import os
import re
import json
import unicodedata
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

SEQ_PATH = "/content/census_out/swara_sequences.jsonl"
XLSX_PATH = "/content/carnatic_ragas_arohana_avarohana.xlsx"
OUT_DIR = "/content/census_out"

POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
IDX = {s: i for i, s in enumerate(POS12)}
SIL = "_"

# reuse the alias map from the visualizer (extend if needed)
ALIASES = {
    "karaharapriya":              "kharaharapriya",
    "suddadhanyasi":              "suddhadhanyasi",
    "cakravakam":                 "chakravakam",
    "simhendra madhyamam":        "simhendramadhyamam",
    "brndavana saranga":          "brindavana saranga",
    "sudda saveri":               "suddha saveri",
    "vagadisvari":                "vagadhisvari",
    "kamavardani/pantuvarali":    "kamavardhani / pantuvarali",
    "purnacandrika":              "purnachandrika",
    "cittaranjani":               "chittaranjani",
}

SYMBOL_TO_POS = {
    "S": "S", "R1": "R1", "R2": "R2", "G1": "R2", "R3": "G2", "G2": "G2",
    "G3": "G3", "M1": "M1", "M2": "M2", "P": "P", "D1": "D1", "D2": "D2",
    "N1": "D2", "D3": "N2", "N2": "N2", "N3": "N3",
}


# ---------- theory (for scale overlap) ----------
def _key(name):
    d = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", d).strip().lower()


def _parse_scale(s):
    out = []
    for tok in str(s).split():
        t = unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode("ascii").strip().upper()
        if not t:
            out.append("S")
        elif t in SYMBOL_TO_POS:
            out.append(SYMBOL_TO_POS[t])
    return out


def load_theory():
    df = pd.read_excel(XLSX_PATH, sheet_name="Ragas")
    T = {}
    for _, row in df.iterrows():
        name = str(row["Raga"]).strip()
        if not name or name.lower() == "nan":
            continue
        allsw = set(_parse_scale(row["Arohana"])) | set(_parse_scale(row["Avarohana"]))
        T[_key(name)] = allsw
    return T


T = load_theory()


def scale_set(raga_key):
    return T.get(ALIASES.get(raga_key, raga_key), set())


# ---------- representations (same as Stage 2) ----------
def bag_of_notes(seq):
    v = np.zeros(12)
    for s in seq:
        if s in IDX:
            v[IDX[s]] += 1
    tot = v.sum()
    return v / tot if tot else v


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


def load_records():
    recs = []
    with open(SEQ_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("raga") and r["raga"].lower() not in ("unknown", "ragamalika"):
                recs.append(r)
    return recs


def cos(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    Xn = X / n
    return Xn @ Xn.T


# =============================================================
# (A) CONFUSION PAIRS
# =============================================================
def confusion_pairs(X, ragas, name, top=15):
    sim = cos(X)
    n = len(ragas)
    counts = Counter(ragas)
    confusions = Counter()
    for i in range(n):
        if counts[ragas[i]] < 2:
            continue
        order = np.argsort(-sim[i])
        order = order[order != i]
        top1 = ragas[order[0]]
        if top1 != ragas[i]:
            pair = tuple(sorted([ragas[i], top1]))
            confusions[pair] += 1
    print(f"\n[{name}] most frequent top-1 CONFUSIONS (retrieved wrong raga):")
    for (a, b), c in confusions.most_common(top):
        j = len(scale_set(a) & scale_set(b))
        u = len(scale_set(a) | scale_set(b)) or 1
        print(f"   {a:22s} <-> {b:22s}  x{c}   scale_overlap={j/u:.2f}")
    return confusions


# =============================================================
# (B) STRATIFY BY SCALE OVERLAP
# =============================================================
def stratified_by_overlap(bon, trans, ragas):
    """For each query, does the top-1 share the query's raga? Bucket queries by
    whether the *confusing* raga (top-1 if wrong) has high scale overlap. Compare
    the two representations on high- vs low-overlap difficulty."""
    def per_rep(X):
        sim = cos(X)
        counts = Counter(ragas)
        # bucket: for wrong top-1s, is the wrong raga high-overlap (allied) or not
        hi_correct = hi_total = lo_correct = lo_total = 0
        for i in range(len(ragas)):
            if counts[ragas[i]] < 2:
                continue
            order = np.argsort(-sim[i]); order = order[order != i]
            top1 = ragas[order[0]]
            correct = int(top1 == ragas[i])
            # difficulty proxy: max scale-overlap this raga has with ANY other raga
            # (allied ragas live in high-overlap neighbourhoods)
            ov = max((len(scale_set(ragas[i]) & scale_set(r)) /
                      (len(scale_set(ragas[i]) | scale_set(r)) or 1))
                     for r in set(ragas) if r != ragas[i])
            if ov >= 0.8:
                hi_total += 1; hi_correct += correct
            else:
                lo_total += 1; lo_correct += correct
        hi = hi_correct / hi_total if hi_total else float("nan")
        lo = lo_correct / lo_total if lo_total else float("nan")
        return hi, lo, hi_total, lo_total

    b_hi, b_lo, hn, ln = per_rep(bon)
    t_hi, t_lo, _, _ = per_rep(trans)
    print("\n=== P@1 stratified by scale-overlap neighbourhood ===")
    print(f"(high-overlap = ragas that share >=80% of notes with some other raga; "
          f"n_hi={hn}, n_lo={ln})")
    df = pd.DataFrame([
        {"representation": "bag_of_notes",      "P@1_high_overlap": round(b_hi, 3), "P@1_low_overlap": round(b_lo, 3)},
        {"representation": "transition_matrix", "P@1_high_overlap": round(t_hi, 3), "P@1_low_overlap": round(t_lo, 3)},
    ])
    print(df.to_string(index=False))
    print("\nKey question: does transition_matrix beat bag_of_notes in the "
          "HIGH-overlap column? That is where movement should matter.")
    df.to_csv(os.path.join(OUT_DIR, "stage3_overlap_stratified.csv"), index=False)
    return df


# =============================================================
# MAIN
# =============================================================
def main():
    recs = load_records()
    ragas = [r["raga"].lower() for r in recs]
    bon = np.array([bag_of_notes(r["collapsed_seq"]) for r in recs])
    trans = np.array([transition_matrix(r["collapsed_seq"]) for r in recs])

    print(f"Loaded {len(recs)} recordings, {len(set(ragas))} ragas.")

    # (A) what gets confused
    confusion_pairs(bon,   ragas, "bag_of_notes")
    confusion_pairs(trans, ragas, "transition_matrix")

    # (B) does movement help specifically on high-overlap (allied) ragas?
    stratified_by_overlap(bon, trans, ragas)

    print("\nInterpretation:")
    print("- If the SAME allied pairs are confused by bag_of_notes but NOT by")
    print("  transition_matrix, that is direct evidence movement separates them.")
    print("- If transition_matrix's P@1 advantage concentrates in the HIGH-overlap")
    print("  column, the aggregate tie was hiding a real effect on allied ragas.")


if __name__ == "__main__":
    main()