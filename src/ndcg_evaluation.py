"""
nDCG with GRADED relevance -- do representations fail musically sensibly?
=============================================================
Binary metrics (P@k, MAP, MRR) treat every wrong retrieval identically.
But retrieving Kambhoji for a Kamas query (same scale, different sanchara) is
a musically defensible error; retrieving Todi is not. Graded relevance
captures this:

    gain 2  = same raga
    gain 1  = ALLIED raga (scale-overlap >= 0.8 with the query raga)
    gain 0  = otherwise

nDCG@k over these grades measures whether a representation's MISTAKES stay in
the right musical neighbourhood -- something no binary metric can see.

Also reports standard BINARY nDCG (gain 1 = same raga) since IR reviewers
expect the conventional number.

Self-contained: paste as ONE cell. Requires frozen jsonl + theory xlsx +
saved .npy embeddings.
-------------------------------------------------------------
"""

import os
import re
import json
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

SEQ_PATH = "/content/census_out/swara_sequences.jsonl"
XLSX_PATH = "/content/carnatic_ragas_arohana_avarohana.xlsx"
OUT_DIR = "/content/census_out"

POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
IDX = {s: i for i, s in enumerate(POS12)}
SIL = "_"
ALLIED_THRESHOLD = 0.8
KS = (1, 3, 5, 10)

ALIASES = {
    "karaharapriya": "kharaharapriya", "suddadhanyasi": "suddhadhanyasi",
    "cakravakam": "chakravakam", "simhendra madhyama": "simhendramadhyamam",
    "simhendra madhyamam": "simhendramadhyamam",
    "brindavani saranga": "brindavana saranga", "brndavana saranga": "brindavana saranga",
    "sudda saveri": "suddha saveri", "vagadeswari": "vagadhisvari",
    "vagadisvari": "vagadhisvari",
    "kamavardhini/panthuvarali": "kamavardhani / pantuvarali",
    "kamavardani/pantuvarali": "kamavardhani / pantuvarali",
    "purnacandrika": "purnachandrika", "cittaranjani": "chittaranjani",
}
SYMBOL_TO_POS = {
    "S": "S", "R1": "R1", "R2": "R2", "G1": "R2", "R3": "G2", "G2": "G2",
    "G3": "G3", "M1": "M1", "M2": "M2", "P": "P", "D1": "D1", "D2": "D2",
    "N1": "D2", "D3": "N2", "N2": "N2", "N3": "N3",
}


def _key(n):
    d = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii")
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
        T[_key(name)] = set(_parse_scale(row["Arohana"])) | set(_parse_scale(row["Avarohana"]))
    return T


T = load_theory()


def scale_set(rk):
    return T.get(ALIASES.get(rk, rk), set())


def overlap(r1, r2):
    s1, s2 = scale_set(r1), scale_set(r2)
    u = len(s1 | s2)
    return len(s1 & s2) / u if u else 0.0


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


def cos(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return (X / n) @ (X / n).T


def dcg(gains):
    return float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))


def ndcg_at_k(gains_ranked, k):
    actual = dcg(np.asarray(gains_ranked[:k], dtype=float))
    ideal = dcg(np.sort(np.asarray(gains_ranked, dtype=float))[::-1][:k])
    return actual / ideal if ideal > 0 else 0.0


def evaluate_ndcg(X, ragas, name, graded=True):
    """graded=True  -> gain 2 same raga, 1 allied, 0 else
       graded=False -> gain 1 same raga, 0 else (conventional binary nDCG)"""
    sim = cos(X)
    counts = Counter(ragas)
    uniq = sorted(set(ragas))
    ov = {(a, b): overlap(a, b) for a in uniq for b in uniq}

    per_k = {k: [] for k in KS}
    for i in range(len(ragas)):
        if counts[ragas[i]] < 2:
            continue
        order = np.argsort(-sim[i]); order = order[order != i]
        gains = []
        for j in order:
            if ragas[j] == ragas[i]:
                gains.append(2 if graded else 1)
            elif graded and ov[(ragas[i], ragas[j])] >= ALLIED_THRESHOLD:
                gains.append(1)
            else:
                gains.append(0)
        for k in KS:
            per_k[k].append(ndcg_at_k(gains, k))

    row = {"representation": name}
    for k in KS:
        row[f"nDCG@{k}"] = round(float(np.mean(per_k[k])), 3)
    return row


def main():
    recs = load_records()
    ragas = [r["raga"].lower() for r in recs]
    print(f"Corpus: {len(recs)} recordings, {len(set(ragas))} ragas.")

    reps = {
        "bag_of_notes":      np.array([bag_of_notes(r["collapsed_seq"]) for r in recs]),
        "transition_matrix": np.array([transition_matrix(r["collapsed_seq"]) for r in recs]),
    }
    for nm, p in [("lstm_lm", "stage4a_lstm_embeddings.npy"),
                  ("transformer_lm", "stage4b_transformer_embeddings.npy"),
                  ("contrastive", "stage5_contrastive_embeddings.npy")]:
        path = os.path.join(OUT_DIR, p)
        if os.path.exists(path):
            X = np.load(path)
            if len(X) == len(recs):
                reps[nm] = X
            else:
                print(f"[skip {nm}] count mismatch")

    binary = pd.DataFrame([evaluate_ndcg(X, ragas, nm, graded=False)
                           for nm, X in reps.items()])
    graded = pd.DataFrame([evaluate_ndcg(X, ragas, nm, graded=True)
                           for nm, X in reps.items()])

    print("\n=== nDCG, BINARY relevance (same raga = 1) ===")
    print(binary.to_string(index=False))
    print("\n=== nDCG, GRADED relevance (same raga = 2, allied = 1) ===")
    print(graded.to_string(index=False))
    print("\nRead: if a representation gains MORE from graded than binary, its")
    print("errors stay in the right musical neighbourhood (allied ragas) rather")
    print("than being arbitrary. Compare the gap per representation.")

    binary.to_csv(os.path.join(OUT_DIR, "ndcg_binary.csv"), index=False)
    graded.to_csv(os.path.join(OUT_DIR, "ndcg_graded.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/ndcg_binary.csv, ndcg_graded.csv")


if __name__ == "__main__":
    main()
