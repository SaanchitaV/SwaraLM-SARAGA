"""
make_demo.py: build demo/index.html, the single-file retrieval explorer.

The page has its data baked in, so it works when opened straight from disk
(double-click) and on GitHub Pages, with no server and no separate data file.

Inputs (all in this repository):
  results/recordings_metadata.csv         recording titles, ragas, artists
  models/*_embeddings.npy                 learned embeddings, one row per labelled recording
  data/carnatic_ragas_arohana_avarohana.xlsx   raga theory table (for "allied raga" marking)
  data/swara_sequences_v2_FROZEN.jsonl    OPTIONAL: if present, the two count-based
                                          representations (histogram, transitions) are added

Run from the repository root:  python src/make_demo.py
"""

import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "results", "recordings_metadata.csv")
THEORY = os.path.join(ROOT, "data", "carnatic_ragas_arohana_avarohana.xlsx")
SEQS = os.path.join(ROOT, "data", "swara_sequences_v2_FROZEN.jsonl")
TEMPLATE = os.path.join(ROOT, "demo", "template.html")
OUT = os.path.join(ROOT, "demo", "index.html")

POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
IDX = {s: i for i, s in enumerate(POS12)}
ALLIED = 0.8

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
SYM = {"S": "S", "R1": "R1", "R2": "R2", "G1": "R2", "R3": "G2", "G2": "G2", "G3": "G3",
       "M1": "M1", "M2": "M2", "P": "P", "D1": "D1", "D2": "D2", "N1": "D2", "D3": "N2",
       "N2": "N2", "N3": "N3"}


def key(n):
    d = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", d).strip().lower()


def scale(s):
    out = []
    for tok in str(s).split():
        t = unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode("ascii").strip().upper()
        out.append("S" if not t else SYM.get(t))
    return {x for x in out if x}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


# recordings, in the same order as the embeddings
m = pd.read_csv(META)
lab = m[~m.raga_key.isin(["unknown", "ragamalika"])].reset_index(drop=True)
ragas = [key(r) for r in lab.raga]

# theory: scale sets, display strings, allied pairs
th = pd.read_excel(THEORY)
T, SHOW = {}, {}
for _, r in th.iterrows():
    k = key(r["Raga"])
    T[k] = scale(r["Arohana"]) | scale(r["Avarohana"])
    SHOW[k] = {"aro": str(r["Arohana"]), "ava": str(r["Avarohana"])}
tk = lambda r: ALIASES.get(r, r)
uniq = sorted(set(ragas))
allied = {}
for a in uniq:
    sa = T.get(tk(a), set())
    nb = []
    for b in uniq:
        sb = T.get(tk(b), set())
        if a != b and sa and sb and len(sa & sb) / len(sa | sb) >= ALLIED:
            nb.append(b)
    allied[a] = nb

reps = {}
for name, f, label in [("contrastive", "stage5_contrastive_embeddings.npy", "Contrastive"),
                       ("lstm", "stage4a_lstm_embeddings.npy", "LSTM language model"),
                       ("transformer", "stage4b_transformer_embeddings.npy", "Transformer language model")]:
    X = np.load(os.path.join(ROOT, "models", f))
    assert len(X) == len(lab), f"{f}: {len(X)} rows, expected {len(lab)}"
    reps[name] = {"label": label, "kind": "learned", "E": np.round(l2(X), 4).tolist()}

if os.path.exists(SEQS):
    recs = [json.loads(l) for l in open(SEQS, encoding="utf-8")]
    recs = [r for r in recs if r.get("raga") and r["raga"].lower() not in ("unknown", "ragamalika")]
    assert [key(r["raga"]) for r in recs] == ragas, "sequence file order does not match the metadata"
    H, M = [], []
    for r in recs:
        seq = r["collapsed_seq"]
        h = np.zeros(12)
        t = np.zeros((12, 12))
        prev = None
        for s in seq:
            if s not in IDX:
                prev = None
                continue
            h[IDX[s]] += 1
            if prev is not None:
                t[prev, IDX[s]] += 1
            prev = IDX[s]
        rs = t.sum(1, keepdims=True)
        rs[rs == 0] = 1
        H.append(h)
        M.append((t / rs).ravel())
    reps = {"transitions": {"label": "Transition matrix", "kind": "count", "E": np.round(l2(np.array(M)), 4).tolist()},
            "histogram": {"label": "Pitch-class histogram", "kind": "count", "E": np.round(l2(np.array(H)), 4).tolist()},
            **reps}
    print("added count-based representations from", os.path.basename(SEQS))
else:
    print("no sequence file: count-based representations left out")

data = {
    "recordings": [{"t": str(r.title), "r": str(r.raga), "k": k, "a": str(r.artist)}
                   for r, k in zip(lab.itertuples(), ragas)],
    "theory": {k: SHOW[tk(k)] for k in uniq if tk(k) in SHOW},
    "allied": allied,
    "reps": reps,
}
html = open(TEMPLATE, encoding="utf-8").read()
assert "/*__SWARALM_DATA__*/null" in html
html = html.replace("/*__SWARALM_DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
open(OUT, "w", encoding="utf-8").write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB): {len(lab)} recordings, reps = {list(reps)}")
