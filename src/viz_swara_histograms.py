"""
COMBINED CELL: raga-theory loader + enharmonic-aware swara histogram visualizer.
Paste this whole thing into ONE Colab cell and run.

Requires:
  - /content/census_out/swara_sequences.jsonl   (from Stage 1 v2)
  - /content/carnatic_ragas_arohana_avarohana.xlsx   (upload it; adjust path if needed)
"""

import os
import re
import json
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================
# PATHS  -- adjust if yours differ
# =============================================================
XLSX_PATH = "/content/carnatic_ragas_arohana_avarohana.xlsx"
SEQ_PATH  = "/content/census_out/swara_sequences.jsonl"

assert os.path.exists(XLSX_PATH), f"Spreadsheet not found at {XLSX_PATH} -- upload it / fix path"
assert os.path.exists(SEQ_PATH),  f"Sequences not found at {SEQ_PATH} -- run Stage 1 v2 first"

ALIASES = {
    "karaharapriya": "kharaharapriya",
    "suddadhanyasi": "suddhadhanyasi",
    "cakravakam": "chakravakam",
    "simhendra madhyama": "simhendramadhyamam",
    "brindavani saranga": "brindavana saranga",
    "sudda saveri": "suddha saveri",
    "vagadeswari": "vagadhisvari",
    "kamavardhini/panthuvarali": "kamavardhani / pantuvarali",
    "purnacandrika": "purnachandrika",
    "cittaranjani": "chittaranjani",
    # ... fill from the `missing` list, matching to sheet keys
}


# =============================================================
# 12 pitch positions  (cents = index * 100)
# =============================================================
POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]

# map every full-notation symbol to its 12-position label (enharmonics folded
# to the shared PITCH; the functional distinction is preserved for display only)
SYMBOL_TO_POS = {
    "S": "S",
    "R1": "R1",
    "R2": "R2", "G1": "R2",     # 200 cents shared
    "R3": "G2", "G2": "G2",     # 300 cents shared
    "G3": "G3",
    "M1": "M1",
    "M2": "M2",
    "P": "P",
    "D1": "D1",
    "D2": "D2", "N1": "D2",     # 900 cents shared
    "D3": "N2", "N2": "N2",     # 1000 cents shared
    "N3": "N3",
}

# display labels showing both enharmonic names on shared bins
POS12_DISPLAY = {
    "S": "S", "R1": "R1", "R2": "R2/G1", "G2": "G2/R3", "G3": "G3",
    "M1": "M1", "M2": "M2", "P": "P", "D1": "D1", "D2": "D2/N1",
    "N2": "N2/D3", "N3": "N3",
}
DISPLAY_LABELS = [POS12_DISPLAY[p] for p in POS12]


# =============================================================
# LOAD RAGA THEORY FROM SPREADSHEET
# =============================================================
def _key(name):
    d = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", d).strip().lower()


def _parse_scale(scale_str):
    labels = []
    for tok in scale_str.split():
        t = unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode("ascii").strip()
        if not t:                      # upper-octave Sa marker -> S
            labels.append("S")
            continue
        t = t.upper()
        if t in SYMBOL_TO_POS:
            labels.append(SYMBOL_TO_POS[t])
    return labels


def load_raga_theory(xlsx_path=XLSX_PATH):
    df = pd.read_excel(xlsx_path, sheet_name="Ragas")
    theory = {}
    for _, row in df.iterrows():
        name = str(row["Raga"]).strip()
        if not name or name.lower() == "nan":
            continue
        aro = _parse_scale(str(row["Arohana"]))
        avaro = _parse_scale(str(row["Avarohana"]))
        theory[_key(name)] = {
            "display": name,
            "aro": aro, "avaro": avaro,
            "all": set(aro) | set(avaro),
            "confidence": str(row.get("Confidence", "")).strip(),
            "notes": str(row.get("Notes", "")).strip(),
        }
    return theory


T = load_raga_theory()
print(f"Loaded theory for {len(T)} ragas.")


# =============================================================
# VISUALIZE
# =============================================================
def _legal_positions(raga_key):
    entry = ALIASES.get(raga_key)
    return set(entry["all"]) if entry else set()


def load_by_raga():
    by = defaultdict(list)
    with open(SEQ_PATH) as f:
        for line in f:
            r = json.loads(line)
            rk = (r.get("raga") or "").lower()
            by[rk].append(r)
    return by


_BY_RAGA = load_by_raga()


def plot_raga_average(raga_key, ax=None):
    recs = _BY_RAGA.get(raga_key, [])
    if not recs:
        if ax is None:
            print(f"No recordings for {raga_key}")
        return
    legal = _legal_positions(raga_key)
    mat = np.array([[r["swara_hist_pct"].get(s, 0) for s in POS12] for r in recs])
    mean = mat.mean(axis=0)
    colors = ["#2e8b57" if s in legal else "#c0392b" for s in POS12]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(DISPLAY_LABELS, mean, color=colors)
    off = sum(v for s, v in zip(POS12, mean) if s not in legal)
    entry = T.get(raga_key, {})
    disp = entry.get("display", raga_key)
    conf = entry.get("confidence", "?")
    tag = f"n={len(recs)}" + (" (single!)" if len(recs) == 1 else "")
    ax.set_title(f"{disp} [{conf}] {tag}  excursion={off:.0f}%", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.tick_params(axis="x", rotation=45)
    if standalone:
        plt.tight_layout(); plt.show()


def plot_all(save_path="/content/census_out/all_ragas_histograms.png"):
    ordered = sorted(_BY_RAGA.items(), key=lambda kv: -len(kv[1]))
    n = len(ordered)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2.5 * rows))
    axes = axes.flatten()
    i = -1
    for i, (rk, _) in enumerate(ordered):
        plot_raga_average(rk, ax=axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"Saved grid -> {save_path}")


# =============================================================
# RUN
# =============================================================
plot_raga_average("mohanam")   # control: pentatonic, should be mostly green
plot_all()                     # the full wall