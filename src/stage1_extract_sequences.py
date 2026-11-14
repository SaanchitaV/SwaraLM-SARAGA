"""
Stage 1 (v2) -- Feature Extraction with raga-BLIND cleaning
=============================================================
Converts f0 contours -> tonic-normalized swara sequences.

KEY PRINCIPLE: all cleaning uses ONLY the signal's own properties
(smoothing, dwell time, stability). The raga label is NEVER used to modify
the sequence -- that would leak the answer into self-supervised training and
erase the gamaka that carries raga identity.

The arohana/avarohana list appears ONLY in an analysis function that MEASURES
excursion / checks pitch-tracker health. It never touches the extracted sequence.

Outputs to OUT_DIR:
    - swara_sequences.jsonl    (collapsed seq + gamaka profile + histogram)
    - Optional printed analysis comparing histograms to theory (diagnostic only)
-------------------------------------------------------------
"""

import os
import glob
import json
import math

import numpy as np
import pandas as pd


# =============================================================
# CONFIG
# =============================================================
try:
    DATA_PATH
except NameError:
    DATA_PATH = "/kaggle/input/saraga-carnatic-music-dataset"

OUT_DIR = "/content/census_out"
os.makedirs(OUT_DIR, exist_ok=True)

SWARA_LABELS = ["S", "R1", "R2", "G2", "G3", "M1",
                "M2", "P", "D1", "D2", "N2", "N3"]
# positions in cents: index * 100
SILENCE_TOKEN = "_"

# --- raga-BLIND cleaning knobs (tune these; none use the raga label) ---
MEDIAN_FILTER_WIN = 21     # frames; wider => collapses gamaka oscillation onto
                           # the central swara before binning. Was 5 (too light).
MIN_DWELL_FRAMES  = 10     # a swara must be held >= this many frames to count
                           # as a note in the collapsed sequence. Was 3.
STABILITY_CENTS   = 35     # a region is "stable" if its cents std < this


# =============================================================
# FILE HELPERS  (unchanged)
# =============================================================
def load_tonic(folder, stem):
    p = os.path.join(folder, stem + ".ctonic.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        t = f.read().strip().split()
    try:
        return float(t[0])
    except (ValueError, IndexError):
        return None


def load_pitch(pitch_path):
    freqs = []
    with open(pitch_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                freqs.append(float(parts[-1]))
            except ValueError:
                continue
    return np.array(freqs, dtype=float)


def median_filter(x, win):
    if win <= 1:
        return x
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + win]) for i in range(len(x))])


# =============================================================
# CORE: f0 -> cents -> swaras  (raga-blind)
# =============================================================
def f0_to_cents(f0_hz, tonic_hz):
    cents = np.full(len(f0_hz), np.nan)
    voiced = (f0_hz > 0) & (tonic_hz > 0)
    cents[voiced] = 1200.0 * np.log2(f0_hz[voiced] / tonic_hz)
    return cents


def cents_to_swara_idx(c):
    if np.isnan(c):
        return None
    pc = c % 1200.0
    return int(round(pc / 100.0)) % 12


def extract_sequence(f0_hz, tonic_hz):
    """Return collapsed note-level sequence + per-note gamaka profile.

    A 'note' = a run of frames that stay near the same swara long enough
    (>= MIN_DWELL_FRAMES). For each note we record the dominant swara AND the
    spread of cents within it (the gamaka width) + direction of drift. This
    KEEPS the gamaka as signal instead of deleting it.
    """
    # smooth voiced frames (raga-blind) to tame oscillation before binning
    f0_s = f0_hz.copy()
    voiced = f0_hz > 0
    if voiced.sum() > MEDIAN_FILTER_WIN:
        f0_s[voiced] = median_filter(f0_hz[voiced], MEDIAN_FILTER_WIN)

    cents = f0_to_cents(f0_s, tonic_hz)
    frame_swara = [cents_to_swara_idx(c) for c in cents]

    notes = []          # list of dicts: {swara, dwell, gamaka_cents_std, drift}
    collapsed = []      # just the swara labels (the "melodic sentence")
    i, n = 0, len(frame_swara)
    while i < n:
        s = frame_swara[i]
        j = i
        while j < n and frame_swara[j] == s:
            j += 1
        run = j - i
        if s is None:
            # silence run -> phrase boundary (dedup handled after)
            if run >= MIN_DWELL_FRAMES:
                collapsed.append(SILENCE_TOKEN)
                notes.append({"swara": SILENCE_TOKEN})
        else:
            if run >= MIN_DWELL_FRAMES:
                seg = cents[i:j]
                seg = seg[~np.isnan(seg)]
                gwidth = float(np.nanstd(seg)) if len(seg) else 0.0
                drift = float(seg[-1] - seg[0]) if len(seg) >= 2 else 0.0
                lbl = SWARA_LABELS[s]
                collapsed.append(lbl)
                notes.append({
                    "swara": lbl,
                    "dwell": run,
                    "gamaka_cents_std": round(gwidth, 1),
                    "drift_cents": round(drift, 1),
                })
        i = j

    # dedup consecutive silences
    dclean, dnotes = [], []
    for lbl, nd in zip(collapsed, notes):
        if lbl == SILENCE_TOKEN and dclean and dclean[-1] == SILENCE_TOKEN:
            continue
        dclean.append(lbl)
        dnotes.append(nd)

    # swara histogram over voiced FRAMES (dominance check, raga-blind)
    hist = {lbl: 0 for lbl in SWARA_LABELS}
    for s in frame_swara:
        if s is not None:
            hist[SWARA_LABELS[s]] += 1
    total = sum(hist.values()) or 1
    hist_pct = {k: round(100.0 * v / total, 1) for k, v in hist.items()}

    return dclean, dnotes, hist_pct, float(voiced.mean())


# =============================================================
# BUILD
# =============================================================
def build_sequences():
    meta_csv = os.path.join(OUT_DIR, "recordings_metadata.csv")
    meta = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else None

    pitch_files = glob.glob(os.path.join(DATA_PATH, "**", "*.pitch.txt"),
                            recursive=True)
    print(f"Found {len(pitch_files)} recordings with pitch tracks.")

    out_path = os.path.join(OUT_DIR, "swara_sequences.jsonl")
    n_ok, n_skip = 0, 0
    with open(out_path, "w") as out:
        for pf in pitch_files:
            folder = os.path.dirname(pf)
            stem = os.path.basename(pf)[:-len(".pitch.txt")]
            tonic = load_tonic(folder, stem)
            f0 = load_pitch(pf)
            if tonic is None or len(f0) == 0:
                n_skip += 1
                continue

            seq, notes, hist, voiced_frac = extract_sequence(f0, tonic)

            track_id = os.path.relpath(folder, DATA_PATH)
            raga = artist = None
            if meta is not None:
                row = meta[meta["track_id"] == track_id]
                if len(row):
                    raga = row.iloc[0].get("raga")
                    artist = row.iloc[0].get("artist")

            out.write(json.dumps({
                "track_id": track_id, "stem": stem,
                "raga": raga, "artist": artist,
                "tonic_hz": tonic, "voiced_frac": voiced_frac,
                "collapsed_seq": seq,
                "collapsed_len": len(seq),
                "notes": notes,               # gamaka profile per note
                "swara_hist_pct": hist,       # dominance histogram
            }) + "\n")
            n_ok += 1

    print(f"Wrote {n_ok} sequences ({n_skip} skipped) -> {out_path}")
    return out_path


# =============================================================
# SANITY + DOMINANCE CHECK
# =============================================================
def sanity_check(jsonl_path, k=3):
    print("\n--- Sanity check (sequence + dominant swaras) ---")
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= k:
                break
            r = json.loads(line)
            hist = r["swara_hist_pct"]
            top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
            print(f"\n[{r['raga']}] {r['stem']}  (voiced={r['voiced_frac']:.0%})")
            print("   seq :", " ".join(r["collapsed_seq"][:50]), "...")
            print("   top swaras:", ", ".join(f"{s}={p}%" for s, p in top if p > 0))


# =============================================================
# ANALYSIS ONLY -- arohana/avarohana comparison (NEVER modifies data)
# =============================================================
# Theory is DIRECTIONAL: arohana (ascent) and avarohana (descent) are stored
# separately, because a swara legal in descent may be illegal in ascent
# (e.g. Bilahari: Ma & Ni appear only in avarohana). A flat SET would erase
# this asymmetry -- which is itself a core part of sanchara.
#
# Format: raga_key -> {"aro": [ordered swara positions up],
#                      "avaro": [ordered swara positions down]}
# The UNION of aro+avaro = all swaras that appear anywhere.
# Still DIAGNOSTIC ONLY -- never used to modify a sequence.
RAGA_THEORY = {
    "bilahari": {
        "aro":   ["S", "R2", "G3", "P", "D2", "S"],
        "avaro": ["S", "N3", "D2", "P", "M1", "G3", "R2", "S"],
    },
    "mohanam": {
        "aro":   ["S", "R2", "G3", "P", "D2", "S"],
        "avaro": ["S", "D2", "P", "G3", "R2", "S"],
    },
    "mohana": {
        "aro":   ["S", "R2", "G3", "P", "D2", "S"],
        "avaro": ["S", "D2", "P", "G3", "R2", "S"],
    },
    "vasantha": {
        "aro":   ["S", "M1", "G3", "M1", "D2", "N2", "S"],
        "avaro": ["S", "N2", "D2", "M1", "G3", "R1", "S"],
    },
    "behag": {
        "aro":   ["S", "G3", "M1", "P", "N3", "D2", "N3", "S"],
        "avaro": ["S", "N3", "D2", "P", "M2", "G3", "M1", "G3", "R2", "S"],
    },
    "kapi": {
        "aro":   ["S", "R2", "M1", "P", "N2", "S"],
        "avaro": ["S", "N2", "D2", "N2", "P", "M1", "G2", "R2", "S"],
    },
    "dwijawanti":{
        "aro":   ["S", "R2", "G3", "M1", "P", "D2", "S"],
        "avaro": ["S", "N2", "D2", "P", "M1", "G3", "M1", "R2", "G2", "R2", "S"],
    },
    "goula": {
        "aro":   ["S", "R2", "M1", "P", "N3", "S"],
        "avaro": ["S", "N3", "P", "M1", "R2", "G3", "M1", "R2", "S"],
    },
    "saveri":{
        "aro":   ["S", "R2", "M1", "P", "D2", "S"],
        "avaro": ["S", "N3", "D2", "P", "M1", "G3", "R2", "S"],
    },
    "malahari": {
        "aro":   ["S", "R2", "M1", "P", "D2", "S"],
        "avaro": ["S", "D2", "P", "M1", "G3", "R2", "S"],
    },
    "mayamalawagowla": {
        "aro":   ["S", "R2", "G3", "M1", "P", "D2", "N3", "S"],
        "avaro": ["S", "N3", "D2", "P", "M1", "G3", "R2", "S"],
    },
    "padi":{
        "aro":   ["S", "R2", "M1", "P", "N3", "S"],
        "avaro": ["S", "N3", "P", "D2", "P", "M1", "R2", "S"],
    }
}


def _theory_sets(entry):
    """Return (all_swaras, aro_set, avaro_set) from a directional entry."""
    aro = set(entry.get("aro", []))
    avaro = set(entry.get("avaro", []))
    return (aro | avaro), aro, avaro


def analyze_against_theory(jsonl_path):
    """Diagnostic. Measures:
      - on-scale % (energy on swaras that appear anywhere in the raga)
      - excursion % (energy off the raga's notes entirely -> gamaka or tracking err)
    Directional legality (a swara appearing in the WRONG direction) needs the
    ordered sequence, not just the histogram -- see analyze_directional() below.
    """
    if not RAGA_THEORY:
        print("\n[analysis skipped] RAGA_THEORY empty. Fill directional "
              "aro/avaro to measure on-scale vs. excursion.")
        return
    print("\n--- On-scale vs. excursion energy (DIAGNOSTIC ONLY) ---")
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            rk = (r["raga"] or "").strip().lower()
            if rk not in RAGA_THEORY:
                continue
            all_sw, _, _ = _theory_sets(RAGA_THEORY[rk])
            hist = r["swara_hist_pct"]
            on = sum(v for k, v in hist.items() if k in all_sw)
            off = sum(v for k, v in hist.items() if k not in all_sw and k != SILENCE_TOKEN)
            rows.append({
                "raga": r["raga"], "stem": r["stem"][:40],
                "on_scale_%": round(on, 1), "excursion_%": round(off, 1),
                "flag": "CHECK" if off > 25 else "",
            })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nNote: high excursion% is NOT necessarily wrong -- gamaka lives "
              "off-scale. Use to spot tracking errors, not to clean.")


def analyze_directional(jsonl_path):
    """Diagnostic. Uses the ORDERED sequence to check directional legality:
    for each ascending step (pitch rising), is the swara legal in arohana?
    for each descending step, legal in avarohana? Reports the fraction of
    direction-consistent notes. This is where aro/avaro SEPARATION matters --
    it catches a swara sung in the 'wrong' direction, which the histogram can't.
    Still diagnostic only.
    """
    if not RAGA_THEORY:
        return
    order = {lbl: i for i, lbl in enumerate(SWARA_LABELS)}
    print("\n--- Directional legality (DIAGNOSTIC ONLY) ---")
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            rk = (r["raga"] or "").strip().lower()
            if rk not in RAGA_THEORY:
                continue
            _, aro, avaro = _theory_sets(RAGA_THEORY[rk])
            seq = [s for s in r["collapsed_seq"] if s != SILENCE_TOKEN]
            legal = total = 0
            for a, b in zip(seq, seq[1:]):
                if a not in order or b not in order:
                    continue
                # ascending step -> b should be legal in arohana;
                # descending step -> b should be legal in avarohana
                if order[b] > order[a]:
                    total += 1; legal += (b in aro)
                elif order[b] < order[a]:
                    total += 1; legal += (b in avaro)
            if total:
                rows.append({
                    "raga": r["raga"], "stem": r["stem"][:40],
                    "direction_consistent_%": round(100.0 * legal / total, 1),
                })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nLower % = more notes sung against the raga's directional grammar. "
              "Some is normal (vakra phrases, gamaka); very low may = tracking error "
              "OR a genuinely vakra raga the simple aro/avaro model doesn't capture.")


if __name__ == "__main__":
    p = build_sequences()
    sanity_check(p)
    analyze_against_theory(p)
    analyze_directional(p)