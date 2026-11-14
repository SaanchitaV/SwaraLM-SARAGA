"""
Saraga Carnatic v1.5 -- download + mirdata census + phrase-coverage probe
=============================================================
Full-dataset (249-track) pre-flight, all in one runnable cell.

Handles: downloading via mirdata (~14GB, once), checksum validation, coverage
of tonic/pitch/pitch_vocal/phrases, FRAME RATE for Section 3, phrase solfege
vocabulary (phrase-representation viability), raga/artist distribution.

STORAGE (Colab): the 14GB audio is INPUT; the extracted sequences are the
artifact. Download to ephemeral /content (fine for one session), then save only
the small outputs to Drive. With paid Drive (>30GB), set DATA_HOME to a Drive
path to persist the audio.

Set DOWNLOAD=False on later runs in the same session.
"""

import os
import csv
from collections import Counter

import numpy as np
import mirdata

# ============ CONFIG ============
DATA_HOME = "/content/saraga_carnatic"   # ephemeral; Drive path to persist
DOWNLOAD = True                          # False if already downloaded this session
OUT_DIR = "/content/census_out"
os.makedirs(OUT_DIR, exist_ok=True)

ds = mirdata.initialize("saraga_carnatic", data_home=DATA_HOME, version="1.5")

# ---- download ----
if DOWNLOAD:
    print("Downloading Saraga Carnatic v1.5 (~14GB, one time)...")
    try:
        ds.download(cleanup=True)
    except Exception as e:
        print(f"download() raised: {e}")
        try:
            print("valid remote keys:", list(ds.remotes.keys()))
        except Exception:
            pass
        raise

# ---- validate ----
print("\nValidating (checksums)...")
missing, invalid = ds.validate(verbose=False)
print(f"  missing: {len(missing)}   invalid: {len(invalid)}")

track_ids = ds.track_ids
n = len(track_ids)
print(f"\nTotal tracks: {n}")

print("\n--- CITATION ---")
try: ds.cite()
except Exception: pass
print("\n--- LICENSE ---")
try: ds.license()
except Exception: pass

# ============ PROBE ============
has_tonic = has_pitch = has_pitch_vocal = has_phrases = has_sections = 0
frame_hops = []
phrase_symbols = Counter()
phrase_ngrams = Counter()
raga_counter = Counter()
artist_counter = Counter()
multi_raga = 0
rows = []

for tid in track_ids:
    t = ds.track(tid)
    tonic_ok = pitch_ok = pv_ok = ph_ok = sec_ok = False

    try: tonic_ok = t.tonic is not None
    except Exception: pass

    try:
        if t.pitch is not None and len(t.pitch.times) > 1:
            pitch_ok = True
            frame_hops.append(float(t.pitch.times[1] - t.pitch.times[0]))
    except Exception: pass

    try: pv_ok = t.pitch_vocal is not None and len(t.pitch_vocal.times) > 1
    except Exception: pass

    try:
        if t.phrases is not None and len(t.phrases.intervals) > 0:
            ph_ok = True
            for lab in t.phrases.labels:
                s = str(lab)
                phrase_ngrams[s] += 1
                for ch in s:
                    if ch in "SsRrGgMmPpDdNn":
                        phrase_symbols[ch] += 1
    except Exception: pass

    try: sec_ok = t.sections is not None and len(t.sections.intervals) > 0
    except Exception: pass

    has_tonic += tonic_ok; has_pitch += pitch_ok; has_pitch_vocal += pv_ok
    has_phrases += ph_ok; has_sections += sec_ok

    raga_names, artist_name = [], "UNKNOWN"
    try:
        md = t.metadata or {}
        raagas = md.get("raaga") or []
        raga_names = [r.get("name", "?") for r in raagas if isinstance(r, dict)]
        if len(raga_names) > 1: multi_raga += 1
        for r in raga_names: raga_counter[r] += 1
        arts = md.get("album_artists") or md.get("artists") or []
        if arts and isinstance(arts[0], dict):
            artist_name = arts[0].get("name", "UNKNOWN")
        artist_counter[artist_name] += 1
    except Exception: pass

    rows.append({"track_id": tid, "tonic": tonic_ok, "pitch": pitch_ok,
                 "pitch_vocal": pv_ok, "phrases": ph_ok,
                 "n_ragas": len(raga_names), "artist": artist_name})

print(f"\n=== COVERAGE (of {n} tracks) ===")
print(f"  tonic       : {has_tonic:3d} ({100*has_tonic/n:.0f}%)")
print(f"  pitch (mix) : {has_pitch:3d} ({100*has_pitch/n:.0f}%)")
print(f"  pitch_vocal : {has_pitch_vocal:3d} ({100*has_pitch_vocal/n:.0f}%)  <- sung line")
print(f"  phrases     : {has_phrases:3d} ({100*has_phrases/n:.0f}%)  <- phrase-rep viability")
print(f"  sections    : {has_sections:3d} ({100*has_sections/n:.0f}%)")

if frame_hops:
    hop = float(np.median(frame_hops))
    print(f"\n=== FRAME RATE (Section 3) ===")
    print(f"  median pitch hop = {hop*1000:.2f} ms")
    print(f"  -> 21-frame median filter ~= {21*hop*1000:.0f} ms")
    print(f"  -> 10-frame min dwell     ~= {10*hop*1000:.0f} ms")

print(f"\n=== RAGA / ARTIST distribution ===")
print(f"  distinct ragas   : {len(raga_counter)}")
print(f"  distinct artists : {len(artist_counter)}")
print(f"  multi-raga (ragamalika): {multi_raga}")
print(f"  ragas >=2 recordings (evaluable): {sum(1 for c in raga_counter.values() if c >= 2)}")
print(f"  ragas >=3 recordings: {sum(1 for c in raga_counter.values() if c >= 3)}")

if phrase_symbols:
    print(f"\n=== PHRASE solfege vocabulary ===")
    for sym in "SsRrGgMmPpDdNn":
        if phrase_symbols[sym]:
            print(f"  {sym}: {phrase_symbols[sym]}")
    print(f"  distinct phrase strings: {len(phrase_ngrams)}")
    print("  (case = functional swara identity, NOT pitch-folded)")

with open(os.path.join(OUT_DIR, "saraga_v15_census.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nSaved per-track census -> {OUT_DIR}/saraga_v15_census.csv")

print("\nDECISIONS THIS INFORMS:")
print("  1. pitch vs pitch_vocal -> pick fully-covered one (prefer vocal)")
print("  2. phrase coverage high -> build phrase-level representation")
print("  3. these counts replace 197/176/95 in the paper")
