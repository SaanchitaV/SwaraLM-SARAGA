"""
Phrase sub-study, Step 1 -- PROBE (patched for EventData: .events not .labels)
=============================================================
Reads the phrase annotations (EventData: .intervals for timing, .events for
solfege labels). Answers: raga distribution, raw label format, vocabulary size,
phrase length, per-track counts, and the KEY question -- do phrases recur
within a raga (same-raga Jaccard) more than across ragas?
"""

import os
import random
from collections import Counter, defaultdict

import numpy as np
import mirdata

DATA_HOME = "/content/saraga_carnatic"
ds = mirdata.initialize("saraga_carnatic", data_home=DATA_HOME, version="1.5")

def phrase_labels(t):
    """Return list of phrase label strings from an EventData object."""
    ph = t.phrases
    if ph is None:
        return []
    ev = getattr(ph, "events", None)
    if ev is None:
        return []
    return [str(e) for e in ev]

# ---- gather phrase-bearing tracks ----
phrase_tracks = []
for tid in ds.track_ids:
    t = ds.track(tid)
    try:
        labs = phrase_labels(t)
        if labs:
            raga = "?"
            md = t.metadata or {}
            raagas = md.get("raaga") or []
            if raagas and isinstance(raagas[0], dict):
                raga = raagas[0].get("name", "?")
            phrase_tracks.append((tid, labs, raga))
    except Exception:
        pass

print(f"Tracks with phrase annotations: {len(phrase_tracks)}")

# ---- (1) raga distribution (exclude ragamalika / unknown for evaluable) ----
raga_counter = Counter(r for _, _, r in phrase_tracks)
def evaluable_raga(r):
    return r not in ("Rāgamālika", "Ragamalika", "?", "")
ev_ragas = {r: c for r, c in raga_counter.items() if evaluable_raga(r)}
print(f"Distinct ragas: {len(raga_counter)}")
print(f"Evaluable ragas (>=2 tracks, excl ragamalika/unknown): "
      f"{sum(1 for r,c in ev_ragas.items() if c>=2)}")

# ---- (2) raw label examples ----
print("\n--- RAW phrase labels (first track) ---")
tid0, labs0, raga0 = phrase_tracks[0]
print(f"track: {tid0}  raga: {raga0}  ({len(labs0)} phrases)")
for lab in labs0[:15]:
    print(f"   {lab!r}")

# ---- (3) vocabulary, (4) length, (5) per-track counts ----
all_labels = Counter()
lengths, per_track = [], []
for tid, labs, raga in phrase_tracks:
    per_track.append(len(labs))
    for l in labs:
        all_labels[l] += 1
        lengths.append(len(l.replace(" ", "")))

print(f"\n--- VOCABULARY ---")
print(f"  total phrase instances : {sum(all_labels.values())}")
print(f"  distinct phrase strings: {len(all_labels)}")
print(f"  appearing >=2x : {sum(1 for c in all_labels.values() if c>=2)}")
print(f"  appearing >=5x : {sum(1 for c in all_labels.values() if c>=5)}")
print("  most common:")
for lab, c in all_labels.most_common(15):
    print(f"     {c:4d}x  {lab!r}")

if lengths:
    L = np.array(lengths)
    print(f"\n--- PHRASE LENGTH (chars) --- min {L.min()} median {int(np.median(L))} "
          f"mean {L.mean():.1f} max {L.max()}")
if per_track:
    P = np.array(per_track)
    print(f"--- PHRASES PER TRACK --- min {P.min()} median {int(np.median(P))} "
          f"mean {P.mean():.1f} max {P.max()}")

# ---- (6) recurrence: same-raga vs diff-raga phrase-set Jaccard ----
print(f"\n--- PHRASE RECURRENCE (the key gate) ---")
raga_to = defaultdict(list)
for tid, labs, raga in phrase_tracks:
    if evaluable_raga(raga):
        raga_to[raga].append((tid, set(labs)))

within = []
for rg, lst in raga_to.items():
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            a, b = lst[i][1], lst[j][1]
            u = len(a | b)
            if u: within.append(len(a & b)/u)

flat = [(rg, phr) for rg, lst in raga_to.items() for _, phr in lst]
random.seed(42)
across = []
for _ in range(1000):
    x, y = random.choice(flat), random.choice(flat)
    if x[0] != y[0]:
        u = len(x[1] | y[1])
        if u: across.append(len(x[1] & y[1])/u)

if within:
    print(f"  same-raga Jaccard: mean {np.mean(within):.3f}  (n={len(within)} pairs)")
if across:
    print(f"  diff-raga Jaccard: mean {np.mean(across):.3f}  (n={len(across)} pairs)")
if within and across:
    ratio = np.mean(within)/max(np.mean(across),1e-9)
    print(f"  ratio within/across: {ratio:.2f}")
    print("  >>1 : characteristic phrases recur within raga -> build the representation")
    print("  ~=1 : phrases too track-specific (or transcription variance) -> "
          "try approximate matching before concluding")
