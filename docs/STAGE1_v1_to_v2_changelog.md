# Stage 1 → Stage 1 v2: Change Log & Rationale

## Where we are in the project

**Phase: Feature Extraction / Data Preprocessing** — the foundation layer.

To be precise about the taxonomy, since these terms get used loosely:

- This is **not yet** the IR, ML, or NLP part. Those come later and all depend on
  this output.
- What we are doing is **converting raw pitch data into a symbolic representation**
  (swara sequences) that the downstream stages consume. In the pipeline this is
  Stage 1 of 6 (Census → **Feature Extraction** → Baselines → Language Model →
  Retrieval → Evaluation).
- Strictly, it spans two sub-activities: **preprocessing** (cleaning the f0
  contour) and **feature extraction** (deriving swara sequences + gamaka
  descriptors from it). They live together here because the cleaning choices and
  the features are tightly coupled.

Why this phase matters disproportionately: every later stage — the self-supervised
language model (ML/NLP), the dense retrieval (IR), the evaluation — operates on the
sequences produced here. A representation error introduced now propagates into
every result downstream. This is why we verify by ear before proceeding.

---

## Summary of the change

Stage 1 (v1) produced swara sequences by hard-binning each pitch frame to the
nearest of 12 swara positions, with light smoothing. Inspection revealed spurious
swaras — notes appearing in a raga's sequence that do not belong to it (e.g. M2
and G2 surfacing in Vasanta, which has neither). Investigation showed these were
not the raga's notes but **gamaka excursions** — the pitch oscillating through
neighbouring positions during ornamentation — being mislabelled as distinct notes.

Stage 1 v2 addresses this **without using the raga label to modify the data**
(which would leak the target and erase the gamaka signal). It does so via
stronger *raga-blind* cleaning, explicit *gamaka profiling*, and *diagnostic-only*
comparison against raga theory.

---

## Change 1 — Stronger raga-blind smoothing

**What changed:** `MEDIAN_FILTER_WIN` 5 → 21 frames; introduced `MIN_DWELL_FRAMES`
(10) replacing the looser `MIN_RUN_FRAMES` (3).

**What it does:** widens the median filter applied to the f0 contour *before*
quantization, and requires a pitch region to be held for ≥10 frames before it
counts as a note. Together these collapse short gamaka oscillations onto the
central (dwelt-upon) swara instead of emitting a jitter of neighbouring symbols.

**Why it is needed:** the v1 spurious-note problem was transient excursions
surviving as notes. A wider smoothing window and a minimum dwell requirement
remove that transient jitter using only the **signal's own properties** (how
stable and how sustained a pitch is) — never the raga label.

**Why raga-blind specifically:** if we used each raga's known notes to decide what
to keep, we would (a) leak the answer into a model whose job is to *learn* raga
structure, making any evaluation circular, and (b) delete the gamaka, which is the
very information that distinguishes ragas sharing the same notes. Cleaning must
therefore depend only on the signal, not on the label.

---

## Change 2 — Gamaka profiling (new)

**What changed:** each note in the collapsed sequence now carries
`gamaka_cents_std` (spread of pitch within the note, in cents) and `drift_cents`
(net pitch movement across the note).

**What it does:** instead of discarding the oscillation around a swara, it
**measures and records it** as a per-note descriptor.

**Why it is needed:** gamaka is not noise — it is the highest-information content
in Carnatic melody, and often the *only* thing separating two ragas with identical
note sets. Deleting it to obtain "clean notes" would remove the signal the project
depends on. Capturing it as a feature means a downstream model can *use* the
oscillation profile to tell ragas apart, rather than being handed a flattened
scale. This converts what v1 treated as a defect (wobble) into an asset (a
discriminative feature).

---

## Change 3 — Dominance histogram (new)

**What changed:** each recording now emits `swara_hist_pct` — the percentage of
voiced frames spent on each of the 12 swara positions.

**What it does:** gives a quantitative view of which swaras dominate a recording,
replacing eyeball inspection of the raw sequence.

**Why it is needed:** to judge whether a spurious swara is a real problem or benign
gamaka transit, we need magnitude, not presence. "M2 appears" is uninformative;
"M2 is 3% of frames" (benign) vs. "M2 is 30%" (a genuine problem) is actionable.
The histogram makes the verification objective and fast.

---

## Change 4 — `analyze_against_theory` (new, diagnostic only)

**What it does:** given a raga's theoretical notes, it measures each recording's
**on-scale %** (energy on notes belonging to the raga) vs. **excursion %** (energy
off them entirely), and flags recordings with unusually high excursion.

**What the metric means:** high excursion % is *not* automatically an error — heavy
gamaka legitimately lives off-scale. The metric is used to (a) spot **pitch-tracker
failures** (e.g. octave errors producing impossible notes) and (b) **compare**
gamaka intensity across recordings/ragas. It is a health check and a comparison
tool, never a cleaning step.

**Why it is needed:** it lets us distinguish two causes of off-note energy that
look identical in the raw sequence — genuine ornamentation vs. extraction error —
by cross-referencing against what the raga *should* contain. This is the one
legitimate use of raga theory in this phase: to **analyse the output**, never to
**modify the input**.

**Critical boundary:** this function reads the sequences and reports numbers. It
does not, and must not, alter them. Using theory to measure is sound; using theory
to rewrite the data would reintroduce the label-leak and gamaka-erasure problems
Change 1 was designed to avoid.

---

## Change 5 — Directional theory + `analyze_directional` (new, diagnostic only)

**What changed:** `RAGA_THEORY` moved from a flat swara **set** to a **directional**
structure — separate ordered `arohana` (ascent) and `avarohana` (descent) — and a
new `analyze_directional` function consumes it.

**Why the set was wrong:** a swara can be legal in one direction and illegal in the
other. Bilahari ascends `S R2 G3 P D2 S` (no Ma, no Ni) but descends
`S N3 D2 P M1 G3 R2 S` (Ma and Ni present). A flat set {S R2 G3 M1 P D2 N3} asserts
Ma and Ni are freely available, which is false — they are descent-only. The set
representation erases this asymmetry, and the asymmetry is itself part of the
raga's grammar.

**What the metric does:** it walks the *ordered* sequence and, for each step,
checks directional legality — an ascending move should land on a swara legal in
arohana, a descending move on one legal in avarohana — reporting the fraction of
direction-consistent notes.

**Why it is needed:** it catches a class of error the histogram cannot — a note
sung in the *wrong direction* — because the histogram is directionless. This is a
concrete, measurable instance of the project's central thesis: raga identity lies
in movement (here, direction), not merely in the note set.

**Known limitation (documented, not a bug):** simple aro/avaro does not model
**vakra** (zigzag) ragas, whose legal ascent is non-monotonic. For such ragas the
metric will report artificially low consistency. Low directional consistency
therefore means *either* a tracking error *or* a raga whose grammar exceeds the
aro/avaro model — domain judgement decides which. This limitation is itself
informative: it demonstrates the ladder the project climbs — note-set ⊂ aro/avaro
⊂ full sanchara — where each rung captures grammar the previous one misses.

---

## Net effect

| Aspect | v1 | v2 |
|---|---|---|
| Smoothing | light (win=5) | stronger, raga-blind (win=21) |
| Note threshold | 3 frames | 10 frames (dwell) |
| Gamaka | discarded as jitter | measured & retained as a feature |
| Dominance view | none | per-recording swara histogram |
| Theory use | none | diagnostic only (on-scale %, directional %) |
| Theory shape | — | directional (aro/avaro), not a flat set |
| Data modified by theory? | — | **never** |

The guiding principle across all changes: **clean using the signal, analyse using
the theory, and never let the theory rewrite the data.** This keeps the
self-supervised learning honest (no label leak) and preserves gamaka (the
discriminative signal) while still removing genuine extraction noise.

---

## Verification gate before proceeding

Stage 1 v2 is not "done" until the output is validated by ear against musical
knowledge:

1. Do the **dominant swaras** of each recording match the raga's actual notes,
   with off-notes reduced to a clear minority?
2. Does **excursion %** track how gamaka-heavy each raga is known to be?
3. Does **directional consistency %** align with expectation — high for
   straightforward ragas, low for known vakra ragas?

If these three align with musical judgement, the representation is trustworthy and
we advance to Stage 2 (baselines: bag-of-notes and swara-transition vectors). If
they do not, we tune the raga-blind knobs (smoothing width, dwell threshold) —
never the theory-based cleaning, which we do not do.
