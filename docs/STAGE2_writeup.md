# Stage 2 — Baselines: Bag-of-Notes & Swara-Transition Retrieval

## Where we are in the project

**Phase: Information Retrieval — baseline representations & first evaluation.**

This is the first stage that is genuinely **IR**. Stage 1 (feature extraction)
turned recordings into swara sequences; Stage 2 turns those sequences into
**vector representations**, defines a **similarity function**, runs **retrieval**,
and measures it with **IR metrics**. That quartet — represent, score, retrieve,
evaluate — is the core loop of information retrieval.

Pipeline position: Census → Feature Extraction → **Baselines (here)** → Language
Model → Dense Retrieval → Evaluation.

Why baselines come before the learned model: the project's central claim is that
**movement (sanchara) matters beyond the note-set**. To show a learned model
captures movement, we first need simple reference points that (a) capture only
note-content and (b) capture only first-order movement. These are the strawman and
the floor the learned model must beat. Without them, a good learned result is
unfalsifiable — you cannot claim improvement without a baseline to improve over.

---

## What Stage 2 builds

### Baseline 1 — Bag-of-Notes (the strawman)
- **Representation:** a 12-dimensional vector — the proportion of each swara
  position in a recording.
- **What it captures:** *what notes* are present, and how often.
- **What it ignores:** order, movement, gamaka — everything sequential.
- **Expected weakness:** it cannot distinguish ragas that share a scale but differ
  in movement (allied ragas). This failure is the point — it motivates everything
  after it. It is the "bag of words" of this project.

### Baseline 2 — Swara-Transition Matrix (the floor)
- **Representation:** a 12×12 first-order Markov matrix, row-normalized, flattened
  to 144 dimensions — P(next swara | current swara).
- **What it captures:** *which swara follows which* — the first computational
  shadow of sanchara.
- **Why it should beat Baseline 1:** movement carries raga identity that raw
  note-content does not. If it wins, the project's core premise is demonstrated at
  the simplest possible level, before any deep learning.
- **What it still misses:** long-range phrase structure, gamaka shape, directional
  (aroha/avaroha) asymmetry beyond immediate transitions. These are what the
  learned model (Stage 4) exists to capture.

---

## Evaluation

**Task:** same-raga retrieval (query-by-example). Each recording is a query; all
others are ranked by cosine similarity of their vectors. A retrieved recording is
**relevant** if it shares the query's raga.

**Why query-by-example:** it is the honest IR framing of "does this representation
place musically-similar recordings near each other," and it needs no training — so
it fairly evaluates the *representation itself*, not a classifier fit on top.

**Metrics:**
- **Precision@k** (k = 1, 3, 5): fraction of the top-k that are same-raga.
- **MAP** (mean average precision): rewards ranking all relevant items high.
- **MRR** (mean reciprocal rank): how high the *first* correct hit appears.

**Query eligibility:** only recordings whose raga has ≥2 recordings are used as
queries — otherwise no correct answer exists to retrieve. This automatically
excludes the singleton-raga tail and keeps the metric meaningful.

**Similarity:** cosine over L2-normalized vectors — standard for this kind of
sparse count/probability representation, and scale-invariant.

---

## What the results will tell us

1. **Absolute level.** With ~20 evaluable ragas, random P@1 is low; scores well
   above chance mean the representations carry real raga signal.
2. **The key comparison — does Baseline 2 beat Baseline 1?**
   - *If yes:* movement helps over note-content alone — core premise supported at
     baseline level. This is a genuine, reportable finding on its own.
   - *If roughly equal:* first-order transitions are insufficient; the learned
     model has clear, well-motivated work to do. Also informative.
3. **The benchmark.** Whatever these numbers are, they are the bar Stage 4's
   learned representation must clear. Improvement is defined relative to them.

---

## Honest limitations (documented, not hidden)

- **First-order only.** The transition matrix sees pairs, not phrases. Ragas
  distinguished by longer characteristic movement will still be confused.
- **Direction-blind aggregate.** A transition matrix mixes ascending and
  descending contexts; it does not separately model aroha vs avaroha the way the
  theory (and Stage 1's directional diagnostic) does.
- **No gamaka shape.** The per-note gamaka descriptors from Stage 1 are not yet
  used here; these baselines operate on the collapsed symbolic sequence only.
- **Small data.** Absolute metric values are on a modest collection; trends and
  relative comparisons are more trustworthy than any single number.

These limitations are not flaws to apologize for — each one names a specific thing
the learned model is designed to address, so Stage 2 also functions as the
motivation for Stage 4.

---

## Output

- `stage2_baseline_results.csv` — the metrics table for both representations.
- Serves as the benchmark row(s) that all later results are compared against.

---

## Next

**Stage 3 (optional refinement) / Stage 4 (the learned model).** Depending on the
baseline gap: if first-order transitions already help meaningfully, proceed to the
self-supervised swara language model, whose learned representation should capture
the longer-range and directional structure these baselines cannot. The baseline
table defines exactly what "better" has to mean.
