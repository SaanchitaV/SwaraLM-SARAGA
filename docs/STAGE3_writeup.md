# Stage 3 — Confusion Analysis: Where Does Movement Matter?

## Where we are in the project

**Phase: Information Retrieval — diagnostic / error analysis.**

Stage 2 evaluated the baselines in aggregate and found a tie. Stage 3 does not add
a new representation; it **interrogates** the Stage 2 result to find out *where* the
representations succeed and fail — specifically, whether the movement signal helps
on the cases that theoretically require it (allied ragas). This is error analysis,
the step that turns a flat metric into an understanding of the model's behaviour.

Pipeline position: Census → Feature Extraction → Baselines → **Confusion Analysis
(here)** → Language Model → Dense Retrieval → Evaluation.

---

## What Stage 3 does

The aggregate P@1 in Stage 2 averages over all 95 ragas. Most ragas have distinct
note-sets, so note-content alone separates them and movement contributes nothing
*there*. Any benefit of movement should appear only on **allied ragas** — ragas
that share a scale but differ in sanchara. Averaged over everything, that effect is
diluted. Stage 3 removes the dilution three ways:

1. **Confusion pairs (A).** For each representation, record which raga is wrongly
   retrieved as the top-1, and how often. This surfaces *which* ragas are actually
   being confused, and each pair is annotated with its scale-overlap (Jaccard of
   note-sets).

2. **Scale-overlap stratification (B).** Split query ragas by whether they live in
   a **high-overlap neighbourhood** (share ≥80% of notes with some other raga —
   i.e. allied) vs. low-overlap. Compare bag-of-notes and transition-matrix P@1
   *within each stratum*. This is the direct test: does movement help where notes
   overlap?

3. **Interpretation against theory.** Cross-check whether the confused pairs are
   the ones music theory predicts to be hard.

---

## Results

### (A) The confusions are musically correct

The most-confused pairs are genuine allied ragas — ragas that share scales and that
even trained listeners separate by movement, not notes:

| Confused pair | Scale overlap |
|---|---|
| harikambhoji ↔ kambhoji | 1.00 |
| karaharapriya ↔ ritigaula | 1.00 |
| kamas ↔ kambhoji | 1.00 |
| kalgada ↔ saveri | 1.00 |
| madhyamavati ↔ manirangu | 0.83 |
| behag ↔ hamir kalyani | 0.88 |

This is itself a finding: **content-based swara representations confuse precisely
the allied-raga pairs that theory predicts.** The system fails on the *right*
cases — evidence the representations behave musically, not randomly.

### (B) Movement does NOT help on allied ragas — it hurts

| Representation | P@1 (high-overlap / allied, n=119) | P@1 (low-overlap, n=4) |
|---|---|---|
| bag_of_notes | **0.303** | 0.75 |
| transition_matrix | 0.193 | 1.00 |

On the allied ragas — the cases that were supposed to favour movement —
**note-content (0.303) beats first-order transitions (0.193).** This is the
opposite of the hypothesis, and it is reported as-is.

*(The low-overlap column has only 4 queries and is not interpretable; it is
excluded from any conclusion.)*

---

## What this means (the corrected finding)

The hypothesis was "movement separates allied ragas." The data says: **naive,
first-order movement modelling is too noisy to help, and on allied ragas that noise
costs more than the faint movement signal gains.**

The mechanism is estimation, not music:
- A 12×12 transition matrix has **144 parameters** estimated from a *single*
  recording's note-runs — sparse and noisy, especially after run-length collapsing.
- Bag-of-notes has **12 parameters** — far more stably estimated.
- On allied ragas the true distinction is subtle, so the transition matrix's
  estimation noise swamps the signal, and it underperforms the simpler
  representation.

So the movement information exists in principle, but **first-order Markov is both
too crude (only adjacent pairs) and too data-hungry (144 params/recording) to
recover it reliably.**

---

## Why this strengthens the project

This is a better motivation for the learned model than the result we expected:

- It shows note-content is a **surprisingly strong** baseline — a non-obvious,
  reportable result.
- It shows naive movement modelling **actively fails** on the hard cases, and
  explains *why* (estimation noise, first-order limitation).
- It therefore **precisely specifies what the learned model must do**: pool
  statistical strength across recordings (not 144 independent params each),
  capture patterns **longer than pairs**, and encode **direction**. A sequence
  model addresses exactly these three failure modes.

The Stage 4 mandate is now concrete and evidence-backed: *recover the movement
signal that first-order statistics cannot, on the allied ragas where content-based
methods top out at ~0.30 P@1.*

---

## Honest limitations

- **Low-overlap stratum is tiny (n=4)** — ignored in all conclusions.
- **Single-recording estimation** is the transition matrix's core weakness; a
  fairer movement baseline might pool transitions per raga, but that would leak the
  label — so we leave it as an honest per-recording baseline.
- **Small collection** overall; trends are more trustworthy than absolute values.

---

## One-line summary

The baselines confuse exactly the theoretically-allied ragas, and on those ragas
**content beats naive movement** because first-order transitions are too noisy to
estimate per recording — which is the precise, evidence-backed reason a *learned*
sequence model (Stage 4) is needed.
