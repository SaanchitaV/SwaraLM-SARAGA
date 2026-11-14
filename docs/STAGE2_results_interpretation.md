# Stage 2 Results — Interpretation & Implications

## The numbers

| Representation | Eval queries | P@1 | P@3 | P@5 | MAP | MRR |
|---|---|---|---|---|---|---|
| Bag-of-notes (strawman) | 123 | 0.423 | 0.290 | 0.233 | **0.423** | 0.519 |
| Transition matrix (floor) | 123 | **0.431** | **0.320** | **0.237** | 0.418 | **0.547** |

176 labelled recordings across 95 ragas; 123 queries evaluable (raga has ≥2 recordings).

---

## What the results mean

### 1. Both representations carry real raga signal
With 95 ragas in the collection, chance-level P@1 is on the order of 1–5%.
Both representations achieve **~42% P@1** — the single nearest neighbour is the
same raga more than 40% of the time. So the swara sequences produced in Stage 1
are musically meaningful: raga identity survives the pitch→swara extraction. This
validates the feature-extraction pipeline from a retrieval standpoint, not just by
eye.

### 2. The two baselines are effectively tied
Transition matrix edges bag-of-notes on P@1, P@3, P@5, and MRR, but loses on MAP,
and **every margin is 1–3 points** — within noise for 123 queries. The honest
reading is **not** "movement beats note-content." It is: **first-order movement and
raw note-content perform about the same at the aggregate level.**

We deliberately do not overclaim the transition win. The gaps are too small to
support the thesis on their own.

### 3. Why the tie is expected — and why it *motivates* the project
A 12×12 transition matrix aggregated over a whole recording is still a coarse view
of movement. It discards exactly the things that carry raga identity:
- **Direction** — it mixes ascending and descending contexts into one matrix, so
  aroha/avaroha asymmetry (Bilahari-type ragas) is invisible.
- **Phrase length** — sancara are multi-note gestures; first-order transitions see
  only adjacent pairs.
- **Gamaka shape** — the oscillation profiles captured in Stage 1 are not used
  here.

Moreover, most of the 95 ragas have **distinct note-sets**, so bag-of-notes already
separates them — and where note-content alone suffices, movement adds nothing. The
place movement *should* matter is the minority of **allied ragas** (same scale,
different movement), and their effect is **diluted** in an average taken over all
ragas.

**Conclusion:** the aggregate tie is evidence for a precise claim — *shallow,
direction-blind, aggregated movement is not enough*. That is the strongest possible
motivation for a sequence model (Stage 4) that captures direction, longer phrases,
and context.

---

## How this affects the study

1. **The evaluation must be sharpened.** The aggregate metric hides the effect that
   matters. The next step (Stage 3) restricts evaluation to **allied / same-scale
   confusions**, where movement should separate what note-content cannot. If the
   transition matrix beats bag-of-notes *there*, the premise is supported on the
   cases that actually test it — a far stronger result than the aggregate.

2. **The benchmark is set.** Stage 4's learned model must (a) exceed ~0.42 P@1
   overall, and — more importantly — (b) separate allied-raga clusters that the
   baselines conflate. "Better" is now concretely defined.

3. **A negative-to-precise reframing.** "The baselines tied" is not a dead end; it
   converts into "here are the exact confusions shallow methods cannot resolve,"
   which is a sharper and more publishable framing than a raw accuracy number.

4. **Reporting integrity.** Carrying the honest tie (rather than spinning the
   1-point transition win) is itself a strength — it demonstrates evaluation rigor,
   which is exactly what an MIR-evaluation audience (e.g. ISMIR reviewers) rewards.

---

## One-line summary

The baselines prove the sequences carry raga signal but show that **crude movement
≈ note-content on average** — so the real question is whether movement helps **on
allied ragas specifically**, which is what Stage 3 isolates and Stage 4 is built to
exploit.
