# Swara-Level Retrieval over Carnatic Music — Project Log

*An information-retrieval project on allied-raga discrimination in the Saraga dataset.*

---

## 1. Motivation & Research Question

**Core question:** Can a retrieval system distinguish *allied ragas* — ragas that
share the same scale / janaka but differ in **sanchara** (characteristic melodic
movement) — where naive note-set matching fails?

**Why it matters (IR framing):** This is retrieval where surface tokens are
(near-)identical but relevance depends on sequential/structural context. The hard
part is a representation that is *invariant* to nuisance variation (performer,
ornament) while *discriminative* on the movement grammar that defines raga identity.

**Hypothesis:** A learned sanchara-based representation separates allied ragas
significantly better than a swara-presence (bag-of-notes) baseline, on held-out
artists.

---

## 2. Dataset

- **Name:** Saraga (Carnatic collection)
- **Source:** CompMusic / MTG-UPF (Zenodo; also on Kaggle)
- **Why this dataset:** largest open, annotated collection of Indian Art Music;
  includes tonic annotations, predominant-melody pitch tracks, and editorial
  raga/artist/concert metadata.
- **Access method:** `mirdata` loader (download + validation + uniform interface).

> _Fill after census:_
> - Total recordings: `___`
> - Distinct ragas: `___`  |  artists: `___`  |  concerts: `___`
> - Tonic-annotation coverage: `___%`  |  pitch coverage: `___%`

---

## 3. Libraries & Tools

| Library | Role |
|---|---|
| `mirdata` | Standardized loading/validation of Saraga; metadata interface |
| `compiam` | Carnatic-specific methods: tonic ID, melody/pitch extraction, baselines |
| `pandas` | Census aggregation, results tables |
| _(later)_ `librosa` / `essentia` | Pitch/chroma feature extraction |
| _(later)_ `pytorch` | Learned sanchara representation (contrastive / sequence model) |
| _(later)_ `faiss` / `hnswlib` | Approximate nearest-neighbor dense retrieval |

---

## 4. Method / Pipeline

Order matters — each stage de-risks the next.

1. **Census** *(collection statistics)* — raga / artist / annotation depth.
   Decides whether the learned path is trainable.
2. **Feature extraction** *(document representation)* — tonic-relative pitch →
   quantize to swara sequences. Each recording becomes a "document."
3. **Baseline** — swara-presence vector (bag-of-notes). *Deliberately fails on
   allied pairs — this is the strawman.*
4. **Transition representation** — first/second-order swara-transition vectors
   (Markov). Non-learned floor.
5. **Learned representation** *(the ML core)* — contrastive / sequence-model
   embedding trained to separate allied pairs, invariant to ornament & artist.
6. **Retrieval + evaluation** — ANN search; IR metrics on **held-out artists**.

**Anti-cheating measures:**
- Tonic-relative normalization (deletes tonic confound).
- Per-recording whitening (suppresses singer/mic).
- **Split by artist** (so learning the singer instead of the grammar shows up as failure).

---

## 5. Allied-Pair Evaluation Set

Grouped by *why* the pair is hard:

| Category | Meaning |
|---|---|
| `scale_identical` | same arohana/avarohana, different sanchara (headline case) |
| `same_janaka` | same melakarta parent, different janya character |
| `gamaka_only` | separated mainly by ornament/prayoga (hardest) |
| `aro_avaro_asym` | same notes, vakra/asymmetric vs linear (optional) |

> _Fill after Pass 2:_ per-pair depth, and which pairs are LEARNED-ELIGIBLE.

---

## 6. Evaluation Protocol

- **Split:** by recording *and* artist (held-out-artist test).
- **Metrics:** Precision@k, MAP, (nDCG if graded relevance defined), MRR.
- **Comparison:** learned representation vs. transition floor vs. swara-presence baseline.
- **Reported per category** (not just aggregate) — shows where the method's limits are.

---

## 7. Findings

> _Log as you go. Keep dated entries. Negative results are results._

### Census findings
- _date_ — ...

### Baseline results
- _date_ — ...

### Learned-representation results
- _date_ — ...

---

## 8. Running Log (chronological)

- **YYYY-MM-DD** — Set up repo, wrote census script.
- **YYYY-MM-DD** — Ran Pass 1. Raga inventory: ...
- ...

---

## 9. References

- Saraga dataset paper (Srinivasamurthy et al., ISMIR 2016 / 2020).
- Manning, Raghavan, Schütze — *Introduction to Information Retrieval*.
- Zhai — *Text Data Management and Analysis* (language-model IR).
- _(add papers as you read them)_
