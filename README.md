# SwaraLM

**Label-free evaluation of melodic representations for Carnatic raga retrieval.**

Given a recording, find others in the same raga using melodic content alone, with
no raga labels used at any point in training. Seven representations are compared
under one protocol on the Saraga Carnatic collection.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24229237.svg)](https://doi.org/10.5281/zenodo.24229237)

Paper: [Counting Still Wins: A Label-Free Evaluation of Melodic Representations for Carnatic Raga Retrieval](https://doi.org/10.5281/zenodo.24229237) (Zenodo preprint). A copy is in `docs/SwaraLM_paper.pdf`, with its source in `docs/paper/`.

---

## Headline result

At this data scale, count-based representations outperform every learned
representation tried, in aggregate and within the allied-raga stratum.

| Representation | P@1 | P@3 | P@5 | MAP | MRR |
|---|---|---|---|---|---|
| Swara-transition matrix | **0.431** | 0.320 | 0.237 | 0.418 | **0.547** |
| Pitch-class histogram | 0.423 | 0.290 | 0.233 | **0.423** | 0.519 |
| LSTM language model | 0.369 ± 0.016 | 0.282 | 0.229 | 0.405 | 0.501 |
| Contrastive (InfoNCE) | 0.366 ± 0.025 | 0.298 | 0.237 | 0.415 | 0.524 |
| Late interaction, warm-start | 0.317 | 0.263 | 0.218 | 0.352 | 0.441 |
| Late interaction, trained | 0.301 | 0.236 | 0.189 | 0.321 | 0.417 |
| Transformer language model | 0.276 | 0.201 | 0.164 | 0.276 | 0.384 |
| BM25 | 0.024 | 0.022 | 0.023 | 0.046 | 0.074 |

176 labelled recordings, 95 ragas, 123 evaluable queries. For the LSTM and
contrastive rows, P@1 is the mean ± sd over five held-out-artist splits; their
other columns come from the development runs whose embeddings are in `models/`
(P@1 0.398 and 0.423). Transformer and late-interaction rows are single runs.

Five supporting findings:

- **Single-run parity is not reproduced.** Contrastive fine-tuning matched the
  histogram in one development run (0.423). Across five artist splits it
  averaged 0.366, and no seed reached the development figure. The harness used
  a more conservative contrastive configuration than the development run, and
  nominally identical LSTM runs differ by up to 0.04, so single runs at this
  scale are not to be trusted.
- **Pooling was not the bottleneck.** Late interaction removes mean-pooling
  entirely, trained cleanly (val loss 17.1 → 1.9), and retrieved *worse* than
  the pooled model it was built from.
- **BM25 collapses structurally.** All twelve pitch positions occur in all 176
  recordings, so document frequency is uniform, IDF weights nothing, every
  query is the same twelve terms, and the ranking does not depend on the query.
- **Order-sensitive representations are fragile to extraction noise.** Raga-blind
  denoising (v1 → v2) improved the histogram by a third (0.317 → 0.423) and
  roughly doubled the transition matrix (0.220 → 0.431), the LSTM (0.179 →
  0.369) and the Transformer (0.154 → 0.276). The v1 baseline aggregates are
  reconstructed from the stratified v1 run.
- **Perplexity moves against retrieval.** The same denoising *raised* LM
  perplexity, because the removed gamaka jitter was the most predictable content.

## Phrase sub-study

On the 94 phrase-annotated recordings (22 ragas, 60 queries), term weighting
over expert phrase transcriptions beats svara counting on the same data:

| Representation | P@1 (95% CI) | MAP | MRR |
|---|---|---|---|
| Svara counts | 0.150 [0.07, 0.25] | 0.212 | 0.276 |
| Phrase VSM, exact | 0.333 [0.22, 0.45] | 0.291 | 0.391 |
| Phrase VSM, approx (τ=0.85) | **0.450** [0.33, 0.58] | **0.406** | **0.519** |

IDF is meaningful over phrases because phrases are rare and raga-specific
(within-raga phrase-set Jaccard 0.018 vs 0.000 across ragas), the exact mirror
of why it failed over svaras. **The unit of representation matters more than the
sophistication of the model.**

---

## Layout

```
src/      pipeline, stage 0 through 9, plus the phrase sub-study
docs/     per-stage writeups and the paper
results/  all result CSVs and run logs
data/     raga theory table (95 ragas, directional aro/avaro, confidence flags)
models/   trained checkpoints and embeddings
demo/     static retrieval explorer (GitHub Pages)
notebooks/ original exploratory notebooks and the first project plan
```

## Reproducing

```bash
pip install numpy pandas torch scikit-learn hnswlib mirdata rank_bm25 openpyxl
```

1. `src/stage0_census_kaggle.py` — corpus census
2. `src/stage1_extract_sequences.py` — f0 → tonic-normalised swara sequences
3. `src/stage2_baselines.py` — histogram and transition matrix
4. `src/stage3_confusion_strata.py` — confusions and allied stratification
5. `src/stage4a_lstm_lm.py`, `src/stage5_contrastive.py` — learned models
6. `src/bm25_baseline.py`, `src/ndcg_evaluation.py`, `src/stage9_colbert_colab.py`
7. `src/phrase_probe.py` → `phrase_vsm.py` → `phrase_vsm_approx.py` → `phrase_vsm_robust.py`

Seeds: 42, 7, 123, 2024, 31415. Every experiment reads one frozen
`swara_sequences.jsonl`; stratified metrics aggregate exactly to overall metrics.

> **Note:** the frozen `data/swara_sequences_v2_FROZEN.jsonl` is not yet in this
> repository. Add it before relying on the reproduction steps.

## Method note

All sequence cleaning is **raga-blind**: median smoothing and a minimum dwell,
both functions of the signal alone. Correcting extracted notes against a raga's
theoretical scale would inject the label into the input of models meant to
recover it, and would erase the gamaka that distinguishes allied ragas. The
raga theory table is used only for evaluation stratification and graded
relevance, never to modify a sequence.

## Data

Saraga Carnatic v1.5 (Bozkurt, Srinivasamurthy, Gulati, Serra), Zenodo
DOI [10.5281/zenodo.4301737](https://doi.org/10.5281/zenodo.4301737), accessed
via `mirdata` with checksum verification.

## Licences

- Code: MIT (`LICENSE`)
- Derived data and the raga theory table: CC BY-NC-SA 4.0 (`LICENSE-DATA`),
  inherited from Saraga Carnatic's share-alike terms.

## Citation

If you use this code, the dataset or the results, please cite the paper:

> Iyer, S. V. (2026). *Counting Still Wins: A Label-Free Evaluation of Melodic
> Representations for Carnatic Raga Retrieval*. Zenodo.
> https://doi.org/10.5281/zenodo.24229237

```bibtex
@misc{iyer2026counting,
  author    = {Iyer, Saanchita V.},
  title     = {Counting Still Wins: A Label-Free Evaluation of Melodic
               Representations for Carnatic Raga Retrieval},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24229237},
  url       = {https://doi.org/10.5281/zenodo.24229237}
}
```
