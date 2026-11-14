# Stage 5 — Contrastive Fine-Tuning: Aligning the Embedding with Retrieval

## Where we are in the project

**Phase: Machine Learning / Representation Learning — the pivotal experiment.**

Everything before this stage either set a bar or diagnosed a failure:

| Stage | Result | What it established |
|---|---|---|
| 2 | bag-of-notes P@1 = 0.423, transitions = 0.431 | Content is a strong baseline; aggregate tie |
| 3 | On allied ragas: content 0.303, transitions 0.193 | Naive movement fails from per-recording estimation noise |
| 4a | LSTM-LM embedding P@1 = 0.179 | Generative objective learns *shared* grammar, not *discriminative* identity |
| 4b | Transformer-LM embedding P@1 = 0.154 | Same failure with a different architecture → **the objective is the problem, not the model** |

Stage 5 is the targeted response: keep the encoder, keep the data, keep the
label-free constraint — **change what the embedding is optimised for.**

Pipeline position: … → Swara LM (4a/4b) → **Contrastive Fine-Tuning (here)** →
Final Comparison & Evaluation.

---

## The diagnosis this stage answers

Next-swara prediction rewards modelling what is **common** across the corpus —
stepwise motion, frequent swaras, generic gamaka patterns. Both 4a and 4b
plateaued at perplexity ≈ 3.9 without overfitting: they *saturated*, meaning
there was no more extractable signal **under that objective**. Mean-pooling the
hidden states of a model trained this way averages out whatever weak
raga-distinctive signal exists, leaving an embedding that says "this is Carnatic
melody" rather than "this is Kalyani". Retrieval needs what is **distinctive
between classes**; the generative objective never asked for it.

This is the classic pretraining/downstream **objective misalignment**, exposed at
small scale where the model cannot afford to learn both.

---

## What Stage 5 builds

### The contrastive objective (still label-free)
- **Positive pair:** two different windows sampled from the **same recording**.
- **Negatives:** windows from **other recordings** in the same batch
  (in-batch negatives, standard InfoNCE loss with temperature 0.1).
- **No raga labels are used anywhere.** The supervision signal is recording
  identity, which is free.

### Why recording-discrimination should transfer to raga-discrimination
Two windows of the same recording share the same raga's movement grammar; windows
from different recordings usually do not. Optimising the embedding to pull
same-recording windows together and push different-recording windows apart
therefore forces it to encode **what makes recordings melodically distinctive** —
and the dominant melodic distinction between Carnatic recordings *is* the raga's
sanchara. The hypothesis: the embedding becomes raga-discriminative as a
side-effect of recording-discrimination, with zero labels.

### Warm-starting from Stage 4a
The encoder loads the trained 4a LSTM weights (with a fresh projection head), so
the generative pretraining is not discarded — it becomes the initialisation. This
is the classic **pretrain → fine-tune** pattern: 4a supplies general melodic
features; Stage 5 re-shapes the embedding space for discrimination.

### The known shortcut, and how it is controlled
The model could satisfy the contrastive objective by learning **recording/artist
style** rather than melodic content — a cheat that would score well without
learning raga. Two controls:

1. **The input is only swara symbols.** No audio, no timbre, no absolute pitch —
   the usual carriers of artist identity are absent by construction (Stage 1's
   tonic normalisation and symbolic representation did this work already).
2. **Held-out-artist evaluation.** If the embedding learned style, it will fail
   to transfer to unseen artists' recordings, and the evaluation will show it.
   Style-learning is not silently rewarded; it is visibly punished.

---

## Evaluation

Identical protocol to every previous stage (same queries, cosine similarity,
P@k / MAP / MRR), so the final table is a clean five-way comparison:

bag-of-notes · transition-matrix · LSTM-LM · Transformer-LM · **contrastive**.

Secondary analysis (as in Stage 3): the **allied-raga / high-overlap stratum**,
where content topped out at 0.303. Movement recovered *there* is the project's
central claim vindicated, even if the aggregate is close.

---

## Stakes — both outcomes are the paper

- **Contrastive beats 0.423:** a label-free learned representation finally
  exceeds content — and the arc explains *why* every prior attempt failed
  (estimation noise → objective misalignment → alignment fix). The thesis lands.
- **Contrastive does not beat 0.423:** the finding becomes "at this data scale, a
  12-dimensional content histogram beats every learned representation tried,
  including a properly aligned contrastive one" — a counterintuitive,
  well-evidenced negative result about low-resource symbolic MIR, with the full
  diagnostic chain documenting *why*. Honest, useful, publishable.

Either way, the project ends with a complete, internally consistent story in
which every experiment either moved the number or explained why it could not.

---

## Honest limitations

- **Single split.** All results so far use one held-out-artist split (seed 42);
  learned-model numbers carry split variance that a single run cannot estimate.
  The final reported comparison should be repeated over ~5 artist splits and
  reported as mean ± std (the baselines are deterministic and do not vary).
- **In-batch negatives at batch 32** give a modest negative pool; larger batches
  or a memory bank are standard upgrades if the effect looks real but weak.
- **Window-level positives** assume any two windows of a recording share
  identity; long recordings with internal sections (e.g. RTP) weaken this
  slightly.

---

## Outputs

- `stage5_contrastive_results.csv` — the pivotal retrieval numbers.
- `stage5_contrastive_embeddings.npy` — embeddings for the final comparison and
  allied-raga stratified analysis.
- `stage5_contrastive.pt` — fine-tuned model.

---

## One-line summary

Stage 5 keeps everything fixed except the one thing diagnosed as broken — the
objective — replacing generative prediction with label-free contrastive
discrimination between recordings, and thereby runs the experiment the entire
project has been building toward: can a properly aligned learned representation
finally beat a 12-dimensional histogram?
