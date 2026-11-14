# Stage 4a — Swara Language Model: LSTM

## Where we are in the project

**Phase: Machine Learning / NLP — the core contribution begins here.**

Stages 1–3 were feature extraction, baseline IR, and error analysis. Stage 4 is
where the project becomes a *learning* project: a neural sequence model is trained
on the swara sequences, and its learned representations are used for retrieval.
This is the "A" component of the project design (the swara language model), and the
first stage involving model training of any kind.

Stage 4 is split into two parts sharing one pipeline:
- **4a (this stage):** LSTM encoder — the robust, small-data-friendly architecture.
- **4b (next):** tiny Transformer — same objective, same data, same evaluation;
  only the encoder swaps. The 4a-vs-4b comparison ("at what data scale does a
  Transformer beat an LSTM on symbolic music?") is itself a reportable result.

Pipeline position: Census → Feature Extraction → Baselines → Confusion Analysis →
**Swara LM: LSTM (here)** → Swara LM: Transformer → Comparison & Final Evaluation.

---

## What Stage 4a builds

### The idea in one paragraph
A language model learns the grammar of text by predicting the next word; nothing is
labelled, yet grammar emerges because predicting well *requires* it. Stage 4a does
the same on melody: an LSTM is trained to predict the **next swara** given the
preceding swaras. To predict well in Carnatic music, the model must internalise
sanchara — which movements are permitted, expected, characteristic — because that
is what governs what comes next. **No raga labels are used at any point in
training.** The model's hidden states then serve as recording embeddings for
retrieval, evaluated identically to the baselines.

### Why this addresses the Stage 3 failure specifically
Stage 3 showed first-order transitions *lose* to note-content on allied ragas, and
diagnosed why: 144 parameters estimated per single recording is too noisy, and
adjacent-pair statistics are too crude. The LSTM addresses all three failure modes:

| First-order failure | LSTM answer |
|---|---|
| Per-recording estimation (noisy) | **One model pooled over all recordings** — statistical strength is shared |
| Only adjacent pairs | **Recurrent state carries longer context** — multi-note phrases |
| Direction-blind | **Sequential processing is inherently directional** — ascent and descent are different contexts |

### Design choices (driven by the 176-recording constraint)
- **Deliberately small:** 1-layer LSTM, 64 hidden units, 32-d embeddings,
  dropout 0.3. A large model would memorise this collection instantly.
- **Window training:** sequences are sliced into overlapping 64-token windows,
  turning 176 recordings into thousands of training examples — the standard trick
  that makes neural training feasible on small corpora.
- **Held-out-ARTIST validation:** whole artists are excluded from training and
  used for validation. Low validation loss therefore means the model learned
  *swara grammar that generalises across performers*, not memorised recordings or
  singer idiosyncrasies. This is the anti-cheating measure carried through from
  the project's design principles.
- **Early stopping** on that held-out-artist loss, so training halts when
  generalisation stops improving.

### From language model to retrieval
After training, each recording is embedded by running its swara sequence through
the LSTM and **mean-pooling the hidden states**. Retrieval is cosine similarity
over these embeddings, evaluated with the *identical* protocol to Stages 2/3
(same queries, same metrics: P@k, MAP, MRR), so all numbers are directly
comparable to the baselines.

---

## What the results will mean

The benchmark: bag-of-notes P@1 = 0.423, transition matrix P@1 = 0.431 (aggregate);
on allied ragas specifically, content = 0.303, naive movement = 0.193.

Three outcomes, all reportable:

1. **LSTM beats the baselines** — the learned representation recovered movement
   signal that per-recording first-order statistics could not. The project's
   central premise is supported, with the mechanism identified (pooled estimation,
   longer context, directionality).

2. **LSTM ties the baselines** — at this data scale, learned sequence
   representations do not add over note-content. An honest scale finding, and the
   4b comparison plus the allied-raga stratum become the deciding evidence.

3. **LSTM loses** — 176 recordings is below the threshold where sequence LMs help;
   a clean negative result about data requirements for symbolic-music
   representation learning, directly relevant to low-resource MIR.

**The decisive secondary analysis:** even if the aggregate ties, the question that
matters is the **allied-raga (high-overlap) stratum**, where content topped out at
0.303 and naive movement failed at 0.193. If the LSTM lifts that number, it
recovered exactly the signal the project set out to capture — that is the headline
result regardless of the aggregate.

**Training diagnostics to record:** the train/validation loss curves. Falling
together = learning grammar; diverging = overfitting (early stopping catches it,
but the epoch at which it occurs is itself data about how much signal the corpus
supports). Validation perplexity is reported as the language-modelling quality
measure, independent of retrieval.

---

## Honest limitations

- **Mean-pooling** hidden states is the simplest possible sequence-to-embedding
  reduction; it may wash out phrase-level detail. (Refinements — attention pooling,
  max-pooling, CLS-style summaries — are 4b/later options.)
- **The embedding is trained for prediction, not retrieval.** There is no
  contrastive or metric-learning objective; retrieval quality is an emergent
  property. This is deliberate (keeps training label-free) but means retrieval is
  an indirect use of the representation.
- **Small collection** remains the binding constraint; all conclusions are about
  this data scale.

---

## Outputs

- `stage4a_lstm_results.csv` — retrieval metrics for the LSTM embeddings.
- `stage4a_lstm_embeddings.npy` — per-recording embeddings (for 4b comparison and
  allied-raga stratified analysis).
- `stage4a_lstm.pt` — trained model weights.
- Training log (epoch losses, early-stop point, best held-out-artist perplexity).

---

## One-line summary

Stage 4a trains a small, label-free LSTM to predict the next swara — forcing it to
learn sanchara — and tests whether its pooled, directional, longer-context
representation recovers the movement signal on allied ragas that first-order
statistics provably could not.
