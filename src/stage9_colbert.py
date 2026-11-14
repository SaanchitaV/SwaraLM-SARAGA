"""
Stage 9 -- ColBERT-style LATE INTERACTION over swara sequences
=============================================================
Directly targets the pooling bottleneck this project identified: every learned
representation so far collapses a sequence into ONE vector (mean-pooling),
which destroys fine-grained structure. ColBERT [Khattab & Zaharia, SIGIR 2020]
avoids that: keep PER-TOKEN embeddings, and score a pair by MaxSim --

    score(Q, D) = sum_{q in Q} max_{d in D} cos(E_q, E_d)

Adaptation notes (state these honestly in the paper):
  - ColBERT assumes ASYMMETRY: a short query vs. long documents. Here both
    sides are whole recordings, so we sample a fixed-length query segment
    (WINDOW tokens) and score it against full documents. This keeps the
    late-interaction spirit while bounding cost.
  - ColBERT's power comes from contextualized embeddings over a LARGE
    vocabulary. Ours has 12 symbols. The open question -- stated in advance --
    is whether MaxSim over swara-level contextual embeddings recovers anything
    beyond note counting, or collapses toward it.

Two encoder options:
  A) reuse the trained 4a LSTM (contextual per-token hidden states)  [default]
  B) train the encoder with a late-interaction contrastive objective
     (--train), positives = two segments of the same recording, MaxSim scoring

Evaluation is the SAME protocol (P@k / MAP / MRR, allied stratum) so the row
drops straight into the main table.

Requires: frozen jsonl + stage4a_lstm.pt.  GPU strongly recommended for --train.
-------------------------------------------------------------
"""

import os
import json
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_PATH = "/content/census_out/swara_sequences.jsonl"
LSTM_CKPT = "/content/census_out/stage4a_lstm.pt"
OUT_DIR = "/content/census_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
PAD, SIL = "<pad>", "_"
VOCAB = [PAD, SIL] + POS12
STOI = {s: i for i, s in enumerate(VOCAB)}
VSIZE = len(VOCAB)

EMB_DIM, HID_DIM, DROPOUT = 32, 64, 0.3
TOKEN_DIM = 32           # per-token projection dim (ColBERT's compressed dim)
WINDOW = 64              # query segment length
DOC_MAX = 1024           # truncate documents for tractable MaxSim
TEMP = 0.1
BATCH_RECS = 16
LR = 3e-4
MAX_EPOCHS = 60
PATIENCE = 8


# ---------------- data ----------------
def load_records():
    recs = []
    with open(SEQ_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("raga") and r["raga"].lower() not in ("unknown", "ragamalika"):
                recs.append(r)
    return recs


def encode(seq):
    return [STOI[s] for s in seq if s in STOI]


def artist_split(recs, seed=SEED, frac=0.2):
    artists = sorted({r.get("artist", "UNKNOWN") for r in recs})
    rng = random.Random(seed); rng.shuffle(artists)
    val_artists = set(artists[:max(1, int(len(artists) * frac))])
    return ([r for r in recs if r.get("artist") not in val_artists],
            [r for r in recs if r.get("artist") in val_artists])


# ---------------- encoder: per-token embeddings ----------------
class LateInteractionEncoder(nn.Module):
    """LSTM contextual encoder + per-token projection (no pooling anywhere)."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VSIZE, EMB_DIM, padding_idx=STOI[PAD])
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HID_DIM, VSIZE)          # kept for 4a weight loading
        self.tok_proj = nn.Linear(HID_DIM, TOKEN_DIM)  # ColBERT-style projection

    def token_embeddings(self, x):
        """(B, L) -> (B, L, TOKEN_DIM), L2-normalized per token."""
        h, _ = self.lstm(self.emb(x))
        z = self.tok_proj(h)
        return F.normalize(z, dim=-1)


def maxsim(q_tokens, d_tokens, d_mask=None):
    """ColBERT MaxSim.
    q_tokens: (Lq, dim)  d_tokens: (Ld, dim)  -> scalar
    Each query token takes its best match in the document; sum over query."""
    sim = q_tokens @ d_tokens.T                 # (Lq, Ld) -- cosine (normalized)
    if d_mask is not None:
        sim = sim.masked_fill(~d_mask.unsqueeze(0), -1e4)
    return sim.max(dim=1).values.sum()


def maxsim_batch(Q, D, d_mask):
    """Batched: Q (B, Lq, dim), D (B, Ld, dim) -> (B, B) score matrix
    (every query against every document in the batch)."""
    # (B, 1, Lq, dim) x (1, B, dim, Ld) -> (B, B, Lq, Ld)
    sim = torch.einsum("qld,bkd->qbl k".replace(" ", ""), Q, D) \
        if False else torch.matmul(Q.unsqueeze(1), D.unsqueeze(0).transpose(-1, -2))
    if d_mask is not None:
        m = d_mask.unsqueeze(0).unsqueeze(2)     # (1, B, 1, Ld)
        sim = sim.masked_fill(~m, -1e4)
    return sim.max(dim=-1).values.sum(dim=-1)    # (B, B)


# ---------------- training (optional) ----------------
def pad_batch(seqs, maxlen):
    x = torch.full((len(seqs), maxlen), STOI[PAD], dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = min(len(s), maxlen)
        x[i, :n] = torch.tensor(s[:n]); mask[i, :n] = True
    return x, mask


def train_late_interaction(model, train_recs, val_recs):
    """Contrastive with MaxSim scoring: positives = two segments of the same
    recording; in-batch negatives. Label-free, same as Stage 5."""
    def sample_pair(ids, rng):
        if len(ids) < WINDOW * 2 + 2:
            return None
        a = rng.randint(0, len(ids) - WINDOW - 1)
        b = rng.randint(0, len(ids) - WINDOW - 1)
        return ids[a:a + WINDOW], ids[b:b + WINDOW]

    tr_ids = [encode(r["collapsed_seq"]) for r in train_recs]
    va_ids = [encode(r["collapsed_seq"]) for r in val_recs]
    tr_ids = [x for x in tr_ids if len(x) >= WINDOW * 2 + 2]
    va_ids = [x for x in va_ids if len(x) >= WINDOW * 2 + 2]
    print(f"train recs: {len(tr_ids)}  val recs: {len(va_ids)}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, bad = float("inf"), None, 0
    rng = random.Random(SEED)

    def run_epoch(ids_list, train=True):
        model.train(train)
        idxs = list(range(len(ids_list)))
        rng.shuffle(idxs)
        total, nb = 0.0, 0
        for s in range(0, len(idxs) - BATCH_RECS + 1, BATCH_RECS):
            batch = [sample_pair(ids_list[i], rng) for i in idxs[s:s + BATCH_RECS]]
            batch = [b for b in batch if b]
            if len(batch) < 2:
                continue
            q_seqs = [b[0] for b in batch]; d_seqs = [b[1] for b in batch]
            qx, _ = pad_batch(q_seqs, WINDOW); dx, dm = pad_batch(d_seqs, WINDOW)
            qx, dx, dm = qx.to(DEVICE), dx.to(DEVICE), dm.to(DEVICE)
            Q = model.token_embeddings(qx); D = model.token_embeddings(dx)
            scores = maxsim_batch(Q, D, dm) / TEMP           # (B, B)
            labels = torch.arange(scores.size(0), device=DEVICE)
            loss = 0.5 * (F.cross_entropy(scores, labels) +
                          F.cross_entropy(scores.T, labels))
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item(); nb += 1
        return total / max(nb, 1)

    for ep in range(1, MAX_EPOCHS + 1):
        tr = run_epoch(tr_ids, True)
        with torch.no_grad():
            va = run_epoch(va_ids, False)
        print(f"epoch {ep:02d}  train={tr:.3f}  val(held-out artists)={va:.3f}")
        if va < best - 1e-3:
            best, best_state, bad = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {ep}."); break
    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------- retrieval with MaxSim ----------------
@torch.no_grad()
def build_token_index(model, recs):
    """Per-recording token embeddings (truncated to DOC_MAX)."""
    model.eval()
    out = []
    for r in recs:
        ids = encode(r["collapsed_seq"])[:DOC_MAX]
        x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        out.append(model.token_embeddings(x).squeeze(0))   # (L, dim)
    return out


@torch.no_grad()
def evaluate_maxsim(model, recs, ragas, n_query_segments=4, ks=(1, 3, 5)):
    """Query = mean of MaxSim scores over n sampled segments of the query
    recording (reduces variance vs. a single arbitrary segment)."""
    docs = build_token_index(model, recs)
    rng = random.Random(SEED)
    counts = Counter(ragas)
    precisions = {k: [] for k in ks}
    aps, rrs = [], []
    evaluable = 0

    for i in range(len(recs)):
        if counts[ragas[i]] < 2:
            continue
        evaluable += 1
        qi_tokens = docs[i]
        # sample query segments from the query recording
        segs = []
        L = qi_tokens.size(0)
        for _ in range(n_query_segments):
            if L <= WINDOW:
                segs.append(qi_tokens)
            else:
                s = rng.randint(0, L - WINDOW)
                segs.append(qi_tokens[s:s + WINDOW])

        scores = np.zeros(len(recs))
        for j in range(len(recs)):
            if j == i:
                scores[j] = -np.inf
                continue
            dj = docs[j]
            sc = 0.0
            for q in segs:
                sc += float(maxsim(q, dj))
            scores[j] = sc / len(segs)

        order = np.argsort(-scores)
        order = order[order != i]
        rel = np.array([ragas[j] == ragas[i] for j in order], dtype=int)
        for k in ks:
            precisions[k].append(rel[:k].sum() / k)
        if rel.sum():
            cum = np.cumsum(rel)
            prec_at = cum / (np.arange(len(rel)) + 1)
            aps.append((prec_at * rel).sum() / rel.sum())
            rrs.append(1.0 / (np.argmax(rel) + 1))

    row = {"representation": "colbert_late_interaction",
           "evaluable_queries": evaluable}
    for k in ks:
        row[f"P@{k}"] = round(float(np.mean(precisions[k])), 3)
    row["MAP"] = round(float(np.mean(aps)), 3)
    row["MRR"] = round(float(np.mean(rrs)), 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="fine-tune with late-interaction contrastive objective")
    args, _ = ap.parse_known_args()

    recs = load_records()
    ragas = [r["raga"].lower() for r in recs]
    print(f"Corpus: {len(recs)} recordings. Device: {DEVICE}")

    model = LateInteractionEncoder().to(DEVICE)
    if os.path.exists(LSTM_CKPT):
        missing, _ = model.load_state_dict(
            torch.load(LSTM_CKPT, map_location=DEVICE), strict=False)
        print(f"Warm-started from 4a LSTM (tok_proj fresh); missing={len(missing)}")

    if args.train:
        train_recs, val_recs = artist_split(recs)
        model = train_late_interaction(model, train_recs, val_recs)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "stage9_colbert.pt"))

    row = evaluate_maxsim(model, recs, ragas)
    df = pd.DataFrame([row])
    print("\n=== Retrieval with LATE INTERACTION (MaxSim) ===")
    print(df.to_string(index=False))
    print("\nReference: bag_of_notes 0.423 | transitions 0.431 | lstm_lm 0.358 | "
          "contrastive 0.423 | bm25 0.024")
    tag = "trained" if args.train else "warmstart"
    df.to_csv(os.path.join(OUT_DIR, f"stage9_colbert_{tag}.csv"), index=False)
    print(f"Saved -> {OUT_DIR}/stage9_colbert_{tag}.csv")


if __name__ == "__main__":
    main()
