"""
Stage 9 -- ColBERT-style LATE INTERACTION over swara sequences (COLAB VERSION)
=============================================================
Paste this whole thing into ONE Colab cell and run. Trains by default.
Set TRAIN = False to only evaluate the warm-started encoder.

Targets the pooling bottleneck: every learned representation so far collapses a
sequence into ONE vector (mean-pooling). ColBERT [Khattab & Zaharia, SIGIR 2020]
keeps PER-TOKEN embeddings and scores pairs with MaxSim:

    score(Q, D) = sum_{q in Q} max_{d in D} cos(E_q, E_d)

Adaptations (state in the paper):
  - ColBERT assumes short query vs. long document; here both are recordings, so
    we sample WINDOW-length query segments and average MaxSim over several.
  - ColBERT's power comes from contextual embeddings over a LARGE vocabulary.
    Ours has 12 symbols -- open question whether MaxSim recovers anything beyond
    note counting.

Requires: frozen swara_sequences.jsonl + stage4a_lstm.pt.  USE A GPU RUNTIME.
-------------------------------------------------------------
"""

import os
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ CONFIG ============
TRAIN = True             # <-- set False to skip training (warm-start eval only)

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
TOKEN_DIM = 32
WINDOW = 64
DOC_MAX = 1024
TEMP = 0.1
BATCH_RECS = 16
LR = 3e-4
MAX_EPOCHS = 60
PATIENCE = 8
N_QUERY_SEGMENTS = 4


# ============ DATA ============
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


def pad_batch(seqs, maxlen):
    x = torch.full((len(seqs), maxlen), STOI[PAD], dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = min(len(s), maxlen)
        x[i, :n] = torch.tensor(s[:n]); mask[i, :n] = True
    return x, mask


# ============ MODEL ============
class LateInteractionEncoder(nn.Module):
    """LSTM contextual encoder + per-token projection. No pooling anywhere."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VSIZE, EMB_DIM, padding_idx=STOI[PAD])
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HID_DIM, VSIZE)          # for 4a weight loading
        self.tok_proj = nn.Linear(HID_DIM, TOKEN_DIM)  # ColBERT-style projection

    def token_embeddings(self, x):
        h, _ = self.lstm(self.emb(x))
        return F.normalize(self.tok_proj(h), dim=-1)   # (B, L, TOKEN_DIM)


def maxsim(q_tokens, d_tokens):
    """q: (Lq, dim)  d: (Ld, dim) -> scalar. Each query token's best match."""
    return (q_tokens @ d_tokens.T).max(dim=1).values.sum()


def maxsim_batch(Q, D, d_mask):
    """Q: (B, Lq, dim)  D: (B, Ld, dim) -> (B, B) all-pairs scores."""
    sim = torch.matmul(Q.unsqueeze(1), D.unsqueeze(0).transpose(-1, -2))  # (B,B,Lq,Ld)
    if d_mask is not None:
        sim = sim.masked_fill(~d_mask.unsqueeze(0).unsqueeze(2), -1e4)
    return sim.max(dim=-1).values.sum(dim=-1)


# ============ TRAIN ============
def train_late_interaction(model, train_recs, val_recs):
    """Label-free contrastive with MaxSim scoring: positives = two segments of
    the same recording; negatives = other recordings in the batch."""
    rng = random.Random(SEED)

    def sample_pair(ids):
        if len(ids) < WINDOW * 2 + 2:
            return None
        a = rng.randint(0, len(ids) - WINDOW - 1)
        b = rng.randint(0, len(ids) - WINDOW - 1)
        return ids[a:a + WINDOW], ids[b:b + WINDOW]

    tr_ids = [x for x in (encode(r["collapsed_seq"]) for r in train_recs)
              if len(x) >= WINDOW * 2 + 2]
    va_ids = [x for x in (encode(r["collapsed_seq"]) for r in val_recs)
              if len(x) >= WINDOW * 2 + 2]
    print(f"train recs: {len(tr_ids)}   val recs (held-out artists): {len(va_ids)}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, bad = float("inf"), None, 0

    def run_epoch(ids_list, train=True):
        model.train(train)
        idxs = list(range(len(ids_list)))
        rng.shuffle(idxs)
        total, nb = 0.0, 0
        for s in range(0, len(idxs) - BATCH_RECS + 1, BATCH_RECS):
            batch = [sample_pair(ids_list[i]) for i in idxs[s:s + BATCH_RECS]]
            batch = [b for b in batch if b]
            if len(batch) < 2:
                continue
            qx, _ = pad_batch([b[0] for b in batch], WINDOW)
            dx, dm = pad_batch([b[1] for b in batch], WINDOW)
            qx, dx, dm = qx.to(DEVICE), dx.to(DEVICE), dm.to(DEVICE)
            Q = model.token_embeddings(qx)
            D = model.token_embeddings(dx)
            scores = maxsim_batch(Q, D, dm) / TEMP
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
            best = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {ep}."); break
    if best_state:
        model.load_state_dict(best_state)
        print(f"Best val loss: {best:.3f}")
    return model


# ============ EVALUATE ============
@torch.no_grad()
def build_token_index(model, recs):
    model.eval()
    out = []
    for r in recs:
        ids = encode(r["collapsed_seq"])[:DOC_MAX]
        x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        out.append(model.token_embeddings(x).squeeze(0))
    return out


@torch.no_grad()
def evaluate_maxsim(model, recs, ragas, ks=(1, 3, 5)):
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
        qt = docs[i]
        L = qt.size(0)
        segs = []
        for _ in range(N_QUERY_SEGMENTS):
            if L <= WINDOW:
                segs.append(qt)
            else:
                s = rng.randint(0, L - WINDOW)
                segs.append(qt[s:s + WINDOW])

        scores = np.full(len(recs), -np.inf)
        for j in range(len(recs)):
            if j == i:
                continue
            dj = docs[j]
            scores[j] = float(sum(maxsim(q, dj) for q in segs)) / len(segs)

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

    row = {"representation": "colbert_late_interaction"
                             + ("_trained" if TRAIN else "_warmstart"),
           "evaluable_queries": evaluable}
    for k in ks:
        row[f"P@{k}"] = round(float(np.mean(precisions[k])), 3)
    row["MAP"] = round(float(np.mean(aps)), 3)
    row["MRR"] = round(float(np.mean(rrs)), 3)
    return row


# ============ RUN ============
recs = load_records()
ragas = [r["raga"].lower() for r in recs]
print(f"Corpus: {len(recs)} recordings, {len(set(ragas))} ragas. Device: {DEVICE}")

model = LateInteractionEncoder().to(DEVICE)
if os.path.exists(LSTM_CKPT):
    missing, _ = model.load_state_dict(
        torch.load(LSTM_CKPT, map_location=DEVICE), strict=False)
    print(f"Warm-started from 4a LSTM (tok_proj fresh); missing={len(missing)}")
else:
    print("No 4a checkpoint -- encoder trained from scratch.")

if TRAIN:
    train_recs, val_recs = artist_split(recs)
    model = train_late_interaction(model, train_recs, val_recs)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "stage9_colbert.pt"))

row = evaluate_maxsim(model, recs, ragas)
df = pd.DataFrame([row])
print("\n=== Retrieval with LATE INTERACTION (MaxSim) ===")
print(df.to_string(index=False))
print("\nReference: transitions 0.431 | bag_of_notes 0.423 | contrastive 0.366±0.025 "
      "| lstm_lm 0.358 | colbert(warmstart) 0.317 | transformer 0.276 | bm25 0.024")
tag = "trained" if TRAIN else "warmstart"
df.to_csv(os.path.join(OUT_DIR, f"stage9_colbert_{tag}.csv"), index=False)
print(f"Saved -> {OUT_DIR}/stage9_colbert_{tag}.csv")
