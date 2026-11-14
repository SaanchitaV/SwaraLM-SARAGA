"""
Stage 5 -- Contrastive fine-tuning: aligning the embedding with retrieval
=============================================================
THE PIVOTAL EXPERIMENT.

Diagnosis from 4a: generative next-swara prediction learns melodic grammar that is
SHARED across ragas, and mean-pooling erases what little discriminative signal
exists -- hence retrieval collapse (P@1 0.179 vs baseline 0.423). The failure was
the OBJECTIVE, not the architecture.

Fix: fine-tune the encoder with a contrastive objective that is still LABEL-FREE:
  positives  = two different windows FROM THE SAME RECORDING
  negatives  = windows from OTHER recordings (in-batch negatives, InfoNCE)
The embedding is thereby optimised to be DISCRIMINATIVE between recordings.
Because same-raga recordings share movement patterns, recording-discrimination
should transfer to raga-discrimination -- WITHOUT ever using a raga label.

Anti-shortcut note: the model could cheat by learning artist/recording style
rather than melodic content. Two mitigations are built in:
  (1) the encoder input is ONLY the swara symbols (no audio/timbre available);
  (2) evaluation includes the held-out-artist stratum -- if it learned style,
      transfer to unseen artists' recordings will fail visibly.

Pipeline: load 4a's LSTM (or train from scratch if missing) -> contrastive
fine-tune -> re-embed -> identical retrieval evaluation -> compare all five
representations.

Requires: swara_sequences.jsonl (+ optionally stage4a_lstm.pt to warm-start).
-------------------------------------------------------------
"""

import os
import json
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

WINDOW = 64
EMB_DIM = 32
HID_DIM = 64
DROPOUT = 0.3
PROJ_DIM = 32        # contrastive projection head output
BATCH_RECS = 16      # recordings per batch (2 windows each -> 64 windows)
TEMP = 0.1           # InfoNCE temperature
LR = 3e-4
MAX_EPOCHS = 100
PATIENCE = 10
VAL_ARTIST_FRAC = 0.2


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


class TwoViewDataset(Dataset):
    """Each item = one recording; returns TWO random windows from it (the
    positive pair). In-batch windows from other recordings are the negatives."""
    def __init__(self, recs):
        self.ids = [encode(r["collapsed_seq"]) for r in recs]
        self.ids = [x for x in self.ids if len(x) >= WINDOW + 8]

    def __len__(self):
        return len(self.ids)

    def _rand_window(self, ids):
        start = random.randint(0, len(ids) - WINDOW - 1)
        w = ids[start:start + WINDOW]
        x = torch.full((WINDOW,), STOI[PAD], dtype=torch.long)
        x[:len(w)] = torch.tensor(w)
        return x

    def __getitem__(self, i):
        ids = self.ids[i]
        return self._rand_window(ids), self._rand_window(ids)


def artist_split(recs, frac=VAL_ARTIST_FRAC):
    artists = sorted({r.get("artist", "UNKNOWN") for r in recs})
    rng = random.Random(SEED)
    rng.shuffle(artists)
    n_val = max(1, int(len(artists) * frac))
    val_artists = set(artists[:n_val])
    return ([r for r in recs if r.get("artist") not in val_artists],
            [r for r in recs if r.get("artist") in val_artists],
            val_artists)


# ---------------- model: LSTM encoder + projection head ----------------
class SwaraLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VSIZE, EMB_DIM, padding_idx=STOI[PAD])
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HID_DIM, VSIZE)   # kept so 4a weights load cleanly
        self.proj = nn.Sequential(               # contrastive projection head
            nn.Linear(HID_DIM, HID_DIM), nn.ReLU(), nn.Linear(HID_DIM, PROJ_DIM))

    def encode_window(self, x):
        h, _ = self.lstm(self.emb(x))
        pooled = h.mean(1)                       # mean over time
        return self.proj(pooled)                 # projected embedding

    @torch.no_grad()
    def embed_sequence(self, ids, max_len=2048):
        self.eval()
        ids = ids[:max_len]
        chunks = [ids[i:i + WINDOW] for i in range(0, len(ids), WINDOW)]
        vecs = []
        for c in chunks:
            if len(c) < 4:
                continue
            x = torch.full((1, WINDOW), STOI[PAD], dtype=torch.long, device=DEVICE)
            x[0, :len(c)] = torch.tensor(c, device=DEVICE)
            vecs.append(self.encode_window(x).squeeze(0).cpu().numpy())
        return np.mean(vecs, axis=0) if vecs else np.zeros(PROJ_DIM)


def info_nce(z1, z2, temp=TEMP):
    """Standard InfoNCE with in-batch negatives. z1[i]<->z2[i] are positives."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temp                    # (B,B); diagonal = positives
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.T, labels))


# ---------------- train ----------------
def train_contrastive(train_recs, val_recs):
    model = SwaraLSTMEncoder().to(DEVICE)
    if os.path.exists(LSTM_CKPT):
        state = torch.load(LSTM_CKPT, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Warm-started from 4a LSTM (proj head fresh). missing={len(missing)}")
    else:
        print("No 4a checkpoint found -- training encoder from scratch.")

    tr_dl = DataLoader(TwoViewDataset(train_recs), batch_size=BATCH_RECS,
                       shuffle=True, drop_last=True)
    va_dl = DataLoader(TwoViewDataset(val_recs), batch_size=BATCH_RECS,
                       drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr, nb = 0.0, 0
        for v1, v2 in tr_dl:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            loss = info_nce(model.encode_window(v1), model.encode_window(v2))
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item(); nb += 1
        model.eval()
        va, vb = 0.0, 0
        with torch.no_grad():
            for v1, v2 in va_dl:
                v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
                va += info_nce(model.encode_window(v1), model.encode_window(v2)).item()
                vb += 1
        tr /= max(nb, 1); va /= max(vb, 1)
        print(f"epoch {epoch:02d}  contrastive train={tr:.3f}  val(held-out artists)={va:.3f}")
        if va < best_val - 1e-3:
            best_val, best_state, bad = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {epoch}.")
                break
    model.load_state_dict(best_state)
    return model


# ---------------- retrieval eval (identical protocol) ----------------
def cosine(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return (X / n) @ (X / n).T


def evaluate_retrieval(X, ragas, name, ks=(1, 3, 5)):
    sim = cosine(X)
    counts = Counter(ragas)
    precisions = {k: [] for k in ks}
    aps, rrs = [], []
    evaluable = 0
    for i in range(len(ragas)):
        if counts[ragas[i]] < 2:
            continue
        evaluable += 1
        order = np.argsort(-sim[i]); order = order[order != i]
        rel = np.array([ragas[j] == ragas[i] for j in order], dtype=int)
        for k in ks:
            precisions[k].append(rel[:k].sum() / k)
        if rel.sum():
            cum = np.cumsum(rel)
            prec_at = cum / (np.arange(len(rel)) + 1)
            aps.append((prec_at * rel).sum() / rel.sum())
            rrs.append(1.0 / (np.argmax(rel) + 1))
    row = {"representation": name, "evaluable_queries": evaluable}
    for k in ks:
        row[f"P@{k}"] = round(float(np.mean(precisions[k])), 3)
    row["MAP"] = round(float(np.mean(aps)), 3)
    row["MRR"] = round(float(np.mean(rrs)), 3)
    return row


# ---------------- main ----------------
def main():
    recs = load_records()
    print(f"Loaded {len(recs)} recordings. Device: {DEVICE}")
    train_recs, val_recs, val_artists = artist_split(recs)
    print(f"Held-out artists: {sorted(val_artists)}")

    model = train_contrastive(train_recs, val_recs)

    embs = np.array([model.embed_sequence(encode(r["collapsed_seq"])) for r in recs])
    ragas = [r["raga"].lower() for r in recs]

    result = evaluate_retrieval(embs, ragas, "contrastive_lstm_embedding")
    df = pd.DataFrame([result])
    print("\n=== Retrieval with CONTRASTIVE embeddings (the pivotal result) ===")
    print(df.to_string(index=False))
    print("\nFull comparison (P@1): bag_of_notes=0.423, transitions=0.431, "
          "lstm_lm=0.179, transformer_lm=<4b>, contrastive=<this>")
    df.to_csv(os.path.join(OUT_DIR, "stage5_contrastive_results.csv"), index=False)
    np.save(os.path.join(OUT_DIR, "stage5_contrastive_embeddings.npy"), embs)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "stage5_contrastive.pt"))
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()