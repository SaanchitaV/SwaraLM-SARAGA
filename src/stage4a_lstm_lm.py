"""
Stage 4a -- Swara Language Model: LSTM
=============================================================
Trains a small LSTM on next-swara prediction (self-supervised, NO raga labels),
then uses its hidden states as recording embeddings for retrieval, evaluated
exactly like Stages 2/3 so results are directly comparable.

Design choices driven by the small collection (176 recordings):
  - Small model (1-layer LSTM, modest hidden size) + dropout + early stopping
  - Training on overlapping WINDOWS of the swara sequences (176 recordings ->
    thousands of training examples)
  - Held-out-ARTIST validation split: the model is validated on artists it never
    trained on, so low val loss means swara grammar, not memorised recordings
  - Same retrieval evaluation (cosine, P@k / MAP / MRR + allied-raga stratum)

Stage 4b will swap the encoder for a tiny Transformer, reusing everything else,
so the LSTM-vs-Transformer comparison is apples-to-apples.

Requires: swara_sequences.jsonl. Install: pip install torch (GPU runtime helps).
-------------------------------------------------------------
"""

import os
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEQ_PATH = "/content/census_out/swara_sequences.jsonl"
OUT_DIR = "/content/census_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- vocabulary ----------------
POS12 = ["S", "R1", "R2", "G2", "G3", "M1", "M2", "P", "D1", "D2", "N2", "N3"]
PAD, SIL = "<pad>", "_"
VOCAB = [PAD, SIL] + POS12
STOI = {s: i for i, s in enumerate(VOCAB)}
VSIZE = len(VOCAB)

# ---------------- hyperparams (small on purpose) ----------------
WINDOW = 64          # swara tokens per training window
STRIDE = 32          # overlap between windows
EMB_DIM = 32
HID_DIM = 64
DROPOUT = 0.3
BATCH = 64
LR = 3e-3
MAX_EPOCHS = 60
PATIENCE = 6         # early stopping on held-out-artist val loss
VAL_ARTIST_FRAC = 0.2


# =============================================================
# DATA
# =============================================================
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


def make_windows(ids, window=WINDOW, stride=STRIDE):
    out = []
    for start in range(0, max(1, len(ids) - window), stride):
        w = ids[start:start + window + 1]     # +1: next-token targets
        if len(w) >= 8:                        # skip tiny tails
            out.append(w)
    return out


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        x = torch.full((WINDOW,), STOI[PAD], dtype=torch.long)
        y = torch.full((WINDOW,), -100, dtype=torch.long)   # -100 = ignore in loss
        n = len(w) - 1
        x[:n] = torch.tensor(w[:-1])
        y[:n] = torch.tensor(w[1:])
        return x, y


def artist_split(recs, frac=VAL_ARTIST_FRAC):
    """Hold out whole ARTISTS for validation -- the anti-cheating split."""
    artists = sorted({r.get("artist", "UNKNOWN") for r in recs})
    rng = random.Random(SEED)
    rng.shuffle(artists)
    n_val = max(1, int(len(artists) * frac))
    val_artists = set(artists[:n_val])
    train = [r for r in recs if r.get("artist") not in val_artists]
    val = [r for r in recs if r.get("artist") in val_artists]
    return train, val, val_artists


# =============================================================
# MODEL
# =============================================================
class SwaraLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VSIZE, EMB_DIM, padding_idx=STOI[PAD])
        self.lstm = nn.LSTM(EMB_DIM, HID_DIM, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HID_DIM, VSIZE)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.head(self.drop(h)), h    # logits, hidden states

    @torch.no_grad()
    def embed_sequence(self, ids, max_len=2048):
        """Mean-pooled hidden state over the (truncated) sequence -> embedding."""
        self.eval()
        ids = ids[:max_len]
        x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        _, h = self.forward(x)
        return h.squeeze(0).mean(0).cpu().numpy()


# =============================================================
# TRAIN
# =============================================================
def train_model(train_recs, val_recs):
    tr_windows, va_windows = [], []
    for r in train_recs:
        tr_windows += make_windows(encode(r["collapsed_seq"]))
    for r in val_recs:
        va_windows += make_windows(encode(r["collapsed_seq"]))
    print(f"Train windows: {len(tr_windows)}   Val windows (held-out artists): {len(va_windows)}")

    tr_dl = DataLoader(WindowDataset(tr_windows), batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(WindowDataset(va_windows), batch_size=BATCH)

    model = SwaraLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss, nb = 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = lossf(logits.reshape(-1, VSIZE), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item(); nb += 1
        model.eval()
        va_loss, vb = 0.0, 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                va_loss += lossf(logits.reshape(-1, VSIZE), y.reshape(-1)).item(); vb += 1
        tr_loss /= max(nb, 1); va_loss /= max(vb, 1)
        print(f"epoch {epoch:02d}  train_loss={tr_loss:.3f}  "
              f"val_loss(held-out artists)={va_loss:.3f}  ppl={math.exp(va_loss):.2f}")
        if va_loss < best_val - 1e-3:
            best_val, best_state, bad = va_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {epoch} (no val improvement for {PATIENCE}).")
                break
    model.load_state_dict(best_state)
    print(f"Best held-out-artist val loss: {best_val:.3f} (ppl {math.exp(best_val):.2f})")
    return model


# =============================================================
# RETRIEVAL EVALUATION  (identical protocol to Stage 2/3)
# =============================================================
def cosine(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    Xn = X / n
    return Xn @ Xn.T


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


# =============================================================
# MAIN
# =============================================================
def main():
    recs = load_records()
    print(f"Loaded {len(recs)} recordings, {len(set(r['raga'].lower() for r in recs))} ragas. Device: {DEVICE}")

    train_recs, val_recs, val_artists = artist_split(recs)
    print(f"Held-out validation artists ({len(val_artists)}): {sorted(val_artists)}")

    model = train_model(train_recs, val_recs)

    # embed ALL recordings with the trained model, evaluate retrieval on all
    embs = np.array([model.embed_sequence(encode(r["collapsed_seq"])) for r in recs])
    ragas = [r["raga"].lower() for r in recs]

    result = evaluate_retrieval(embs, ragas, "lstm_lm_embedding")
    df = pd.DataFrame([result])
    print("\n=== Retrieval with LSTM-LM embeddings ===")
    print(df.to_string(index=False))
    print("\nBaselines to beat (Stage 2): bag_of_notes P@1=0.423, transitions P@1=0.431")
    df.to_csv(os.path.join(OUT_DIR, "stage4a_lstm_results.csv"), index=False)

    # save embeddings + model for Stage 4b comparison and further analysis
    np.save(os.path.join(OUT_DIR, "stage4a_lstm_embeddings.npy"), embs)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "stage4a_lstm.pt"))
    print(f"\nSaved results, embeddings, model -> {OUT_DIR}")


if __name__ == "__main__":
    main()