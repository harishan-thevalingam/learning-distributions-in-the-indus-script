# =====================================================================
# indus_nplm_train.py
#
# A Neural Probabilistic Language Model (Bengio et al. 2003/2006),
# scaled down from the paper's m=30-100/h=50-100/|V|=10k-100k to this
# corpus's size: |V|=377 signs, ~7000 tokens. Architecture is exactly
# Eq. 6.1 of the paper: y = b + Wx + U*tanh(d + Hx), softmax(y), with
# x the concatenation of the context signs' embeddings and W optional
# (direct input->output connections).
#
# CONFIGS swept: context order n in {1,2,3} (bracketing the existing
# order-1/2/3 Markov ladder from earlier work, though that comparison
# is intentionally not made numerically in this script -- see
# indus_nplm_analyse.py's docstring) x direct connections on/off
# (Bengio's Table 6.1 shows this matters), x 10 seeds = 60 models.
# m=24 (matches word2vec's choice), h=32.
#
# SPLIT: whole-inscription 85/15, fixed ONCE via split_seed=1234 and
# reused for every model (matches this project's established practice
# of holding the split fixed so model-vs-model comparisons don't pick
# up split variance, which earlier work found to be larger than the
# entire spread across several architecture choices). Note this is a
# fresh Python-side split -- it is not literally the same held-out set
# as any earlier Julia-side split (Julia's RNG doesn't reproduce in
# Python), so held-out numbers from this script are not directly
# comparable to earlier ppl figures; they're only compared internally,
# NPLM-config to NPLM-config.
#
# Per-position training examples: for each inscription, every real
# token is a target, with context = the preceding n tokens, left-
# padded with <bos> for positions near the start (so the model also
# learns to predict a text's first sign, from an all-<bos> context --
# useful for later start_rate-style analysis).
#
# Weight decay applied to C, H, U, W only (not biases b, d), matching
# Bengio's footnote 1. Early stopping on held-out loss, patience=20
# (matches the patience already used elsewhere in this project).
#
# Requires: pip install torch numpy
# Run in the same folder as induscorpus.txt: python indus_nplm_train.py
# =====================================================================

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CORPUS_PATH = "induscorpus.txt"
OUT_DIR = "nplm_models"
M = 24          # embedding dim, matches word2vec's choice
H = 32          # hidden units
SEEDS = list(range(10))
SPLIT_SEED = 1234
VAL_FRAC = 0.15
WEIGHT_DECAY = 1e-4
LR = 1e-2
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 20

CONFIGS = [
    dict(name="n1_nodirect", n=1, direct=False),
    dict(name="n1_direct",   n=1, direct=True),
    dict(name="n2_nodirect", n=2, direct=False),
    dict(name="n2_direct",   n=2, direct=True),
    dict(name="n3_nodirect", n=3, direct=False),
    dict(name="n3_direct",   n=3, direct=True),
]


def load_inscriptions(path=CORPUS_PATH):
    inscriptions = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                inscriptions.append(line.split())
    return inscriptions


def build_vocab(inscriptions):
    """signs_sorted: real signs only, index 0..V-1 (this doubles as
    the TARGET index space, since <bos> is never a target). bos_idx =
    V, used only as a context/input index. Embedding table size is
    V+1."""
    signs_sorted = sorted({s for ins in inscriptions for s in ins}, key=int)
    sign_to_idx = {s: i for i, s in enumerate(signs_sorted)}
    bos_idx = len(signs_sorted)
    return signs_sorted, sign_to_idx, bos_idx


def split_inscriptions(inscriptions, val_frac=VAL_FRAC, split_seed=SPLIT_SEED):
    idx = list(range(len(inscriptions)))
    random.Random(split_seed).shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx = set(idx[:n_val])
    train = [ins for i, ins in enumerate(inscriptions) if i not in val_idx]
    val = [ins for i, ins in enumerate(inscriptions) if i in val_idx]
    return train, val


def build_examples(inscriptions, n, sign_to_idx, bos_idx):
    contexts, targets = [], []
    for ins in inscriptions:
        seq = [bos_idx] * n + [sign_to_idx[s] for s in ins]
        for t in range(n, len(seq)):
            contexts.append(seq[t - n:t])
            targets.append(seq[t])
    return (torch.tensor(contexts, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


class NPLM(nn.Module):
    """y = b + Wx + U*tanh(d + Hx); x = concat of context embeddings.
    W is the optional direct input->output connection (Bengio Eq 6.1);
    omitted entirely (not just zeroed) when direct=False, matching the
    paper's own "W is optionally zero" framing."""

    def __init__(self, vocab_size, out_size, n, m, h, direct=True):
        super().__init__()
        self.n = n
        self.C = nn.Embedding(vocab_size, m)
        self.H = nn.Linear(n * m, h)
        self.U = nn.Linear(h, out_size)
        self.direct = direct
        if direct:
            self.W = nn.Linear(n * m, out_size, bias=False)

    def context_features(self, ctx_idx):
        emb = self.C(ctx_idx)                    # (batch, n, m)
        return emb.reshape(emb.size(0), -1)       # (batch, n*m)

    def forward(self, ctx_idx, return_hidden=False):
        x = self.context_features(ctx_idx)
        hidden = torch.tanh(self.H(x))
        y = self.U(hidden)
        if self.direct:
            y = y + self.W(x)
        if return_hidden:
            return y, hidden
        return y


def param_groups(model):
    decay = [model.C.weight, model.H.weight, model.U.weight]
    if model.direct:
        decay.append(model.W.weight)
    nodecay = [model.H.bias, model.U.bias]
    return [
        dict(params=decay, weight_decay=WEIGHT_DECAY),
        dict(params=nodecay, weight_decay=0.0),
    ]


@torch.no_grad()
def evaluate_ppl(model, contexts, targets):
    model.eval()
    logits = model(contexts)
    nll = nn.functional.cross_entropy(logits, targets, reduction="mean")
    return float(torch.exp(nll))


def train_one(cfg, seed, train_ex, val_ex, vocab_size, out_size):
    torch.manual_seed(seed)
    model = NPLM(vocab_size, out_size, cfg["n"], M, H, direct=cfg["direct"])
    opt = torch.optim.Adam(param_groups(model), lr=LR)

    tr_ctx, tr_tgt = train_ex
    val_ctx, val_tgt = val_ex
    loader = DataLoader(TensorDataset(tr_ctx, tr_tgt), batch_size=BATCH_SIZE,
                         shuffle=True, generator=torch.Generator().manual_seed(seed))

    best_val, best_state, patience_left = float("inf"), None, PATIENCE
    for epoch in range(MAX_EPOCHS):
        model.train()
        for ctx_b, tgt_b in loader:
            opt.zero_grad()
            logits = model(ctx_b)
            loss = nn.functional.cross_entropy(logits, tgt_b)
            loss.backward()
            opt.step()

        val_ppl = evaluate_ppl(model, val_ctx, val_tgt)
        if val_ppl < best_val - 1e-4:
            best_val, best_state, patience_left = val_ppl, {
                k: v.clone() for k, v in model.state_dict().items()
            }, PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    train_ppl = evaluate_ppl(model, tr_ctx, tr_tgt)
    return model, train_ppl, best_val, epoch + 1


def main():
    inscriptions = load_inscriptions()
    signs_sorted, sign_to_idx, bos_idx = build_vocab(inscriptions)
    vocab_size = len(signs_sorted) + 1
    out_size = len(signs_sorted)
    print(f"{len(inscriptions)} inscriptions, {len(signs_sorted)} sign types "
          f"(vocab_size incl <bos> = {vocab_size})")

    train_ins, val_ins = split_inscriptions(inscriptions)
    print(f"split: {len(train_ins)} train inscriptions, {len(val_ins)} val "
          f"(fixed across all configs/seeds, split_seed={SPLIT_SEED})\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    rows = []

    for cfg in CONFIGS:
        train_ex = build_examples(train_ins, cfg["n"], sign_to_idx, bos_idx)
        val_ex = build_examples(val_ins, cfg["n"], sign_to_idx, bos_idx)
        for seed in SEEDS:
            model, train_ppl, val_ppl, epochs = train_one(
                cfg, seed, train_ex, val_ex, vocab_size, out_size
            )
            out_path = os.path.join(OUT_DIR, f"{cfg['name']}_seed{seed}.pt")
            torch.save(model.state_dict(), out_path)
            rows.append(dict(
                config=cfg["name"], seed=seed, n=cfg["n"], direct=cfg["direct"],
                m=M, h=H, weight_decay=WEIGHT_DECAY, epochs=epochs,
                train_ppl=train_ppl, val_ppl=val_ppl, path=out_path,
            ))
            print(f"  {cfg['name']} seed {seed}: {epochs:>3} epochs, "
                  f"train_ppl={train_ppl:.2f} val_ppl={val_ppl:.2f} -> {out_path}")

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} models trained. manifest -> {manifest_path}")
    print("Now run: python indus_nplm_analyse.py")


if __name__ == "__main__":
    main()
