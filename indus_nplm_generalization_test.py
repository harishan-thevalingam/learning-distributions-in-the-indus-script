# =====================================================================
# indus_nplm_generalization_test.py
#
# Tests the paper's actual founding claim (not tested anywhere else in
# this project so far, for any method): does a distributed
# representation let the model assign meaningful probability to
# combinations it never saw, because they resemble combinations it
# did see? Bengio's own framing: n-gram models "do not want to assign
# zero probability" to unseen sequences; distributed representations
# are proposed specifically as the fix.
#
# METHOD:
#   1. Use the SAME 85/15 train/val split as indus_nplm_train.py
#      (identical split_seed=1234) -- so "novel" means genuinely never
#      seen during this model's training, not just held out from some
#      other analysis.
#   2. Find bigrams that occur in validation but never once in
#      training -- signs the model was never directly shown adjacent.
#   3. Read the model's own predicted P(next|context) for each novel
#      pair -- this is the model actually generalizing, not anything
#      counted from data.
#   4. THE KEY TEST: does that predicted probability correlate with
#      CLUSTER COMPATIBILITY -- i.e. how often (in TRAINING only, no
#      leakage) do other members of the first sign's cluster get
#      followed by other members of the second sign's cluster? If the
#      model is generalizing via "this sign behaves like its cluster-
#      mates", predicted probability for novel pairs should track this.
#      Tested with Spearman correlation + a permutation-test null
#      (same style as this project's other permutation tests): shuffle
#      cluster labels, recompute the correlation, repeat 500x.
#
# Requires n=1 (bigram context) for the "novel BIGRAM" framing to be
# literal -- hardcoded to TARGET_CONFIG="n1_nodirect". Extending to
# n=2/n=3 (novel trigrams/4-grams) is a straightforward parameter
# change if wanted later, but changes what "novel" means (novel given
# 2 or 3 tokens of context, not 1).
#
# Requires: pip install torch scikit-learn numpy matplotlib
# Run in the same folder as induscorpus.txt and nplm_models/:
#   python indus_nplm_generalization_test.py
# =====================================================================

import csv
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CORPUS_PATH = "induscorpus.txt"
MANIFEST_PATH = "nplm_models/manifest.csv"
MIN_COUNT_CLUSTER = 15
MIN_CONTEXT_OCC = 5
SPLIT_SEED = 1234
VAL_FRAC = 0.15
N_PERM = 500

TARGET_CONFIG = "n1_nodirect"
TARGET_KIND = "softmax"
TARGET_SEED = 0


# ---------------------------------------------------------------------
# corpus + vocab + split (must match indus_nplm_train.py exactly)
# ---------------------------------------------------------------------
def load_inscriptions(path=CORPUS_PATH):
    inscriptions = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                inscriptions.append(line.split())
    return inscriptions


def build_vocab(inscriptions):
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


def sign_metadata(inscriptions):
    count, texts, ends, starts = (defaultdict(int) for _ in range(4))
    for ins in inscriptions:
        for s in ins:
            count[s] += 1
        for s in set(ins):
            texts[s] += 1
        ends[ins[-1]] += 1
        starts[ins[0]] += 1
    meta = {}
    for s in count:
        t = texts[s]
        meta[s] = dict(count=count[s], texts=t)
    return meta


# ---------------------------------------------------------------------
# model definition + clustering reproduction (must match earlier scripts)
# ---------------------------------------------------------------------
class NPLM(nn.Module):
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
        emb = self.C(ctx_idx)
        return emb.reshape(emb.size(0), -1)

    def forward(self, ctx_idx, return_hidden=False):
        x = self.context_features(ctx_idx)
        hidden = torch.tanh(self.H(x))
        y = self.U(hidden)
        if self.direct:
            y = y + self.W(x)
        if return_hidden:
            return y, hidden
        return y


def load_manifest(path=MANIFEST_PATH):
    configs = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            configs[row["config"]].append(row)
    return configs


def load_model(row, vocab_size, out_size):
    m = NPLM(vocab_size, out_size, int(row["n"]), int(row["m"]), int(row["h"]),
              direct=(row["direct"] == "True"))
    m.load_state_dict(torch.load(row["path"], map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def sign_representations(model, inscriptions, signs_sorted, sign_to_idx,
                          bos_idx, n, min_occ=MIN_CONTEXT_OCC):
    contexts = []
    for ins in inscriptions:
        seq = [bos_idx] * n + [sign_to_idx[s] for s in ins]
        for t in range(n, len(seq)):
            contexts.append(seq[t - n:t])
    ctx_idx = torch.tensor(contexts, dtype=torch.long)
    logits, hidden = model(ctx_idx, return_hidden=True)
    probs = torch.softmax(logits, dim=-1)
    last_token = ctx_idx[:, -1]
    softmax_rep = {}
    for i, s in enumerate(signs_sorted):
        mask = last_token == i
        if int(mask.sum()) < min_occ:
            continue
        softmax_rep[s] = probs[mask].mean(dim=0).numpy()
    return softmax_rep


def dict_to_matrix(d, order):
    kept = [s for s in order if s in d]
    X = np.stack([d[s] for s in kept]) if kept else np.zeros((0, 1))
    return kept, X


def best_kmeans(X, k_range=range(2, 11), seed=0):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    best_k, best_score, best_labels = None, -1.0, None
    for k in k_range:
        if k >= X.shape[0]:
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xn)
        score = silhouette_score(Xn, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_k, best_score, best_labels


def reproduce_clustering(inscriptions, meta, signs_sorted, sign_to_idx, bos_idx,
                          vocab_size, out_size, configs):
    row = next(r for r in configs[TARGET_CONFIG] if int(r["seed"]) == TARGET_SEED)
    n = int(row["n"])
    model = load_model(row, vocab_size, out_size)
    softmax_rep = sign_representations(
        model, inscriptions, signs_sorted, sign_to_idx, bos_idx, n
    )
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_CLUSTER], key=int
    )
    kept, X = dict_to_matrix(softmax_rep, kept_master)
    k, score, labels = best_kmeans(X)
    return kept, labels, k, model


# ---------------------------------------------------------------------
# novel-bigram identification
# ---------------------------------------------------------------------
def bigram_set(inscriptions):
    return {(a, b) for ins in inscriptions for a, b in zip(ins, ins[1:])}


def train_sign_counts(train_ins):
    c = defaultdict(int)
    for ins in train_ins:
        for s in ins:
            c[s] += 1
    return c


@torch.no_grad()
def predict_probs(model, sign_to_idx, pairs):
    """pairs: list of (a, b). Returns array of P(b|a) for each, via a
    single batched forward pass (n=1 context)."""
    ctx = torch.tensor([[sign_to_idx[a]] for a, b in pairs], dtype=torch.long)
    logits = model(ctx)
    probs = torch.softmax(logits, dim=-1)
    targets = torch.tensor([sign_to_idx[b] for a, b in pairs], dtype=torch.long)
    return probs[torch.arange(len(pairs)), targets].numpy()


# ---------------------------------------------------------------------
# training-only cluster transition matrix (no leakage from val)
# ---------------------------------------------------------------------
def classified_sign_sequences(inscriptions, sign_to_cluster):
    seqs = []
    for ins in inscriptions:
        seq = [s for s in ins if s in sign_to_cluster]
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


def cluster_transition_matrix(train_ins, sign_to_cluster, k):
    seqs = classified_sign_sequences(train_ins, sign_to_cluster)
    counts = np.zeros((k, k))
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            counts[sign_to_cluster[a], sign_to_cluster[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / np.where(row_sums == 0, 1, row_sums)


def spearman_corr(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_null_corr(kept, labels, train_ins, k, novel_pairs, model_preds,
                           n_perm=N_PERM, seed=2):
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    corrs = np.zeros(n_perm)
    for p in range(n_perm):
        perm_labels = rng.permutation(labels_arr)
        sign_to_cluster_p = dict(zip(kept, perm_labels))
        trans_p = cluster_transition_matrix(train_ins, sign_to_cluster_p, k)
        compat_p = np.array([
            trans_p[sign_to_cluster_p[a], sign_to_cluster_p[b]]
            for a, b in novel_pairs
        ])
        corrs[p] = spearman_corr(compat_p, model_preds)
    return corrs


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------
def plot_results(compat, preds, observed_corr, null_corrs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    quartiles = np.quantile(compat, [0, 0.25, 0.5, 0.75, 1.0])
    bin_idx = np.clip(np.digitize(compat, quartiles[1:-1]), 0, 3)
    means = [preds[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
             for b in range(4)]
    sems = [preds[bin_idx == b].std() / max(1, np.sqrt((bin_idx == b).sum()))
            if (bin_idx == b).any() else 0 for b in range(4)]
    ax.bar(range(4), means, yerr=sems, capsize=4, color="steelblue")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"])
    ax.set_xlabel("cluster compatibility quartile (training-derived)")
    ax.set_ylabel("mean model-predicted P(b|a) for novel pairs")
    ax.set_title("Predicted probability by cluster compatibility")

    ax = axes[1]
    ax.hist(null_corrs, bins=30, color="lightgray", edgecolor="k", label="permutation null")
    ax.axvline(observed_corr, color="red", linewidth=2, label=f"observed r={observed_corr:.3f}")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("count (500 permutations)")
    ax.set_title("Observed correlation vs permutation null")
    ax.legend()

    plt.tight_layout()
    plt.savefig("nplm_generalization_test.png", dpi=150)
    plt.close(fig)
    print("\n  saved nplm_generalization_test.png")


def main():
    inscriptions = load_inscriptions()
    meta = sign_metadata(inscriptions)
    signs_sorted, sign_to_idx, bos_idx = build_vocab(inscriptions)
    vocab_size = len(signs_sorted) + 1
    out_size = len(signs_sorted)
    configs = load_manifest()

    train_ins, val_ins = split_inscriptions(inscriptions)
    print(f"{len(train_ins)} train inscriptions, {len(val_ins)} val "
          f"(split_seed={SPLIT_SEED}, matches indus_nplm_train.py exactly)")

    kept, labels, k, model = reproduce_clustering(
        inscriptions, meta, signs_sorted, sign_to_idx, bos_idx,
        vocab_size, out_size, configs
    )
    sign_to_cluster = dict(zip(kept, labels))
    print(f"Reproduced {TARGET_CONFIG}/{TARGET_KIND} seed {TARGET_SEED}: "
          f"k={k}, {len(kept)} classified signs")

    # ---- Part 1: novel bigrams, unrestricted vocabulary ----
    train_bigrams = bigram_set(train_ins)
    val_bigrams = bigram_set(val_ins)
    novel_all = sorted(val_bigrams - train_bigrams)
    train_counts = train_sign_counts(train_ins)
    total_train_tokens = sum(train_counts.values())

    # exclude pairs involving a sign with zero training occurrences --
    # that sign's embedding never received a gradient update, so its
    # predicted probability reflects random init, not learned smoothing
    novel_all = [(a, b) for a, b in novel_all
                 if train_counts.get(a, 0) > 0 and train_counts.get(b, 0) > 0]

    print(f"\n{len(train_bigrams)} distinct training bigrams, "
          f"{len(val_bigrams)} distinct validation bigrams, "
          f"{len(novel_all)} novel (never seen in training, both signs "
          f"otherwise trained)")

    preds_all = predict_probs(model, sign_to_idx, novel_all)
    chance = 1.0 / len(signs_sorted)
    unigram_probs = np.array([train_counts[b] / total_train_tokens for a, b in novel_all])

    print(f"\n--- Part 1: does the model beat naive baselines on truly novel pairs? ---")
    print(f"  chance (1/V)                    = {chance:.5f}")
    print(f"  mean unigram P(target)          = {unigram_probs.mean():.5f}")
    print(f"  mean model-predicted P(b|a)     = {preds_all.mean():.5f}  "
          f"({preds_all.mean()/chance:.1f}x chance, "
          f"{preds_all.mean()/unigram_probs.mean():.2f}x unigram fallback)")
    print(f"  median model-predicted P(b|a)   = {np.median(preds_all):.5f}")

    order = np.argsort(-preds_all)
    print(f"\n  highest-probability novel pairs:")
    for idx in order[:5]:
        a, b = novel_all[idx]
        print(f"    {a} -> {b}: P={preds_all[idx]:.4f}  "
              f"(cluster {sign_to_cluster.get(a, '?')} -> {sign_to_cluster.get(b, '?')})")
    print(f"  lowest-probability novel pairs:")
    for idx in order[-5:]:
        a, b = novel_all[idx]
        print(f"    {a} -> {b}: P={preds_all[idx]:.6f}  "
              f"(cluster {sign_to_cluster.get(a, '?')} -> {sign_to_cluster.get(b, '?')})")

    # ---- Part 2: cluster-compatibility correlation, classified subset ----
    novel_classified = [(a, b) for a, b in novel_all
                        if a in sign_to_cluster and b in sign_to_cluster]
    print(f"\n--- Part 2: does predicted probability track cluster "
          f"compatibility? ({len(novel_classified)} novel pairs with both "
          f"signs classified) ---")

    if len(novel_classified) < 15:
        print("  too few novel classified pairs for a meaningful correlation "
              "test -- consider a less strict MIN_COUNT_CLUSTER, or treat "
              "this as descriptive only")
        return

    preds_c = predict_probs(model, sign_to_idx, novel_classified)
    train_trans = cluster_transition_matrix(train_ins, sign_to_cluster, k)
    compat = np.array([
        train_trans[sign_to_cluster[a], sign_to_cluster[b]] for a, b in novel_classified
    ])

    observed_corr = spearman_corr(compat, preds_c)
    null_corrs = permutation_null_corr(kept, labels, train_ins, k,
                                        novel_classified, preds_c)
    z = (observed_corr - null_corrs.mean()) / (null_corrs.std() + 1e-9)
    percentile = float((null_corrs < observed_corr).mean() * 100)

    print(f"  Spearman r(cluster compatibility, predicted probability) "
          f"= {observed_corr:+.3f}")
    print(f"  permutation null: mean={null_corrs.mean():+.3f} "
          f"std={null_corrs.std():.3f}  z={z:+.2f}  "
          f"(observed exceeds {percentile:.1f}% of {N_PERM} shuffles)")
    if abs(z) > 3:
        direction = "tracks" if observed_corr > 0 else "inversely tracks"
        print(f"  --> significant: predicted probability for genuinely novel "
              f"pairs {direction} cluster compatibility beyond chance")
    else:
        print(f"  --> not significant at |z|>3: no detectable relationship "
              f"between cluster compatibility and predicted probability for "
              f"novel pairs at this sample size")

    plot_results(compat, preds_c, observed_corr, null_corrs)


if __name__ == "__main__":
    main()
