# =====================================================================
# indus_nplm_generalization_allseeds.py
#
# Seed-robustness check on indus_nplm_generalization_test.py's single-
# seed result (r=+0.238, z=1.86 -- borderline). Runs both parts of that
# test across all 10 n1_nodirect seeds independently: each seed gets
# its OWN clustering (recomputed from that seed's own softmax
# representations, not reused from seed 0 -- a different seed's model
# can and does cluster signs somewhat differently) and its own
# predicted probabilities. The set of novel bigrams itself is fixed
# (determined only by the train/val split, which is the same for every
# seed), so what varies across rows below is purely the model.
#
# Reports, per seed: ratio of mean predicted probability to the
# unigram-fallback baseline (Part 1), and the cluster-compatibility
# correlation + permutation z-score (Part 2) -- then aggregates both
# across seeds with a sign test (how many of 10 seeds are positive,
# binomial p-value against 50/50) and mean+/-std, matching this
# project's standing discipline of never trusting a single seed.
#
# Requires: pip install torch scikit-learn numpy matplotlib
# Run in the same folder as induscorpus.txt and nplm_models/:
#   python indus_nplm_generalization_allseeds.py
# =====================================================================

import csv
import math
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
SEEDS = list(range(10))


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
    count, texts = defaultdict(int), defaultdict(int)
    for ins in inscriptions:
        for s in ins:
            count[s] += 1
        for s in set(ins):
            texts[s] += 1
    return {s: dict(count=count[s], texts=texts[s]) for s in count}


# ---------------------------------------------------------------------
# model definition + clustering (must match earlier scripts)
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
    logits, _ = model(ctx_idx, return_hidden=True)
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


def reproduce_clustering_for_seed(seed, inscriptions, meta, signs_sorted,
                                   sign_to_idx, bos_idx, vocab_size, out_size,
                                   configs):
    row = next(r for r in configs[TARGET_CONFIG] if int(r["seed"]) == seed)
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
# novel bigrams (fixed across seeds -- depends only on the split)
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
    ctx = torch.tensor([[sign_to_idx[a]] for a, b in pairs], dtype=torch.long)
    logits = model(ctx)
    probs = torch.softmax(logits, dim=-1)
    targets = torch.tensor([sign_to_idx[b] for a, b in pairs], dtype=torch.long)
    return probs[torch.arange(len(pairs)), targets].numpy()


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
        s2c_p = dict(zip(kept, perm_labels))
        trans_p = cluster_transition_matrix(train_ins, s2c_p, k)
        compat_p = np.array([trans_p[s2c_p[a], s2c_p[b]] for a, b in novel_pairs])
        corrs[p] = spearman_corr(compat_p, model_preds)
    return corrs


def binomial_sign_test(n_positive, n_total):
    """One-tailed p-value: P(>= n_positive successes out of n_total,
    p=0.5), i.e. is 'more positive than expected by chance'."""
    return sum(math.comb(n_total, k) for k in range(n_positive, n_total + 1)) / (2 ** n_total)


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------
def plot_seed_summary(results):
    seeds = [r["seed"] for r in results]
    ratios = [r["ratio_vs_unigram"] for r in results]
    corrs = [r["corr"] for r in results]
    z_scores = [r["z"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.bar(seeds, ratios, color="steelblue")
    ax.axhline(1.0, color="red", linestyle="--", label="unigram parity")
    ax.set_xlabel("seed")
    ax.set_ylabel("mean predicted P / mean unigram P")
    ax.set_title("Novel-pair probability vs unigram fallback, per seed")
    ax.set_xticks(seeds)
    ax.legend()

    ax = axes[1]
    colors = ["steelblue" if z > 0 else "indianred" for z in z_scores]
    ax.bar(seeds, corrs, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("seed")
    ax.set_ylabel("Spearman r (cluster compatibility vs predicted P)")
    ax.set_title("Cluster-compatibility correlation, per seed\n"
                  "(blue = positive z, red = negative z)")
    ax.set_xticks(seeds)

    plt.tight_layout()
    plt.savefig("nplm_generalization_allseeds.png", dpi=150)
    plt.close(fig)
    print("\n  saved nplm_generalization_allseeds.png")


def main():
    inscriptions = load_inscriptions()
    meta = sign_metadata(inscriptions)
    signs_sorted, sign_to_idx, bos_idx = build_vocab(inscriptions)
    vocab_size = len(signs_sorted) + 1
    out_size = len(signs_sorted)
    configs = load_manifest()

    train_ins, val_ins = split_inscriptions(inscriptions)
    train_bigrams = bigram_set(train_ins)
    val_bigrams = bigram_set(val_ins)
    train_counts = train_sign_counts(train_ins)
    total_train_tokens = sum(train_counts.values())

    novel_all = sorted(val_bigrams - train_bigrams)
    novel_all = [(a, b) for a, b in novel_all
                 if train_counts.get(a, 0) > 0 and train_counts.get(b, 0) > 0]
    unigram_probs = np.array([train_counts[b] / total_train_tokens for a, b in novel_all])
    print(f"{len(train_ins)} train / {len(val_ins)} val inscriptions, "
          f"{len(novel_all)} novel bigrams (fixed across all seeds)\n")

    results = []
    print(f"{'seed':<5}{'k':<4}{'n_cls':<7}{'n_novel_cls':<13}"
          f"{'ratio_vs_uni':<14}{'corr':<9}{'z':<8}{'percentile':<11}")
    print("-" * 70)

    for seed in SEEDS:
        kept, labels, k, model = reproduce_clustering_for_seed(
            seed, inscriptions, meta, signs_sorted, sign_to_idx, bos_idx,
            vocab_size, out_size, configs
        )
        sign_to_cluster = dict(zip(kept, labels))

        preds_all = predict_probs(model, sign_to_idx, novel_all)
        ratio_vs_unigram = float(preds_all.mean() / unigram_probs.mean())

        novel_classified = [(a, b) for a, b in novel_all
                            if a in sign_to_cluster and b in sign_to_cluster]
        if len(novel_classified) < 15:
            print(f"{seed:<5}{k:<4}{len(kept):<7}{len(novel_classified):<13}"
                  f"{ratio_vs_unigram:<14.3f}{'--':<9}{'--':<8}{'too few':<11}")
            results.append(dict(seed=seed, ratio_vs_unigram=ratio_vs_unigram,
                                 corr=np.nan, z=np.nan))
            continue

        preds_c = predict_probs(model, sign_to_idx, novel_classified)
        train_trans = cluster_transition_matrix(train_ins, sign_to_cluster, k)
        compat = np.array([
            train_trans[sign_to_cluster[a], sign_to_cluster[b]]
            for a, b in novel_classified
        ])
        corr = spearman_corr(compat, preds_c)
        null_corrs = permutation_null_corr(kept, labels, train_ins, k,
                                            novel_classified, preds_c,
                                            seed=seed + 100)
        z = (corr - null_corrs.mean()) / (null_corrs.std() + 1e-9)
        percentile = float((null_corrs < corr).mean() * 100)

        print(f"{seed:<5}{k:<4}{len(kept):<7}{len(novel_classified):<13}"
              f"{ratio_vs_unigram:<14.3f}{corr:<9.3f}{z:<8.2f}{percentile:<11.1f}")
        results.append(dict(seed=seed, ratio_vs_unigram=ratio_vs_unigram,
                             corr=corr, z=z))

    valid = [r for r in results if not np.isnan(r["corr"])]
    ratios = np.array([r["ratio_vs_unigram"] for r in results])
    corrs = np.array([r["corr"] for r in valid])
    n_positive = int((corrs > 0).sum())
    sign_p = binomial_sign_test(n_positive, len(corrs))

    print("\n" + "=" * 70)
    print("AGGREGATE ACROSS SEEDS")
    print("=" * 70)
    print(f"ratio vs unigram fallback: mean={ratios.mean():.3f} "
          f"std={ratios.std():.3f}  "
          f"({(ratios > 1).sum()}/{len(ratios)} seeds beat unigram)")
    print(f"cluster-compatibility correlation: mean r={corrs.mean():+.3f} "
          f"std={corrs.std():.3f}")
    print(f"{n_positive}/{len(corrs)} seeds positive  "
          f"(sign-test p={sign_p:.3f} against 50/50 by chance)")
    if sign_p < 0.05 and corrs.mean() > 0:
        print("--> consistently positive across seeds: the single-seed "
              "result replicates")
    elif sign_p < 0.05 and corrs.mean() < 0:
        print("--> consistently NEGATIVE across seeds: opposite direction "
              "to the single-seed result -- investigate before reporting "
              "either")
    else:
        print("--> not consistent across seeds: the single-seed r=+0.238 "
              "does not reliably replicate; treat as noise, not a finding")

    plot_seed_summary(results)


if __name__ == "__main__":
    main()
