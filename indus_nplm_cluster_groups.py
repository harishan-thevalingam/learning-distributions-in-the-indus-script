# =====================================================================
# indus_nplm_cluster_groups.py
#
# Takes the best clustering found so far (n1_nodirect, softmax
# representation, k=9, ARI=1.0000 vs the terminal/opener ground truth)
# and asks two new questions the earlier battery didn't:
#
#   1. What actually IS in each of the 9 groups, and how do they
#      profile beyond the binary terminal/opener split -- in
#      particular, a new "mean relative position within the
#      inscription" metric (0=start, 1=end) that can reveal genuinely
#      medial groups the end_rate/start_rate split can't see.
#
#   2. Do the groups themselves obey sequencing rules -- i.e. is
#      cluster-to-cluster transition structured (a "grammar" in the
#      loose sense: group A reliably precedes group B), or is it just
#      9 independently-positioned groups with no relationship to each
#      other? Tested with a permutation test (500 shuffles of the
#      sign->cluster mapping, same style as this project's earlier
#      site-level permutation tests): for each of the k*k transition
#      cells, compare the observed count against the null distribution
#      from reshuffling which cluster label each sign gets (corpus
#      structure held fixed).
#
# METHODOLOGICAL CHOICE, stated explicitly: transitions are counted
# between consecutive CLASSIFIED signs (count>=15 AND reliable softmax
# representation), compressing over any unclassified/rare signs that
# sit between them in the actual text -- i.e. this asks "does group A
# tend to precede group B among the signs we can classify" rather than
# "are A and B literally adjacent in the text". The literal-adjacency
# version would be far sparser (most of the 377 signs aren't in any
# cluster) and answers a stricter but different question -- worth
# rerunning if the compressed version turns up something interesting.
#
# MULTIPLE COMPARISONS: k=9 gives 81 transition cells. |z|>3 is used
# as the reporting threshold below; with 81 uncorrected tests at that
# threshold, expect ~0.2-0.3 false positives by chance, so treat
# borderline z~3-3.5 hits as suggestive, not confirmed, and prioritise
# whichever transitions clear z>4.
#
# v2 ADDITIONS -- two follow-up tests, because the naive transition
# test above is close to circular: at n=1 the softmax representation
# is essentially "predicted distribution over the next sign given this
# sign", so clustering on it mostly groups signs by WHERE they sit in
# the sequence. A transition test on those clusters will show
# position-correlated clusters transitioning in position-correlated
# ways almost by construction -- not new information.
#
#   (a) POSITION-CONTROLLED permutation test: bins each classified
#       sign into a position tercile (by its own mean relative
#       position) and permutes cluster labels ONLY WITHIN each tercile
#       -- so a sign can only swap identities with another sign at
#       roughly the same point in inscriptions. Whatever survives this
#       is a real sequential dependency between groups, independent of
#       position; whatever drops out was position all along.
#
#   (b) TRANSITION DECOMPOSITION: for each flagged transition, breaks
#       it down into the underlying member-level sign-pair bigram
#       counts and reports what fraction comes from the single most
#       common pair. A rule spread across many member pairs is a real
#       class-level regularity; a rule that's mostly one pair is a
#       lexical idiosyncrasy (e.g. a specific formulaic bigram already
#       on record) that happens to fall inside two loosely-drawn
#       clusters, not a productive grammatical rule.
#
# Requires: pip install torch scikit-learn numpy matplotlib
# Run in the same folder as induscorpus.txt and nplm_models/:
#   python indus_nplm_cluster_groups.py
# =====================================================================

import csv
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
N_PERM = 500
N_POSITION_BINS = 3
Z_REPORT_THRESHOLD = 3.0

# Which model to reproduce -- change these to inspect a different
# (config, representation) combo.
TARGET_CONFIG = "n1_nodirect"
TARGET_KIND = "softmax"
TARGET_SEED = 0


# ---------------------------------------------------------------------
# corpus + vocab (must match indus_nplm_train.py / indus_nplm_analyse.py)
# ---------------------------------------------------------------------
def load_inscriptions(path=CORPUS_PATH):
    inscriptions = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                inscriptions.append(line.split())
    return inscriptions


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
        meta[s] = dict(
            count=count[s], texts=t,
            end_rate=ends[s] / t if t else 0.0,
            start_rate=starts[s] / t if t else 0.0,
        )
    return meta


def compute_relative_positions(inscriptions):
    """Mean fractional position (0=start, 1=end) of each sign across
    all its occurrences -- a continuous alternative to the binary
    start_rate/end_rate split, useful for spotting medial groups."""
    positions = defaultdict(list)
    for ins in inscriptions:
        L = len(ins)
        for i, s in enumerate(ins):
            rel = i / (L - 1) if L > 1 else 0.5
            positions[s].append(rel)
    return {s: float(np.mean(v)) for s, v in positions.items()}


def build_vocab(inscriptions):
    signs_sorted = sorted({s for ins in inscriptions for s in ins}, key=int)
    sign_to_idx = {s: i for i, s in enumerate(signs_sorted)}
    bos_idx = len(signs_sorted)
    return signs_sorted, sign_to_idx, bos_idx


# ---------------------------------------------------------------------
# model definition (must match indus_nplm_train.py exactly)
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

    embedding, softmax_rep, hidden_rep = {}, {}, {}
    for i, s in enumerate(signs_sorted):
        embedding[s] = model.C.weight[i].detach().numpy()
        mask = last_token == i
        if int(mask.sum()) < min_occ:
            continue
        softmax_rep[s] = probs[mask].mean(dim=0).numpy()
        hidden_rep[s] = hidden[mask].mean(dim=0).numpy()
    return embedding, softmax_rep, hidden_rep


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


# ---------------------------------------------------------------------
# cluster profiling
# ---------------------------------------------------------------------
def describe_clusters(kept, labels, meta, rel_pos):
    print("\n" + "=" * 70)
    print(f"CLUSTER MEMBERSHIP ({TARGET_CONFIG}, {TARGET_KIND}, seed {TARGET_SEED})")
    print("=" * 70)
    profiles = {}
    for lab in sorted(set(labels)):
        members = sorted([s for s, l in zip(kept, labels) if l == lab], key=int)
        ek = np.mean([meta[s]["end_rate"] for s in members])
        sk = np.mean([meta[s]["start_rate"] for s in members])
        ck = np.mean([meta[s]["count"] for s in members])
        rp = np.mean([rel_pos.get(s, np.nan) for s in members])
        profiles[lab] = dict(members=members, end_rate=ek, start_rate=sk,
                              count=ck, rel_pos=rp)
        print(f"\nCluster {lab} (n={len(members)}): {members}")
        print(f"    mean end_rate={ek:.3f}  mean start_rate={sk:.3f}  "
              f"mean count={ck:.1f}  mean relative position={rp:.3f}")
    return profiles


def plot_cluster_profile(profiles):
    labs = sorted(profiles)
    x = np.arange(len(labs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, [profiles[l]["end_rate"] for l in labs], width,
           label="mean end_rate")
    ax.bar(x, [profiles[l]["start_rate"] for l in labs], width,
           label="mean start_rate")
    ax.bar(x + width, [profiles[l]["rel_pos"] for l in labs], width,
           label="mean relative position")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{l}\n(n={len(profiles[l]['members'])})" for l in labs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate / fractional position")
    ax.set_title(f"Cluster role profile ({TARGET_CONFIG}, {TARGET_KIND})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("nplm_cluster_profile.png", dpi=150)
    plt.close(fig)
    print("\n  saved nplm_cluster_profile.png")


# ---------------------------------------------------------------------
# class-level "grammar" test
# ---------------------------------------------------------------------
def class_sequences(inscriptions, sign_to_cluster):
    """Per-inscription sequence of cluster labels, compressing over any
    sign not in sign_to_cluster (see methodology note in the module
    docstring)."""
    seqs = []
    for ins in inscriptions:
        seq = [sign_to_cluster[s] for s in ins if s in sign_to_cluster]
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


def transition_counts(seqs, k):
    counts = np.zeros((k, k))
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            counts[a, b] += 1
    return counts


def permutation_null(kept, labels, inscriptions, k, n_perm=N_PERM, seed=0):
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    null_counts = np.zeros((n_perm, k, k))
    for p in range(n_perm):
        perm_labels = rng.permutation(labels_arr)
        sign_to_cluster = dict(zip(kept, perm_labels))
        seqs = class_sequences(inscriptions, sign_to_cluster)
        null_counts[p] = transition_counts(seqs, k)
    return null_counts


def assign_position_bins(kept, rel_pos, n_bins=N_POSITION_BINS):
    """Tercile-bin each classified sign by its own mean relative
    position. Returns dict sign -> bin index (0=earliest tercile)."""
    values = np.array([rel_pos.get(s, 0.5) for s in kept])
    cuts = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(values, cuts)
    return dict(zip(kept, bins))


def stratified_permutation_null(kept, labels, bin_of, inscriptions, k,
                                 n_perm=N_PERM, seed=1):
    """Same permutation test as permutation_null, but each shuffle only
    swaps cluster labels among signs in the SAME position tercile --
    isolating structure that isn't explained by position alone."""
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    bins_arr = np.array([bin_of[s] for s in kept])
    null_counts = np.zeros((n_perm, k, k))
    for p in range(n_perm):
        perm_labels = labels_arr.copy()
        for b in np.unique(bins_arr):
            idx = np.where(bins_arr == b)[0]
            perm_labels[idx] = rng.permutation(labels_arr[idx])
        sign_to_cluster = dict(zip(kept, perm_labels))
        seqs = class_sequences(inscriptions, sign_to_cluster)
        null_counts[p] = transition_counts(seqs, k)
    return null_counts


def flagged_transitions(z, k, threshold=Z_REPORT_THRESHOLD):
    hits = [(i, j, z[i, j]) for i in range(k) for j in range(k)
            if abs(z[i, j]) > threshold]
    hits.sort(key=lambda t: -abs(t[2]))
    return hits


def classified_sign_sequences(inscriptions, sign_to_cluster):
    """Same compression rule as class_sequences, but keeps sign
    identity (not just cluster label) -- needed to decompose a
    cluster-to-cluster transition into its member-level sign pairs."""
    seqs = []
    for ins in inscriptions:
        seq = [s for s in ins if s in sign_to_cluster]
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


def pairwise_bigram_counts(sign_seqs):
    counts = defaultdict(int)
    for seq in sign_seqs:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def decompose_transition(pair_counts, members_i, members_j, top_n=3):
    """For a flagged cluster i -> cluster j transition: how many
    distinct member-level sign pairs contribute, and what fraction of
    the total comes from the single most common pair. High
    concentration in one pair = lexical idiosyncrasy, not a class-wide
    rule."""
    sub = [((a, b), c) for (a, b), c in pair_counts.items()
           if a in members_i and b in members_j]
    sub.sort(key=lambda t: -t[1])
    total = sum(c for _, c in sub)
    top1_frac = sub[0][1] / total if sub and total else 0.0
    top3_frac = sum(c for _, c in sub[:3]) / total if total else 0.0
    return dict(total=total, n_distinct_pairs=len(sub), top1_frac=top1_frac,
                top3_frac=top3_frac, top_pairs=sub[:top_n])


def plot_transition_zscores(z, k, title_suffix, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-6, vmax=6)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xlabel("to cluster"); ax.set_ylabel("from cluster")
    ax.set_title(f"Class-level transition z-scores ({TARGET_CONFIG}, {TARGET_KIND})\n"
                 f"{title_suffix}")
    for i in range(k):
        for j in range(k):
            if abs(z[i, j]) > 2:
                ax.text(j, i, f"{z[i, j]:.1f}", ha="center", va="center",
                         fontsize=7, color="white" if abs(z[i, j]) > 3.5 else "black")
    plt.colorbar(im, ax=ax, label="z-score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    inscriptions = load_inscriptions()
    meta = sign_metadata(inscriptions)
    rel_pos = compute_relative_positions(inscriptions)
    signs_sorted, sign_to_idx, bos_idx = build_vocab(inscriptions)
    vocab_size = len(signs_sorted) + 1
    out_size = len(signs_sorted)

    configs = load_manifest()
    row = next(r for r in configs[TARGET_CONFIG] if int(r["seed"]) == TARGET_SEED)
    n = int(row["n"])
    model = load_model(row, vocab_size, out_size)

    emb, softmax_rep, hidden_rep = sign_representations(
        model, inscriptions, signs_sorted, sign_to_idx, bos_idx, n
    )
    rep_dict = {"embedding": emb, "hidden": hidden_rep, "softmax": softmax_rep}[TARGET_KIND]

    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_CLUSTER], key=int
    )
    kept, X = dict_to_matrix(rep_dict, kept_master)
    print(f"Reproducing {TARGET_CONFIG}/{TARGET_KIND} seed {TARGET_SEED}: "
          f"{len(kept)} classifiable signs")

    k, score, labels = best_kmeans(X)
    print(f"k={k} (silhouette {score:.4f})")

    profiles = describe_clusters(kept, labels, meta, rel_pos)
    plot_cluster_profile(profiles)

    print("\n" + "=" * 70)
    print(f"CLASS-LEVEL TRANSITION TEST ({N_PERM}-permutation null)")
    print("=" * 70)
    sign_to_cluster = dict(zip(kept, labels))
    seqs = class_sequences(inscriptions, sign_to_cluster)
    print(f"{len(seqs)} inscriptions with >=2 classifiable signs, "
          f"{sum(len(s) - 1 for s in seqs)} total transitions\n")

    obs = transition_counts(seqs, k)

    # --- naive test (position-confounded) ---
    null = permutation_null(kept, labels, inscriptions, k)
    null_mean = null.mean(axis=0)
    null_std = null.std(axis=0) + 1e-9
    z_naive = (obs - null_mean) / null_std
    hits_naive = flagged_transitions(z_naive, k)

    print("--- naive test (free permutation -- position-confounded) ---")
    print(f"Significant transitions (|z| > {Z_REPORT_THRESHOLD}), sorted by |z|:")
    if not hits_naive:
        print("  none")
    for i, j, zval in hits_naive:
        print(f"  cluster {i} -> cluster {j}: observed={obs[i, j]:.0f}  "
              f"expected={null_mean[i, j]:.1f}+/-{null_std[i, j]:.1f}  "
              f"z={zval:+.2f}")
    plot_transition_zscores(
        z_naive, k, f"permutation null, {N_PERM} shuffles; |z|>3 flagged below",
        "nplm_class_transitions.png"
    )

    # --- position-controlled test ---
    bin_of = assign_position_bins(kept, rel_pos)
    print(f"\nPosition tercile composition (bin 0=earliest, {N_POSITION_BINS - 1}=latest):")
    for b in range(N_POSITION_BINS):
        members_in_bin = [s for s in kept if bin_of[s] == b]
        clusters_in_bin = sorted({sign_to_cluster[s] for s in members_in_bin})
        print(f"  bin {b}: {len(members_in_bin)} signs, spanning clusters "
              f"{clusters_in_bin}")

    null_strat = stratified_permutation_null(kept, labels, bin_of, inscriptions, k)
    null_strat_mean = null_strat.mean(axis=0)
    null_strat_std = null_strat.std(axis=0) + 1e-9
    z_strat = (obs - null_strat_mean) / null_strat_std
    hits_strat = flagged_transitions(z_strat, k)

    print(f"\n--- position-controlled test (within-tercile permutation) ---")
    print(f"Significant transitions (|z| > {Z_REPORT_THRESHOLD}), sorted by |z|:")
    if not hits_strat:
        print("  none -- everything in the naive test was explained by position")
    for i, j, zval in hits_strat:
        print(f"  cluster {i} -> cluster {j}: observed={obs[i, j]:.0f}  "
              f"expected={null_strat_mean[i, j]:.1f}+/-{null_strat_std[i, j]:.1f}  "
              f"z={zval:+.2f}")
    plot_transition_zscores(
        z_strat, k,
        f"POSITION-CONTROLLED (within-tercile permutation, {N_PERM} shuffles)",
        "nplm_class_transitions_position_controlled.png"
    )

    naive_set = {(i, j) for i, j, _ in hits_naive}
    strat_set = {(i, j) for i, j, _ in hits_strat}
    print(f"\nComparison: {len(naive_set)} naive hits, {len(strat_set)} survive "
          f"position control.")
    dropped = naive_set - strat_set
    survived = naive_set & strat_set
    if dropped:
        print(f"  explained by position alone (dropped): {sorted(dropped)}")
    if survived:
        print(f"  genuine class-level structure (survived): {sorted(survived)}")

    # --- decompose every flagged transition (either test) into member
    #     sign-pairs, to check for single-bigram idiosyncrasies ---
    all_flagged = sorted(naive_set | strat_set)
    if all_flagged:
        print("\n" + "=" * 70)
        print("TRANSITION DECOMPOSITION (is this a class rule or one loud bigram?)")
        print("=" * 70)
        sign_seqs = classified_sign_sequences(inscriptions, sign_to_cluster)
        pair_counts = pairwise_bigram_counts(sign_seqs)
        members_by_cluster = {
            lab: [s for s, l in zip(kept, labels) if l == lab]
            for lab in range(k)
        }
        for i, j in all_flagged:
            d = decompose_transition(pair_counts, members_by_cluster[i],
                                      members_by_cluster[j])
            tag = "survives position control" if (i, j) in strat_set else \
                  "position-explained" if (i, j) in naive_set else ""
            print(f"\n  cluster {i} -> cluster {j} ({tag}):")
            print(f"    {d['total']} transitions across {d['n_distinct_pairs']} "
                  f"distinct member sign-pairs")
            print(f"    top pair {d['top_pairs'][0][0]}: "
                  f"{d['top1_frac']*100:.0f}% of the total"
                  f"{'  <-- mostly one bigram, not a class rule' if d['top1_frac'] > 0.5 else ''}")
            print(f"    top 3 pairs: {d['top3_frac']*100:.0f}% of the total")
            print(f"    top pairs: {d['top_pairs']}")


if __name__ == "__main__":
    main()
