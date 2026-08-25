# =====================================================================
# indus_nplm_hub_analysis.py
#
# Systematic version of the "sign 59" finding from
# indus_nplm_cluster_groups.py's transition decomposition: 59 turned
# up by accident as a convergence point (three different signs from
# cluster 6 all precede it). This script scans EVERY classified sign
# for the same property directly, in both directions:
#
#   IN-HUB  : a sign with an unusually wide variety of distinct
#             predecessors for how often it occurs -- a convergence
#             point ("many different things can come before this").
#   OUT-HUB : the mirror case -- a sign followed by an unusually wide
#             variety of distinct successors.
#   RESTRICTED (either direction): the opposite -- a sign with far
#             FEWER distinct neighbours than its frequency predicts,
#             i.e. locked into one or two fixed partners (the mirror
#             of the 267->99 bigram that dominated cluster 8->4).
#
# THE CONFOUND THIS CONTROLS FOR: a common sign will accumulate more
# distinct predecessors than a rare one just by having more chances to
# see them, with no special structural property at all. So raw
# in-degree/out-degree isn't the right statistic. Instead this uses a
# species-accumulation-style null (same family of idea as a rarefaction
# curve in ecology): for a sign with C actual predecessor-slots,
# E[distinct predecessors] = sum_i [1 - (1-p_i)^C], where p_i is sign
# i's share of the corpus-wide predecessor pool (i.e. what you'd expect
# if this sign's C predecessor-slots were filled by C independent draws
# from the same distribution every other position in the corpus draws
# from). EXCESS = observed - expected is the statistic that actually
# answers "is this sign unusually diverse", controlling for frequency.
#
# Uses LITERAL corpus adjacency (not the compressed classified-sign-
# sequence used for the class-transition test) -- this is about a
# specific sign's real local neighbourhood, so the true text order is
# what matters, not just its neighbours among the 87 classified signs.
# Cluster-diversity of predecessors/successors is still restricted to
# the classified subset (that's the only vocabulary with cluster
# labels), and noted as a fraction of total predecessor mass.
#
# Requires: pip install torch scikit-learn numpy matplotlib
# Run in the same folder as induscorpus.txt and nplm_models/:
#   python indus_nplm_hub_analysis.py
# =====================================================================

import csv
from collections import defaultdict, Counter

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
TOP_N_REPORT = 10

TARGET_CONFIG = "n1_nodirect"
TARGET_KIND = "softmax"
TARGET_SEED = 0


# ---------------------------------------------------------------------
# corpus + vocab (must match the earlier NPLM scripts exactly)
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


def build_vocab(inscriptions):
    signs_sorted = sorted({s for ins in inscriptions for s in ins}, key=int)
    sign_to_idx = {s: i for i, s in enumerate(signs_sorted)}
    bos_idx = len(signs_sorted)
    return signs_sorted, sign_to_idx, bos_idx


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


def reproduce_clustering(inscriptions, meta):
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
    k, score, labels = best_kmeans(X)
    return kept, labels, k


# ---------------------------------------------------------------------
# literal corpus adjacency + hub/restricted analysis
# ---------------------------------------------------------------------
def build_neighbour_counts(inscriptions):
    """pred_counts[t] = Counter of signs that literally precede sign t
    somewhere in the corpus; succ_counts[t] = mirror for what follows.
    Also returns the corpus-wide predecessor-slot and successor-slot
    pools (used as the null-model sampling distributions)."""
    pred_counts = defaultdict(Counter)
    succ_counts = defaultdict(Counter)
    pred_pool = Counter()   # every sign's occurrences as *someone's* predecessor
    succ_pool = Counter()   # every sign's occurrences as *someone's* successor
    for ins in inscriptions:
        for a, b in zip(ins, ins[1:]):
            pred_counts[b][a] += 1
            succ_counts[a][b] += 1
            pred_pool[a] += 1
            succ_pool[b] += 1
    return pred_counts, succ_counts, pred_pool, succ_pool


def expected_distinct(C, pool_counter):
    """E[distinct categories] for C i.i.d. draws from pool_counter's
    empirical distribution -- the species-accumulation-curve formula."""
    total = sum(pool_counter.values())
    probs = np.array([c / total for c in pool_counter.values()])
    return float(np.sum(1 - (1 - probs) ** C))


def analyse_direction(kept, target_signs, neighbour_counts, pool, sign_to_cluster):
    """Returns a dict: sign -> stats, for one direction (predecessor or
    successor). neighbour_counts is pred_counts or succ_counts;
    pool is the matching pred_pool/succ_pool."""
    out = {}
    for s in target_signs:
        nb = neighbour_counts.get(s)
        if not nb:
            continue
        total = sum(nb.values())
        degree = len(nb)
        ps = np.array(list(nb.values()), dtype=float)
        ps = ps / ps.sum()
        entropy = float(-(ps * np.log2(ps)).sum())
        classified_neighbours = {n_: c for n_, c in nb.items() if n_ in sign_to_cluster}
        classified_mass = sum(classified_neighbours.values())
        n_clusters = len({sign_to_cluster[n_] for n_ in classified_neighbours})
        expected = expected_distinct(total, pool)
        out[s] = dict(total=total, degree=degree, entropy=entropy,
                       expected_degree=expected, excess=degree - expected,
                       n_neighbour_clusters=n_clusters,
                       classified_frac=classified_mass / total if total else 0,
                       top_neighbours=nb.most_common(3))
    return out


def print_ranked(stats, label, top_n=TOP_N_REPORT):
    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["excess"])
    print(f"\n--- Top {label} (excess = observed - expected distinct "
          f"neighbours, frequency-controlled) ---")
    for s, d in ranked[:top_n]:
        print(f"  sign {s:<5} total={d['total']:<4} degree={d['degree']:<3} "
              f"expected={d['expected_degree']:.1f}  excess={d['excess']:+.1f}  "
              f"spans {d['n_neighbour_clusters']} clusters  "
              f"top: {d['top_neighbours']}")
    print(f"\n--- Bottom {label} (most RESTRICTED -- far fewer distinct "
          f"neighbours than expected) ---")
    for s, d in ranked[-top_n:][::-1]:
        print(f"  sign {s:<5} total={d['total']:<4} degree={d['degree']:<3} "
              f"expected={d['expected_degree']:.1f}  excess={d['excess']:+.1f}  "
              f"spans {d['n_neighbour_clusters']} clusters  "
              f"top: {d['top_neighbours']}")


def plot_rarefaction(pred_stats, succ_stats):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, stats, title, tag in [
        (axes[0], pred_stats, "Predecessor diversity (IN-hub / restricted)", "pred"),
        (axes[1], succ_stats, "Successor diversity (OUT-hub / restricted)", "succ"),
    ]:
        totals = np.array([d["total"] for d in stats.values()])
        degrees = np.array([d["degree"] for d in stats.values()])
        signs = list(stats.keys())
        excess = np.array([d["excess"] for d in stats.values()])

        order = np.argsort(totals)
        c_grid = np.unique(totals)
        # reuse expected_distinct per unique C via the same pool each
        # stats dict was built with -- approximate the curve by
        # interpolating from the actual per-point expected values
        exp_vals = np.array([stats[s]["expected_degree"] for s in signs])
        curve_order = np.argsort(totals)
        ax.plot(totals[curve_order], exp_vals[curve_order], "--", color="gray",
                linewidth=1.5, label="expected (frequency-only null)")
        ax.scatter(totals, degrees, c=excess, cmap="RdBu_r", vmin=-15, vmax=15,
                   edgecolors="k", linewidths=0.3, zorder=3)

        top_hub = max(range(len(signs)), key=lambda i: excess[i])
        top_restricted = min(range(len(signs)), key=lambda i: excess[i])
        for i in [top_hub, top_restricted]:
            ax.annotate(signs[i], (totals[i], degrees[i]), fontsize=9,
                        fontweight="bold", xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("total occurrences as predecessor/successor slot")
        ax.set_ylabel("distinct neighbours observed")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("nplm_hub_rarefaction.png", dpi=150)
    plt.close(fig)
    print("\n  saved nplm_hub_rarefaction.png")


def main():
    inscriptions = load_inscriptions()
    meta = sign_metadata(inscriptions)
    kept, labels, k = reproduce_clustering(inscriptions, meta)
    sign_to_cluster = dict(zip(kept, labels))
    print(f"Reproduced {TARGET_CONFIG}/{TARGET_KIND} seed {TARGET_SEED}: "
          f"{len(kept)} classified signs, k={k}")

    pred_counts, succ_counts, pred_pool, succ_pool = build_neighbour_counts(inscriptions)

    pred_stats = analyse_direction(kept, kept, pred_counts, pred_pool, sign_to_cluster)
    succ_stats = analyse_direction(kept, kept, succ_counts, succ_pool, sign_to_cluster)

    print_ranked(pred_stats, "IN-hubs (many distinct predecessors)")
    print_ranked(succ_stats, "OUT-hubs (many distinct successors)")

    if "59" in pred_stats:
        d = pred_stats["59"]
        print(f"\nSanity check -- sign 59 (found by accident earlier): "
              f"excess={d['excess']:+.1f}, spans {d['n_neighbour_clusters']} "
              f"predecessor clusters, top predecessors {d['top_neighbours']}")

    plot_rarefaction(pred_stats, succ_stats)


if __name__ == "__main__":
    main()
