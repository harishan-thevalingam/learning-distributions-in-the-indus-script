# =====================================================================
# indus_nplm_analyse.py
#
# Runs the SAME battery used on word2vec (seed stability, role
# coherence, clustering, linear structure / analogy) on THREE NPLM
# representations per model:
#   - embedding : the raw input embedding C[i] for sign i -- this is
#                 the "match the embedding-stage results" check,
#                 architecturally the same kind of object as
#                 word2vec's vectors, just trained differently.
#   - hidden    : tanh(d + Hx), averaged over every corpus position
#                 where sign i is the most recent context token --
#                 an intermediate, nonlinearly-transformed
#                 representation with no word2vec analogue.
#   - softmax   : the model's own predicted next-sign distribution,
#                 same averaging -- a genuine trained predictive
#                 distribution, the NPLM analogue of the transformer's
#                 best-performing representation (0.1119 in the
#                 original table). If structure sharpens embedding ->
#                 hidden -> softmax, that's informative about where in
#                 the computation it lives. If none of the three beat
#                 the embedding-stage numbers, that is itself evidence
#                 -- NPLM's whole mechanism is built to exploit
#                 higher-order structure via smoothing over similar
#                 contexts, so if its own best representation doesn't
#                 organise by role any better than a raw embedding
#                 does, that's a real absence-of-structure finding,
#                 not a null result to discard.
#
# Also reports held-out perplexity by context order n (1/2/3) --
# purely NPLM-internal, not compared against any earlier project
# figures (different language, different split -- see
# indus_nplm_train.py's docstring). A flat or worsening ppl trend with
# increasing n is itself a legitimate finding: no higher-order
# structure for this architecture to exploit.
#
# Requires: pip install torch scikit-learn numpy
# Run in the same folder as induscorpus.txt and nplm_models/:
#   python indus_nplm_analyse.py
# =====================================================================

import csv
import itertools
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False


def project_2d(X, seed=42):
    if HAVE_UMAP:
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                          random_state=seed).fit_transform(X)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X)

CORPUS_PATH = "induscorpus.txt"
MANIFEST_PATH = "nplm_models/manifest.csv"
MIN_COUNT_ROLE = 20
MIN_COUNT_CLUSTER = 15
MIN_CONTEXT_OCC = 5   # minimum times a sign must appear as "most recent
                       # context token" for its hidden/softmax average
                       # to be considered reliable
N_NEIGHBOURS = 5
N_SHUFFLES = 50
RNG = np.random.default_rng(0)

TERMINAL_CLASS = {"342", "176", "211", "1", "15", "254", "12", "60"}
OPENER_CLASS = {"293", "216", "150", "222"}


# ---------------------------------------------------------------------
# corpus + vocab (must match indus_nplm_train.py exactly)
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


# ---------------------------------------------------------------------
# representation extraction
# ---------------------------------------------------------------------
@torch.no_grad()
def sign_representations(model, inscriptions, signs_sorted, sign_to_idx,
                          bos_idx, n, min_occ=MIN_CONTEXT_OCC):
    """embedding[s]: raw input embedding, always available.
    hidden[s]/softmax[s]: averaged over every corpus position where s
    is the most recent context token; omitted if fewer than min_occ
    such positions exist (this can genuinely happen for signs that
    are almost always inscription-final -- worth noting if it does,
    since those are exactly the terminal-class signs of interest)."""
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
    skipped = []
    for i, s in enumerate(signs_sorted):
        embedding[s] = model.C.weight[i].detach().numpy()
        mask = last_token == i
        if int(mask.sum()) < min_occ:
            skipped.append(s)
            continue
        softmax_rep[s] = probs[mask].mean(dim=0).numpy()
        hidden_rep[s] = hidden[mask].mean(dim=0).numpy()
    return embedding, softmax_rep, hidden_rep, skipped


def dict_to_matrix(d, order):
    kept = [s for s in order if s in d]
    X = np.stack([d[s] for s in kept]) if kept else np.zeros((0, 1))
    return kept, X


# ---------------------------------------------------------------------
# shared metric functions (same formulas as indus_w2v_analyse.py)
# ---------------------------------------------------------------------
def cosine_matrix(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def topk_neighbours(X, k=N_NEIGHBOURS):
    C = cosine_matrix(X)
    np.fill_diagonal(C, -np.inf)
    return np.argsort(-C, axis=1)[:, :k]


def role_coherence(X, kept_signs, meta, n_shuffles=N_SHUFFLES, rng=RNG):
    ek = np.array([meta[s]["end_rate"] for s in kept_signs])
    C = cosine_matrix(X)
    np.fill_diagonal(C, -np.inf)
    nn_idx = np.argsort(-C, axis=1)[:, :N_NEIGHBOURS]
    got = float(np.mean(np.abs(ek[:, None] - ek[nn_idx])))
    base = float(np.mean([
        np.mean(np.abs((p := rng.permutation(ek))[:, None] - p[nn_idx]))
        for _ in range(n_shuffles)
    ]))
    return got, base


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


def ari_vs_ground_truth(kept, labels):
    sign_to_cluster = dict(zip(kept, labels))
    both = [s for s in (TERMINAL_CLASS | OPENER_CLASS) if s in sign_to_cluster]
    if len(both) < 3:
        return None
    true_labels = [0 if s in TERMINAL_CLASS else 1 for s in both]
    pred_labels = [sign_to_cluster[s] for s in both]
    return adjusted_rand_score(true_labels, pred_labels)


# ---------------------------------------------------------------------
# battery runners -- one row per (config, representation)
# ---------------------------------------------------------------------
def collect_representations(configs, inscriptions, signs_sorted, sign_to_idx,
                             bos_idx, vocab_size, out_size):
    """Loads every model once, computes all 3 representations per
    (config, seed), returns nested dict: reps[rep_kind][config] =
    list of (kept_signs, X) pairs, one per seed, ready for the battery
    functions below. Also returns skip diagnostics."""
    reps = {"embedding": defaultdict(list), "hidden": defaultdict(list),
            "softmax": defaultdict(list)}
    skip_report = {}
    for cname, rows in configs.items():
        n = int(rows[0]["n"])
        for row in rows:
            model = load_model(row, vocab_size, out_size)
            emb, sm, hid, skipped = sign_representations(
                model, inscriptions, signs_sorted, sign_to_idx, bos_idx, n
            )
            reps["embedding"][cname].append((emb, signs_sorted))
            reps["softmax"][cname].append((sm, signs_sorted))
            reps["hidden"][cname].append((hid, signs_sorted))
            if row is rows[0]:
                skip_report[cname] = skipped
    return reps, skip_report


def section1_seed_stability(reps, kind, meta):
    print(f"\n--- Section 1: seed stability [{kind}] ---")
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_ROLE], key=int
    )
    for cname, seed_list in reps[kind].items():
        nn_by_seed = []
        for d, order in seed_list:
            kept, X = dict_to_matrix(d, kept_master)
            if len(kept) < N_NEIGHBOURS + 2:
                continue
            nn_by_seed.append((kept, topk_neighbours(X)))
        if len(nn_by_seed) < 2:
            print(f"  {cname:<12} insufficient signs with reliable "
                  f"{kind} representations -- skipped")
            continue
        base_signs = nn_by_seed[0][0]
        if not all(ks == base_signs for ks, _ in nn_by_seed):
            print(f"  {cname:<12} kept-sign set varies across seeds "
                  f"(some signs occasionally fall below the min-context-"
                  f"occurrence threshold) -- comparing on the common subset")
            common = set.intersection(*[set(ks) for ks, _ in nn_by_seed])
            base_signs = sorted(common, key=int)
        n = len(base_signs)
        if n < N_NEIGHBOURS + 2:
            print(f"  {cname:<12} too few common signs -- skipped")
            continue
        chance = N_NEIGHBOURS / (n - 1)
        idx_maps = [{s: i for i, s in enumerate(ks)} for ks, _ in nn_by_seed]
        overlaps = []
        for a in range(len(nn_by_seed)):
            for b in range(a + 1, len(nn_by_seed)):
                ks_a, nn_a = nn_by_seed[a]
                ks_b, nn_b = nn_by_seed[b]
                for s in base_signs:
                    ia, ib = idx_maps[a][s], idx_maps[b][s]
                    na = {ks_a[j] for j in nn_a[ia]}
                    nb = {ks_b[j] for j in nn_b[ib]}
                    overlaps.append(len(na & nb) / N_NEIGHBOURS)
        mean_overlap = float(np.mean(overlaps))
        flag = "  <-- AT CHANCE" if mean_overlap < 1.5 * chance else ""
        print(f"  {cname:<12} n={n:<3} overlap={mean_overlap:.4f}  "
              f"chance={chance:.4f}{flag}")


def section2_role_coherence(reps, kind, meta):
    print(f"\n--- Section 2: role coherence [{kind}] ---")
    print("  (existing table for reference: raw 64d transformer 0.2161 "
          "[0.2212], model successor dist 0.1119 [0.2195], word2vec best "
          "0.157 [0.221])")
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_ROLE], key=int
    )
    out = {}
    for cname, seed_list in reps[kind].items():
        gots, bases = [], []
        for d, order in seed_list:
            kept, X = dict_to_matrix(d, kept_master)
            if len(kept) < N_NEIGHBOURS + 2:
                continue
            g, b = role_coherence(X, kept, meta)
            gots.append(g)
            bases.append(b)
        if not gots:
            print(f"  {cname:<12} skipped (too few reliable signs)")
            continue
        out[cname] = dict(mean_got=np.mean(gots), std_got=np.std(gots),
                           mean_base=np.mean(bases))
        print(f"  {cname:<12} {np.mean(gots):.4f} +/- {np.std(gots):.4f} "
              f"[{np.mean(bases):.4f}]  (n~{len(kept)} signs, {len(gots)} seeds)")
    return out


def section3_clusters(reps, kind, meta):
    print(f"\n--- Section 3: clusters [{kind}] ---")
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_CLUSTER], key=int
    )
    out = {}
    for cname, seed_list in reps[kind].items():
        labelings = []
        X0 = None
        for d, order in seed_list:
            kept, X = dict_to_matrix(d, kept_master)
            if len(kept) < 6:
                continue
            k, score, labels = best_kmeans(X)
            labelings.append((kept, labels))
            if X0 is None:
                X0 = X
        if len(labelings) < 2:
            print(f"  {cname:<12} skipped (too few reliable signs)")
            continue
        common = sorted(set.intersection(*[set(ks) for ks, _ in labelings]), key=int)
        aris = []
        for a in range(len(labelings)):
            for b in range(a + 1, len(labelings)):
                ks_a, la = labelings[a]
                ks_b, lb = labelings[b]
                ia = {s: i for i, s in enumerate(ks_a)}
                ib = {s: i for i, s in enumerate(ks_b)}
                la_c = [la[ia[s]] for s in common]
                lb_c = [lb[ib[s]] for s in common]
                aris.append(adjusted_rand_score(la_c, lb_c))
        stability = float(np.mean(aris)) if aris else float("nan")
        ks0, l0 = labelings[0]
        ari_gt = ari_vs_ground_truth(ks0, l0)
        gt_str = f"{ari_gt:.4f}" if ari_gt is not None else "n/a"
        print(f"  {cname:<12} k={len(set(l0))}  ARI stability={stability:.4f}  "
              f"ARI vs ground truth (seed 0) = {gt_str}")
        out[cname] = dict(stability=stability, ari_gt=ari_gt, k=len(set(l0)),
                           kept=ks0, labels=l0, X=X0)
    return out


def section5a_linear_direction(reps, kind, meta):
    print(f"\n--- Section 5a [trusted]: linear direction [{kind}] ---")
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_ROLE], key=int
    )
    out = {}
    for cname, seed_list in reps[kind].items():
        rs = []
        oof0 = None
        for d, order in seed_list:
            kept, X = dict_to_matrix(d, kept_master)
            if len(kept) < 10:
                continue
            ek = np.array([meta[s]["end_rate"] for s in kept])
            preds = np.zeros_like(ek)
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            for tr, te in kf.split(X):
                reg = Ridge(alpha=1.0).fit(X[tr], ek[tr])
                preds[te] = reg.predict(X[te])
            rs.append(np.corrcoef(preds, ek)[0, 1])
            if oof0 is None:
                oof0 = (ek.copy(), preds.copy())
        if not rs:
            print(f"  {cname:<12} skipped (too few reliable signs)")
            continue
        print(f"  {cname:<12} out-of-fold r = {np.mean(rs):.3f} +/- {np.std(rs):.3f}")
        out[cname] = dict(mean_r=np.mean(rs), std_r=np.std(rs), oof=oof0)
    return out


def section5b_analogy(reps, kind, meta):
    print(f"\n--- Section 5b [less trusted]: analogy completion [{kind}] ---")
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_ROLE], key=int
    )
    by_end = sorted(kept_master, key=lambda s: -meta[s]["end_rate"])
    by_start = sorted(kept_master, key=lambda s: -meta[s]["start_rate"])
    n_pairs = max(4, len(kept_master) // 10)
    enders = by_end[:n_pairs]
    starters_all = [s for s in by_start if s not in enders]
    n_pairs = min(n_pairs, len(starters_all))
    enders, starters = enders[:n_pairs], starters_all[:n_pairs]
    n_candidates = len(kept_master)
    chance = 1.0 / (n_candidates - 3)
    print(f"  {n_pairs} pairs, {n_candidates} candidates, chance={chance:.4f}")

    out = {}
    for cname, seed_list in reps[kind].items():
        accs = []
        for d, order in seed_list:
            kept, X = dict_to_matrix(d, kept_master)
            idx = {s: i for i, s in enumerate(kept)}
            if not all(s in idx for s in enders + starters):
                continue
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            hits, total = 0, 0
            for i, j in itertools.permutations(range(n_pairs), 2):
                query = Xn[idx[enders[i]]] - Xn[idx[starters[i]]] + Xn[idx[starters[j]]]
                query = query / (np.linalg.norm(query) + 1e-12)
                sims = Xn @ query
                exclude = {idx[enders[i]], idx[starters[i]], idx[starters[j]]}
                order_ = [k for k in np.argsort(-sims) if k not in exclude]
                total += 1
                hits += int(order_[0] == idx[enders[j]])
            accs.append(hits / total if total else 0.0)
        if not accs:
            print(f"  {cname:<12} skipped (too few reliable signs)")
            continue
        print(f"  {cname:<12} top-1 acc = {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        out[cname] = dict(mean_acc=np.mean(accs), std_acc=np.std(accs), chance=chance)
    return out


def report_context_order_ppl(configs):
    print("\n" + "=" * 70)
    print("Held-out perplexity by context order (NPLM-internal comparison")
    print("only -- not comparable to any earlier project figures)")
    print("=" * 70)
    by_n = defaultdict(list)
    for cname, rows in configs.items():
        for row in rows:
            by_n[(int(row["n"]), row["direct"])].append(float(row["val_ppl"]))
    for (n, direct), vals in sorted(by_n.items()):
        print(f"  n={n}  direct={direct:<5}  val_ppl = "
              f"{np.mean(vals):.2f} +/- {np.std(vals):.2f}  ({len(vals)} seeds)")


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------
def plot_role_coherence(role_results):
    kinds = ["embedding", "hidden", "softmax"]
    configs = list(role_results["embedding"].keys())
    x = np.arange(len(configs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, kind in enumerate(kinds):
        means = [role_results[kind].get(c, {}).get("mean_got", np.nan) for c in configs]
        stds = [role_results[kind].get(c, {}).get("std_got", 0) for c in configs]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=kind)
    ax.axhline(0.2161, color="gray", linestyle="--", linewidth=1,
               label="transformer raw embed (0.2161)")
    ax.axhline(0.1119, color="green", linestyle="--", linewidth=1,
               label="transformer successor dist (0.1119)")
    ax.axhline(0.157, color="orange", linestyle="--", linewidth=1,
               label="word2vec best (0.157)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20)
    ax.set_ylabel("5-NN mean |end-rate gap| (lower = better)")
    ax.set_title("NPLM role coherence by representation and config")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig("nplm_role_coherence.png", dpi=150)
    plt.close(fig)
    print("  saved nplm_role_coherence.png")


def plot_best_clusters(cluster_results, meta):
    best = None
    for kind, d in cluster_results.items():
        for cname, r in d.items():
            if r["ari_gt"] is None:
                continue
            if best is None or r["ari_gt"] > best[2]["ari_gt"]:
                best = (kind, cname, r)
    if best is None:
        print("  no valid cluster combo to plot (skipped)")
        return
    kind, cname, r = best
    kept, labels, X = r["kept"], r["labels"], r["X"]
    counts = np.array([meta[s]["count"] for s in kept])
    Y = project_2d(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10",
               s=40 + 200 * counts / counts.max(),
               edgecolors="k", linewidths=0.3, zorder=2)
    for j, s in enumerate(kept):
        if s in TERMINAL_CLASS:
            ax.scatter(*Y[j], s=260, facecolors="none", edgecolors="red",
                       linewidths=2, zorder=3)
        elif s in OPENER_CLASS:
            ax.scatter(*Y[j], s=260, facecolors="none", edgecolors="blue",
                       linewidths=2, zorder=3)
        ax.annotate(s, Y[j], fontsize=7, zorder=4)
    ax.set_title(f"NPLM ({cname}, {kind}) sign clusters, k={len(set(labels))}\n"
                 f"ARI vs ground truth = {r['ari_gt']:.4f} -- best of all "
                 f"18 (config, representation) combos tested\n"
                 f"red ring = terminal class, blue ring = opener class "
                 f"({'UMAP' if HAVE_UMAP else 'PCA'} projection)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("nplm_clusters.png", dpi=150)
    plt.close(fig)
    print(f"  saved nplm_clusters.png (best combo: {cname}, {kind})")


def plot_best_linear_direction(linear_results):
    best = None
    for kind, d in linear_results.items():
        for cname, r in d.items():
            if best is None or r["mean_r"] > best[2]["mean_r"]:
                best = (kind, cname, r)
    if best is None:
        print("  no valid linear-direction combo to plot (skipped)")
        return
    kind, cname, r = best
    ek, preds = r["oof"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ek, preds, alpha=0.75, edgecolors="k", linewidths=0.3)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("actual end_rate")
    ax.set_ylabel("out-of-fold predicted end_rate")
    ax.set_title(f"NPLM ({cname}, {kind}): starter->ender linear direction\n"
                 f"out-of-fold r = {r['mean_r']:.3f} (seed 0 shown) -- best "
                 f"of all 18 combos tested")
    plt.tight_layout()
    plt.savefig("nplm_linear_direction.png", dpi=150)
    plt.close(fig)
    print(f"  saved nplm_linear_direction.png (best combo: {cname}, {kind})")


def plot_analogy(analogy_results, flagship="n2_direct"):
    kinds = ["embedding", "hidden", "softmax"]
    means, stds, chance = [], [], None
    for k in kinds:
        r = analogy_results.get(k, {}).get(flagship)
        means.append(r["mean_acc"] if r else np.nan)
        stds.append(r["std_acc"] if r else 0)
        if r is not None:
            chance = r["chance"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(kinds, means, yerr=stds, capsize=4, color="steelblue")
    if chance is not None:
        ax.axhline(chance, color="red", linestyle="--", label=f"chance ({chance:.3f})")
        ax.legend()
    ax.set_ylabel("analogy top-1 accuracy")
    ax.set_title(f"NPLM analogy completion by representation ({flagship})")
    plt.tight_layout()
    plt.savefig("nplm_analogy.png", dpi=150)
    plt.close(fig)
    print("  saved nplm_analogy.png")


def plot_ppl_by_n(configs):
    by_n = defaultdict(list)
    for cname, rows in configs.items():
        for row in rows:
            by_n[(int(row["n"]), row["direct"])].append(float(row["val_ppl"]))
    ns = sorted({n for n, _ in by_n})
    fig, ax = plt.subplots(figsize=(7, 5))
    for direct in ["True", "False"]:
        means = [np.mean(by_n[(n, direct)]) for n in ns]
        stds = [np.std(by_n[(n, direct)]) for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", capsize=4,
                     label=f"direct={direct}")
    ax.set_xlabel("context order n")
    ax.set_ylabel("held-out perplexity")
    ax.set_title("NPLM: does more context reduce perplexity?\n"
                  "(NPLM-internal comparison only)")
    ax.set_xticks(ns)
    ax.legend()
    plt.tight_layout()
    plt.savefig("nplm_ppl_by_n.png", dpi=150)
    plt.close(fig)
    print("  saved nplm_ppl_by_n.png")


def main():
    inscriptions = load_inscriptions()
    meta = sign_metadata(inscriptions)
    signs_sorted, sign_to_idx, bos_idx = build_vocab(inscriptions)
    vocab_size = len(signs_sorted) + 1
    out_size = len(signs_sorted)
    configs = load_manifest()
    n_seeds = len(next(iter(configs.values())))
    print(f"Loaded {len(inscriptions)} inscriptions, {len(signs_sorted)} sign "
          f"types, {len(configs)} configs x {n_seeds} seeds")

    reps, skip_report = collect_representations(
        configs, inscriptions, signs_sorted, sign_to_idx, bos_idx,
        vocab_size, out_size
    )
    for cname, skipped in skip_report.items():
        if skipped:
            print(f"\n  Note: {cname} seed 0 -- {len(skipped)} signs had "
                  f"< {MIN_CONTEXT_OCC} occurrences as a preceding context "
                  f"token, excluded from hidden/softmax reps: "
                  f"{sorted(skipped, key=int)[:10]}"
                  f"{'...' if len(skipped) > 10 else ''}")

    role_results, cluster_results = {}, {}
    linear_results, analogy_results = {}, {}
    for kind in ["embedding", "hidden", "softmax"]:
        print("\n" + "#" * 70)
        print(f"# REPRESENTATION: {kind}")
        print("#" * 70)
        section1_seed_stability(reps, kind, meta)
        role_results[kind] = section2_role_coherence(reps, kind, meta)
        cluster_results[kind] = section3_clusters(reps, kind, meta)
        linear_results[kind] = section5a_linear_direction(reps, kind, meta)
        analogy_results[kind] = section5b_analogy(reps, kind, meta)

    report_context_order_ppl(configs)

    print("\n" + "=" * 70)
    print("PLOTS")
    print("=" * 70)
    plot_role_coherence(role_results)
    plot_best_clusters(cluster_results, meta)
    plot_best_linear_direction(linear_results)
    plot_analogy(analogy_results)
    plot_ppl_by_n(configs)


if __name__ == "__main__":
    main()
