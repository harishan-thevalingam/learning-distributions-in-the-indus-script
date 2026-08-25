# =====================================================================
# indus_w2v_analyse.py
#
# Analysis of the 50 word2vec models trained by indus_w2v_train.py.
# Mirrors indus_umap_v2.py's role-coherence formula exactly so the
# numbers slot directly into the existing table:
#   raw 64d transformer embeddings   0.2161 [0.2212]
#   model successor distribution     0.1119 [0.2195]
#   bigram successor distribution    0.1333 [0.2218]
#
# and reproduces sign_meta.csv's exact definitions (count, texts,
# end_rate, start_rate) directly from induscorpus.txt -- from
# indus_export_vectors.jl's own code -- so no Julia CSV export is
# required to run this.
#
# Sections (numbering follows the carry-forward summary; there's no
# Section 4, that gap is inherited, not a mistake here):
#   1. Seed stability   -- do independently-seeded models agree on
#                           neighbour structure? Chance = k/(N-1).
#                           GATES everything below.
#   2. Role coherence    -- same 5-NN |end-rate gap| metric as the
#                           transformer/bigram table above.
#   3. Clusters          -- KMeans on COSINE-NORMALISED vectors, k
#                           picked by silhouette, ARI stability
#                           across seeds; the silhouette-vs-k curve;
#                           a fixed k=6 comparison (matches the
#                           transformer successor-distribution KMeans
#                           run); THREE diagnostics for why clustering
#                           looks weak on raw embeddings --
#                           (i) does the found split correlate with
#                           end_rate/start_rate at all, even loosely;
#                           (ii) does a PCA-reduced subspace do any
#                           better (curse-of-dimensionality check);
#                           (iii) THE FAIR COMPARISON -- cluster on
#                           the corpus-derived bigram successor+
#                           predecessor distribution instead of any
#                           learned embedding. This is the exact
#                           representation the terminal/opener ground-
#                           truth classes were originally found in
#                           (via the transformer's version of it), so
#                           it's the honest baseline for whether
#                           word2vec's raw vectors are underperforming
#                           or whether global KMeans on an arbitrary
#                           embedding basis was never a fair test to
#                           begin with -- the transformer's own raw
#                           embeddings failed exactly this way too
#                           (0.2161, at chance, in the Section 2
#                           table above).
#
#   5. Linear structure  -- (a) [trusted] cross-validated ridge
#                           regression of end_rate on the embedding,
#                           out-of-fold Pearson r (non-circular: no
#                           sign's prediction uses its own label).
#                           (b) [less trusted] analogy completion
#                           (ender_i - starter_i + starter_j ~
#                           ender_j), top-1 accuracy vs printed
#                           chance rate -- ~76 candidates means some
#                           hits are luck.
#
# Requires: pip install gensim scikit-learn numpy matplotlib
# Optional: pip install umap-learn  (nicer 2D projection for the cluster
#           plot; falls back to PCA if not installed)
# Run in the same folder as induscorpus.txt and w2v_models/:
#   python indus_w2v_analyse.py
#
# Writes three PNGs: w2v_clusters.png, w2v_linear_direction.png,
# w2v_analogy.png
# =====================================================================

import csv
import itertools
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
    return PCA(n_components=2, random_state=seed).fit_transform(X)

CORPUS_PATH = "induscorpus.txt"
MANIFEST_PATH = "w2v_models/manifest.csv"
MIN_COUNT_ROLE = 20     # matches indus_umap_v2.py's MIN_COUNT (76 signs)
MIN_COUNT_CLUSTER = 15  # matches the transformer KMeans run (87 signs)
N_NEIGHBOURS = 5
N_SHUFFLES = 50
RNG = np.random.default_rng(0)

TERMINAL_CLASS = {"342", "176", "211", "1", "15", "254", "12", "60"}
OPENER_CLASS = {"293", "216", "150", "222"}


# ---------------------------------------------------------------------
# corpus + sign metadata (reproduces indus_export_vectors.jl exactly)
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
    """count, texts, end_rate, start_rate -- identical definitions to
    indus_export_vectors.jl's sign_meta.csv: count = total token
    occurrences; texts = number of distinct inscriptions containing
    the sign at least once; end_rate/start_rate = fraction of those
    texts where the sign is the last/first token."""
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


def bigram_context_vectors(inscriptions, row_signs, all_signs=None):
    """Empirical successor+predecessor distribution per sign, computed
    directly from the corpus -- no model, no embedding involved. Same
    definition as indus_export_vectors.jl's ctx_bigram.csv (successor
    half ++ predecessor half). This is the canonical, sign-indexed
    representation the terminal/opener ground-truth classes were
    originally found in (via the transformer's version of it), so
    it's the fair baseline to cluster against."""
    if all_signs is None:
        all_signs = sorted({s for ins in inscriptions for s in ins}, key=int)
    col_idx = {s: i for i, s in enumerate(all_signs)}
    row_idx = {s: i for i, s in enumerate(row_signs)}
    ns_row, ns_col = len(row_signs), len(all_signs)
    succ = np.zeros((ns_row, ns_col))
    pred = np.zeros((ns_row, ns_col))
    for ins in inscriptions:
        for a, b in zip(ins, ins[1:]):
            if a in row_idx and b in col_idx:
                succ[row_idx[a], col_idx[b]] += 1
            if b in row_idx and a in col_idx:
                pred[row_idx[b], col_idx[a]] += 1
    succ = succ / (succ.sum(axis=1, keepdims=True) + 1e-12)
    pred = pred / (pred.sum(axis=1, keepdims=True) + 1e-12)
    return np.hstack([succ, pred])


# ---------------------------------------------------------------------
# manifest / model loading
# ---------------------------------------------------------------------
def load_manifest(path=MANIFEST_PATH):
    configs = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            configs[row["config"]].append(row)
    return configs


def load_vectors(model_path, signs):
    """Returns (kept_signs, matrix) for the signs in `signs` that are
    present in this model's vocab, preserving `signs` order."""
    m = Word2Vec.load(model_path)
    kept = [s for s in signs if s in m.wv]
    X = np.stack([m.wv[s] for s in kept])
    return kept, X


def cosine_matrix(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


# ---------------------------------------------------------------------
# Section 1: seed stability
# ---------------------------------------------------------------------
def topk_neighbours(X, k=N_NEIGHBOURS):
    C = cosine_matrix(X)
    np.fill_diagonal(C, -np.inf)
    return np.argsort(-C, axis=1)[:, :k]


def section1_seed_stability(configs, meta):
    print("=" * 70)
    print("SECTION 1: seed stability")
    print("=" * 70)
    print(f"Restricted to signs with count >= {MIN_COUNT_ROLE}.\n")

    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_ROLE], key=int
    )
    results = {}
    for cname, rows in configs.items():
        nn_by_seed = []
        for row in rows:
            kept, X = load_vectors(row["path"], kept_master)
            nn_by_seed.append((kept, topk_neighbours(X)))
        base_signs = nn_by_seed[0][0]
        assert all(ks == base_signs for ks, _ in nn_by_seed), (
            f"{cname}: kept-sign set differs across seeds -- check "
            f"min_count in indus_w2v_train.py (should be 1)"
        )
        n = len(base_signs)
        chance = N_NEIGHBOURS / (n - 1)

        overlaps = []
        for (_, nn_a), (_, nn_b) in itertools.combinations(nn_by_seed, 2):
            for i in range(n):
                overlaps.append(len(set(nn_a[i]) & set(nn_b[i])) / N_NEIGHBOURS)
        mean_overlap = float(np.mean(overlaps))
        results[cname] = dict(mean_overlap=mean_overlap, chance=chance, n=n)
        flag = "  <-- AT CHANCE" if mean_overlap < 1.5 * chance else ""
        print(f"  {cname:<10} overlap={mean_overlap:.4f}  chance={chance:.4f}{flag}")

    print()
    if all(r["mean_overlap"] < 1.5 * r["chance"] for r in results.values()):
        print("ALL configs at chance: on this corpus size, word2vec neighbour")
        print("structure is not stable across seeds. Sections 2/3/5 below are")
        print("still computed but should be read as noise characterisation,")
        print("not as evidence about the script -- '~7,000 tokens is too few")
        print("for word2vec' is itself the finding.\n")
    return results


# ---------------------------------------------------------------------
# Section 2: role coherence (same formula as indus_umap_v2.py)
# ---------------------------------------------------------------------
def role_coherence(X, kept_signs, meta, n_shuffles=N_SHUFFLES, rng=RNG):
    ek = np.array([meta[s]["end_rate"] for s in kept_signs])
    C = cosine_matrix(X)
    np.fill_diagonal(C, -np.inf)
    nn = np.argsort(-C, axis=1)[:, :N_NEIGHBOURS]
    got = float(np.mean(np.abs(ek[:, None] - ek[nn])))
    base = float(np.mean([
        np.mean(np.abs((p := rng.permutation(ek))[:, None] - p[nn]))
        for _ in range(n_shuffles)
    ]))
    return got, base


def section2_role_coherence(configs, meta):
    print("=" * 70)
    print("SECTION 2: role coherence (5-NN mean |end-rate gap|, lower=better)")
    print("=" * 70)
    print("Existing table for comparison:")
    print("  raw 64d transformer embeddings    0.2161 [0.2212]")
    print("  model successor distribution      0.1119 [0.2195]")
    print("  bigram successor distribution     0.1333 [0.2218]\n")

    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_ROLE], key=int
    )
    for cname, rows in configs.items():
        gots, bases = [], []
        for row in rows:
            kept, X = load_vectors(row["path"], kept_master)
            g, b = role_coherence(X, kept, meta)
            gots.append(g)
            bases.append(b)
        print(f"  w2v {cname:<10} {np.mean(gots):.4f} +/- {np.std(gots):.4f} "
              f"[{np.mean(bases):.4f}]  (n={len(kept_master)} signs, 10 seeds)")
    print()


# ---------------------------------------------------------------------
# Section 3: clusters
# ---------------------------------------------------------------------
def best_kmeans(X, k_range=range(2, 11), seed=0):
    """KMeans on COSINE-NORMALISED vectors: word2vec vector norm
    correlates with sign frequency, so KMeans on raw vectors would
    partly cluster on 'how common is this sign' rather than 'what
    direction does it point in'. Normalising makes Euclidean-on-unit-
    vectors equivalent to cosine, consistent with Sections 1/2 and
    the UMAP plot. Also returns the full silhouette-vs-k curve."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    scores = {}
    best_k, best_score, best_labels = None, -1.0, None
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xn)
        score = silhouette_score(Xn, labels)
        scores[k] = score
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_k, best_score, best_labels, scores


def cluster_diagnostics(kept, labels, meta):
    """Does the found partition relate to function at all, even
    loosely? A real gap here (without clean separation) says the
    signal is present but diffuse across many clusters; near-
    identical means across clusters say KMeans found some other
    dominant axis entirely."""
    ek = np.array([meta[s]["end_rate"] for s in kept])
    sk = np.array([meta[s]["start_rate"] for s in kept])
    ck = np.array([meta[s]["count"] for s in kept])
    labels = np.asarray(labels)
    for lab in sorted(set(labels)):
        mask = labels == lab
        print(f"    cluster {lab}: n={mask.sum():<3} "
              f"mean end_rate={ek[mask].mean():.3f}  "
              f"mean start_rate={sk[mask].mean():.3f}  "
              f"mean count={ck[mask].mean():.1f}")


def pca_then_cluster(X, n_components=6, k_range=range(2, 11), seed=0):
    """Curse-of-dimensionality check: cluster on a PCA-reduced
    subspace instead of all 24 raw dims. 87 signs in 24 dimensions is
    thin for unsupervised structure to win out in a silhouette
    search, independent of which embedding algorithm produced the
    vectors."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    ncomp = min(n_components, Xn.shape[1])
    pca = PCA(n_components=ncomp, random_state=seed)
    reduced = pca.fit_transform(Xn)
    print(f"    PCA explained variance (top {ncomp} of {Xn.shape[1]} dims): "
          f"{', '.join(f'{v:.3f}' for v in pca.explained_variance_ratio_)}"
          f"  (cumulative {pca.explained_variance_ratio_.sum():.3f})")
    best_k, best_score, best_labels = None, -1.0, None
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_k, best_labels


def section3_clusters(configs, meta, inscriptions):
    print("=" * 70)
    print("SECTION 3: clusters (KMeans, k by silhouette, ARI stability)")
    print("=" * 70)
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_CLUSTER], key=int
    )
    print(f"Restricted to signs with count >= {MIN_COUNT_CLUSTER} "
          f"({len(kept_master)} signs, matches the transformer KMeans run).\n")

    best_config, best_stability = None, -1.0
    per_config_labels = {}
    per_config_scores = {}

    for cname, rows in configs.items():
        labelings, ks, all_scores = [], [], []
        for row in rows:
            kept, X = load_vectors(row["path"], kept_master)
            assert kept == kept_master, (
                f"{cname}: kept-sign set differs across seeds -- check "
                f"min_count in indus_w2v_train.py (should be 1)"
            )
            k, score, labels, scores = best_kmeans(X)
            labelings.append(labels)
            ks.append(k)
            all_scores.append(scores)
        aris = [
            adjusted_rand_score(la, lb)
            for la, lb in itertools.combinations(labelings, 2)
        ]
        stability = float(np.mean(aris))
        per_config_labels[cname] = (kept_master, labelings[0])  # seed 0
        per_config_scores[cname] = all_scores
        mode_k = max(set(ks), key=ks.count)
        print(f"  {cname:<10} k={min(ks)}-{max(ks)} (mode {mode_k})  "
              f"ARI stability={stability:.4f} +/- {np.std(aris):.4f}")
        if stability > best_stability:
            best_stability, best_config = stability, cname

    print(f"\nMost stable config: {best_config} (ARI={best_stability:.4f}).")
    print(f"\nSilhouette-vs-k for {best_config}, seed 0 (flat/noisy here means")
    print(f"k=2 is a silhouette artifact, not a finding):")
    for k, s in per_config_scores[best_config][0].items():
        print(f"    k={k:<2} silhouette={s:.4f}")

    print(f"\nChecking its seed-0 clustering against the known classes:\n")

    kept, labels = per_config_labels[best_config]
    sign_to_cluster = dict(zip(kept, labels))
    # reload seed-0 vectors for this config for the diagnostics + plot below
    _, X = load_vectors(configs[best_config][0]["path"], kept_master)

    for name, cls in [("terminal", TERMINAL_CLASS), ("opener", OPENER_CLASS)]:
        present = [s for s in cls if s in sign_to_cluster]
        missing = cls - set(present)
        if not present:
            print(f"  {name} class: none of {sorted(cls, key=int)} present "
                  f"at count >= {MIN_COUNT_CLUSTER}")
            continue
        labs = [sign_to_cluster[s] for s in present]
        majority = max(set(labs), key=labs.count)
        purity = labs.count(majority) / len(labs)
        miss_str = f" (missing {sorted(missing, key=int)})" if missing else ""
        print(f"  {name} class {sorted(present, key=int)}{miss_str}")
        print(f"    -> {labs.count(majority)}/{len(labs)} in cluster {majority} "
              f"(purity {purity:.2f})")

    both = [s for s in (TERMINAL_CLASS | OPENER_CLASS) if s in sign_to_cluster]
    if len(both) >= 3:
        true_labels = [0 if s in TERMINAL_CLASS else 1 for s in both]
        pred_labels = [sign_to_cluster[s] for s in both]
        ari_vs_truth = adjusted_rand_score(true_labels, pred_labels)
        print(f"\n  ARI (w2v clusters vs terminal/opener ground truth, "
              f"{len(both)} labelled signs): {ari_vs_truth:.4f}")

    print(f"\nFor comparison, fixed k=6 (matches the transformer successor-")
    print(f"distribution KMeans run, independent of silhouette's k choice):")
    Xn6 = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    labels6 = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(Xn6)
    sign_to_cluster6 = dict(zip(kept, labels6))
    for name, cls in [("terminal", TERMINAL_CLASS), ("opener", OPENER_CLASS)]:
        present = [s for s in cls if s in sign_to_cluster6]
        if not present:
            continue
        labs = [sign_to_cluster6[s] for s in present]
        majority = max(set(labs), key=labs.count)
        print(f"  {name}: {labs.count(majority)}/{len(labs)} in cluster "
              f"{majority} (purity {labs.count(majority)/len(labs):.2f})")
    both6 = [s for s in (TERMINAL_CLASS | OPENER_CLASS) if s in sign_to_cluster6]
    if len(both6) >= 3:
        true6 = [0 if s in TERMINAL_CLASS else 1 for s in both6]
        pred6 = [sign_to_cluster6[s] for s in both6]
        print(f"  ARI vs ground truth (k=6): "
              f"{adjusted_rand_score(true6, pred6):.4f}")

    print(f"\nDiagnostic (i): does the {best_config} seed-0 clustering "
          f"(k={len(set(labels))}) relate to function at all, even loosely?")
    cluster_diagnostics(kept, labels, meta)

    print(f"\nDiagnostic (ii): clustering on a PCA-reduced subspace instead "
          f"of all {X.shape[1]} raw dims:")
    pca_k, pca_labels = pca_then_cluster(X)
    print(f"    best k on PCA subspace: {pca_k}")
    cluster_diagnostics(kept, pca_labels, meta)
    sign_to_cluster_pca = dict(zip(kept, pca_labels))
    both_pca = [s for s in (TERMINAL_CLASS | OPENER_CLASS) if s in sign_to_cluster_pca]
    if len(both_pca) >= 3:
        true_pca = [0 if s in TERMINAL_CLASS else 1 for s in both_pca]
        pred_pca = [sign_to_cluster_pca[s] for s in both_pca]
        print(f"    ARI vs ground truth (PCA subspace): "
              f"{adjusted_rand_score(true_pca, pred_pca):.4f}")

    print(f"\nDiagnostic (iii) -- THE FAIR COMPARISON: cluster on the")
    print(f"corpus-derived bigram successor+predecessor distribution")
    print(f"instead of any learned embedding. This is the representation")
    print(f"the terminal/opener ground-truth classes were originally found")
    print(f"in -- if THIS clusters cleanly and word2vec's raw vectors don't,")
    print(f"that confirms it's an arbitrary-embedding-basis issue (which the")
    print(f"transformer's own raw embeddings also had, 0.2161 at chance in")
    print(f"Section 2), not a word2vec-specific failure:")
    bigram_vecs = bigram_context_vectors(inscriptions, kept)
    bk, bscore, blabels, _ = best_kmeans(bigram_vecs)
    print(f"    best k={bk} (silhouette {bscore:.4f})")
    cluster_diagnostics(kept, blabels, meta)
    sign_to_cluster_bg = dict(zip(kept, blabels))
    both_bg = [s for s in (TERMINAL_CLASS | OPENER_CLASS) if s in sign_to_cluster_bg]
    if len(both_bg) >= 3:
        true_bg = [0 if s in TERMINAL_CLASS else 1 for s in both_bg]
        pred_bg = [sign_to_cluster_bg[s] for s in both_bg]
        print(f"    ARI vs ground truth (bigram distribution): "
              f"{adjusted_rand_score(true_bg, pred_bg):.4f}")

    # ---- plot: 2D projection of the most-stable config, coloured by
    # cluster, terminal/opener class ringed ----
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
    ax.set_title(f"word2vec ({best_config}) sign clusters, "
                 f"k={len(set(labels))}, count >= {MIN_COUNT_CLUSTER}\n"
                 f"red ring = terminal class, blue ring = opener class "
                 f"({'UMAP' if HAVE_UMAP else 'PCA'} projection)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("w2v_clusters.png", dpi=150)
    plt.close(fig)
    print(f"\n  saved w2v_clusters.png")
    print()


# ---------------------------------------------------------------------
# Section 5: linear structure
# ---------------------------------------------------------------------
def section5a_linear_direction(configs, meta):
    print("=" * 70)
    print("SECTION 5a [trusted]: starter->ender linear direction")
    print("=" * 70)
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_ROLE], key=int
    )
    print(f"Cross-validated (5-fold) Ridge regression of end_rate on the "
          f"embedding, restricted to count >= {MIN_COUNT_ROLE} "
          f"({len(kept_master)} signs). Out-of-fold Pearson r reported -- "
          f"non-circular, no sign's prediction uses its own label.\n")

    results = {}
    for cname, rows in configs.items():
        rs = []
        oof_seed0 = None
        for row in rows:
            kept, X = load_vectors(row["path"], kept_master)
            ek = np.array([meta[s]["end_rate"] for s in kept])
            preds = np.zeros_like(ek)
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            for tr, te in kf.split(X):
                reg = Ridge(alpha=1.0).fit(X[tr], ek[tr])
                preds[te] = reg.predict(X[te])
            rs.append(np.corrcoef(preds, ek)[0, 1])
            if oof_seed0 is None:
                oof_seed0 = (ek.copy(), preds.copy())
        results[cname] = dict(mean_r=np.mean(rs), std_r=np.std(rs), oof=oof_seed0)
        print(f"  w2v {cname:<10} out-of-fold r = {np.mean(rs):.3f} +/- {np.std(rs):.3f}")

    best_cname = max(results, key=lambda c: results[c]["mean_r"])
    ek, preds = results[best_cname]["oof"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ek, preds, alpha=0.75, edgecolors="k", linewidths=0.3)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("actual end_rate")
    ax.set_ylabel("out-of-fold predicted end_rate")
    ax.set_title(f"w2v {best_cname}: starter->ender linear direction\n"
                 f"out-of-fold r = {results[best_cname]['mean_r']:.3f} "
                 f"(seed 0 shown)")
    plt.tight_layout()
    plt.savefig("w2v_linear_direction.png", dpi=150)
    plt.close(fig)
    print(f"\n  saved w2v_linear_direction.png (best config: {best_cname})")
    print()
    return results


def section5b_analogy(configs, meta):
    print("=" * 70)
    print("SECTION 5b [less trusted]: analogy completion")
    print("=" * 70)
    kept_master = sorted(
        [s for s, d in meta.items() if d["count"] >= MIN_COUNT_ROLE], key=int
    )
    n_candidates = len(kept_master)
    by_end = sorted(kept_master, key=lambda s: -meta[s]["end_rate"])
    by_start = sorted(kept_master, key=lambda s: -meta[s]["start_rate"])

    n_pairs = max(4, n_candidates // 10)
    enders = by_end[:n_pairs]
    starters_all = [s for s in by_start if s not in enders]
    n_pairs = min(n_pairs, len(starters_all))
    enders, starters = enders[:n_pairs], starters_all[:n_pairs]
    chance = 1.0 / (n_candidates - 3)

    print(f"{n_pairs} (ender, starter) pairs from the top end-rate / "
          f"start-rate signs. Testing ender_i - starter_i + starter_j ~ "
          f"ender_j for i != j, top-1 nearest neighbour among all "
          f"{n_candidates} kept signs (excluding the 3 query terms). "
          f"Chance = {chance:.4f}.\n")

    results = {}
    for cname, rows in configs.items():
        accs = []
        for row in rows:
            kept, X = load_vectors(row["path"], kept_master)
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
                order = [k for k in np.argsort(-sims) if k not in exclude]
                total += 1
                hits += int(order[0] == idx[enders[j]])
            accs.append(hits / total if total else 0.0)
        results[cname] = dict(mean_acc=np.mean(accs), std_acc=np.std(accs))
        print(f"  w2v {cname:<10} top-1 acc = {np.mean(accs):.3f} +/- {np.std(accs):.3f} "
              f"(chance {chance:.4f})")

    names = list(results.keys())
    means = [results[c]["mean_acc"] for c in names]
    stds = [results[c]["std_acc"] for c in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, means, yerr=stds, capsize=4, color="steelblue")
    ax.axhline(chance, color="red", linestyle="--", label=f"chance ({chance:.3f})")
    ax.set_ylabel("analogy top-1 accuracy (mean +/- std over 10 seeds)")
    ax.set_title("Section 5b: ender_i - starter_i + starter_j ~ ender_j")
    ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("w2v_analogy.png", dpi=150)
    plt.close(fig)
    print(f"\n  saved w2v_analogy.png")
    print()
    return results


def main():
    inscriptions = load_inscriptions()
    meta = sign_metadata(inscriptions)
    configs = load_manifest()
    n_seeds = len(next(iter(configs.values())))
    print(f"Loaded {len(inscriptions)} inscriptions, {len(meta)} sign types, "
          f"{len(configs)} configs x {n_seeds} seeds\n")

    section1_seed_stability(configs, meta)
    section2_role_coherence(configs, meta)
    section3_clusters(configs, meta, inscriptions)
    section5a_linear_direction(configs, meta)
    section5b_analogy(configs, meta)


if __name__ == "__main__":
    main()