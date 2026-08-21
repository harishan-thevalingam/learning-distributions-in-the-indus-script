# =====================================================================
#  indus_umap_v2.py
#
#  UMAP of the sign space, on three representations exported by
#  indus_export_vectors.jl. All three are now compared fairly:
#
#    A  raw 64-dim transformer embeddings
#    B  model's next-sign distribution per sign, SPECIALS DROPPED
#    C  empirical next-sign distribution (bigram), successor half only
#
#  B previously included the <eos> column. For text-enders the model puts
#  most of its mass there, so that one column dominated the vector and
#  the end-rate metric became partly circular. C never had an end marker,
#  so B was being handed an advantage. Both are now 377-dim distributions
#  over real signs only, renormalised, on identical axes.
#
#  Requires: pip install umap-learn numpy matplotlib
#  Run in the same folder as the CSVs:  python indus_umap_v2.py
# =====================================================================
import numpy as np, matplotlib.pyplot as plt
import umap

MIN_COUNT    = 20      # rarer signs sit near random initialisation
FOCUS_SIGN   = "342"   # the jar, dominant text-ender. Change freely.
N_NEIGHBOURS = 10
N_SPECIALS   = 3       # <pad>, <bos>, <eos> occupy the first 3 columns

print(">>> indus_umap_v2.py  (specials dropped, axes matched)")

# ---------------------------------------------------------------------
# load
# ---------------------------------------------------------------------
meta = np.genfromtxt("sign_meta.csv", delimiter=",", names=True,
                     dtype=None, encoding=None)
signs  = [str(s) for s in meta["sign"]]
count  = np.array(meta["count"], dtype=int)
endr   = np.array(meta["end_rate"], dtype=float)
startr = np.array(meta["start_rate"], dtype=float)
NS     = len(signs)

emb = np.loadtxt("emb64.csv", delimiter=",")

# B: drop the special-token columns, renormalise over signs only
Bm = np.loadtxt("ctx_model.csv", delimiter=",")[:, N_SPECIALS:]
Bm = Bm / (Bm.sum(axis=1, keepdims=True) + 1e-12)

# C: successor half only, to match B
Cb = np.loadtxt("ctx_bigram.csv", delimiter=",")[:, :NS]

print(f"    shapes -- A {emb.shape}, B {Bm.shape}, C {Cb.shape}")

reps = {
    "A. raw embeddings (64d)":         emb,
    "B. model successor, signs only":  Bm,
    "C. bigram successor, signs only": Cb,
}

keep = count >= MIN_COUNT
print(f"{keep.sum()} of {NS} signs with count >= {MIN_COUNT} "
      f"({100*count[keep].sum()/count.sum():.1f}% of tokens)")
sk = [s for s, k in zip(signs, keep) if k]
ek, sr, ck = endr[keep], startr[keep], count[keep]
reps = {k: v[keep] for k, v in reps.items()}

if FOCUS_SIGN not in sk:
    raise SystemExit(f"focus sign {FOCUS_SIGN} not in the kept set")
fi = sk.index(FOCUS_SIGN)

def cosine_matrix(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T

# ---------------------------------------------------------------------
# neighbours of the focus sign
# ---------------------------------------------------------------------
print(f"\nNearest neighbours of sign {FOCUS_SIGN} "
      f"(end-rate {endr[signs.index(FOCUS_SIGN)]:.2f}), by cosine:\n")
for name, X in reps.items():
    Cm = cosine_matrix(X)
    order = [j for j in np.argsort(-Cm[fi]) if j != fi][:N_NEIGHBOURS]
    print(f"  {name}")
    print("    " + ", ".join(f"{sk[j]}({Cm[fi,j]:.2f},e={ek[j]:.2f})" for j in order))
    print(f"    mean end-rate of those {N_NEIGHBOURS}: "
          f"{np.mean([ek[j] for j in order]):.2f}   (corpus mean {ek.mean():.2f})\n")

# ---------------------------------------------------------------------
# role coherence: do a sign's nearest neighbours share its end-rate?
# ---------------------------------------------------------------------
print("Role coherence: mean |end-rate difference| to the 5 nearest")
print("neighbours. Lower = the space groups signs by position.")
print("Bracketed figure shuffles the end-rate labels, keeping neighbours fixed.\n")
rng = np.random.default_rng(0)
for name, X in reps.items():
    Cm = cosine_matrix(X)
    np.fill_diagonal(Cm, -np.inf)
    nn = np.argsort(-Cm, axis=1)[:, :5]
    got = np.mean(np.abs(ek[:, None] - ek[nn]))
    base = np.mean([np.mean(np.abs((p := rng.permutation(ek))[:, None] - p[nn]))
                    for _ in range(50)])
    print(f"  {name:<34} {got:.4f}   [{base:.4f}]")

# ---------------------------------------------------------------------
# UMAP panels, coloured by end-rate
# ---------------------------------------------------------------------
embs = {}
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
for ax, (name, X) in zip(axes, reps.items()):
    Y = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                  random_state=42).fit_transform(X)
    embs[name] = Y
    sc = ax.scatter(Y[:,0], Y[:,1], c=ek, cmap="coolwarm",
                    s=40 + 120*ck/ck.max(), vmin=0, vmax=1,
                    edgecolors="k", linewidths=0.3)
    for j, s in enumerate(sk):
        if ek[j] > 0.45 or sr[j] > 0.45 or s == FOCUS_SIGN:
            ax.annotate(s, Y[j], fontsize=7)
    ax.set_title(name, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=axes, label="end-rate (red = usually ends a text)",
             fraction=0.02, pad=0.01)
fig.suptitle(f"UMAP of the sign space, {keep.sum()} signs with count >= {MIN_COUNT}. "
             f"Point size = corpus frequency.", fontsize=12)
plt.savefig("umap_panels.png", dpi=150, bbox_inches="tight")
print("\nsaved umap_panels.png")

# ---------------------------------------------------------------------
# same layouts, coloured by similarity to the focus sign
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
for ax, (name, X) in zip(axes, reps.items()):
    Y = embs[name]
    sim = cosine_matrix(X)[fi]
    sc = ax.scatter(Y[:,0], Y[:,1], c=sim, cmap="viridis",
                    s=40 + 120*ck/ck.max(), edgecolors="k", linewidths=0.3)
    ax.scatter(*Y[fi], s=400, facecolors="none", edgecolors="red", linewidths=2.5)
    ax.annotate(FOCUS_SIGN, Y[fi], fontsize=11, color="red", weight="bold")
    for j in [t for t in np.argsort(-sim) if t != fi][:8]:
        ax.annotate(sk[j], Y[j], fontsize=7)
    ax.set_title(name, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=axes, label=f"cosine similarity to sign {FOCUS_SIGN}",
             fraction=0.02, pad=0.01)
fig.suptitle(f"Same layouts, coloured by similarity to sign {FOCUS_SIGN} "
             f"(circled). Its 8 nearest neighbours are labelled.", fontsize=12)
plt.savefig("umap_relative.png", dpi=150, bbox_inches="tight")
print("saved umap_relative.png")
print("\nB against C is the fair comparison: same axes, model vs raw counts.")
