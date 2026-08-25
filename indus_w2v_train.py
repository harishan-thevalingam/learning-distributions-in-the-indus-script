# =====================================================================
# indus_w2v_train.py
#
# Independent embedding baseline for the Indus sign corpus, to check
# whether the earlier embedding-space null (raw 64d transformer
# embeddings at chance on the 5-NN role-coherence metric, 0.2161
# [0.2212]) is a fact about the transformer or a fact about having
# ~7,000 tokens over 377 sign types (~19 tokens/type -- well below
# where word2vec is usually reliable).
#
# 5 configs x 10 seeds (0-9) = 50 models. vector_size=24, not 64, per
# Mikolov's own guidance against high dims on sparse/small data.
#
# CONFIG CHOICES -- not fully specified in the carried-over summary,
# this is my reconstruction. Check before trusting downstream numbers:
#   cbow_w2  : CBOW,      window=2, negative=10
#   cbow_w5  : CBOW,      window=5, negative=10
#   sg_w2    : skip-gram, window=2, negative=10
#   sg_w5    : skip-gram, window=5, negative=10
#   sg_w2_hs : skip-gram, window=2, hierarchical softmax (no negative sampling)
# Rationale: CBOW-vs-skip-gram and window=2-vs-5 (mean inscription
# length is ~4.5 signs, so window=5 already covers nearly a whole
# inscription) are the two axes most likely to matter at this corpus
# size. sg_w2_hs swaps the training objective entirely, since
# hierarchical softmax is the standard recommendation for rare-word
# quality on tiny corpora, whereas the other four all use negative
# sampling. Skip-gram (3 of the 5 configs) is the one to trust given
# corpus size -- it gives each occurrence of a rare sign several
# gradient updates per context word, where CBOW averages context to
# predict one target and dilutes exactly the low-frequency signs this
# corpus is full of. CBOW is kept as an empirical control, not because
# it's expected to win.
#
# sample=0 (subsampling disabled): unlike natural-language corpora,
# every sign here is a content token -- there's no stopword class to
# down-weight, and with only 377 types, subsampling would remove
# signal disproportionately from the most frequent (and probably most
# functionally important) signs, e.g. 342, 267, 99.
#
# epochs=50 (gensim default is 5): the corpus is tiny, so more passes
# are cheap and necessary for the vectors to settle.
#
# workers=1: gensim's multi-threaded training is not deterministic
# across workers even with a fixed seed. For full reproducibility of
# vocabulary hashing too (not just training), set PYTHONHASHSEED=0 in
# the environment before running -- not essential here since the
# vocab is small and every sign is kept (min_count=1), but worth
# knowing if you rerun and expect bit-identical models.
#
# Requires: pip install gensim
# Run in the same folder as induscorpus.txt: python indus_w2v_train.py
# =====================================================================

import csv
import os
from gensim.models import Word2Vec

CORPUS_PATH = "induscorpus.txt"
OUT_DIR = "w2v_models"
VECTOR_SIZE = 24
EPOCHS = 50
SEEDS = list(range(10))
MIN_COUNT = 1  # keep every sign, including hapax -- matches the
               # min_count=1 / V=381 transformer runs, and hapax
               # behaviour is exactly what Section 1 (seed stability)
               # is designed to check

CONFIGS = [
    dict(name="cbow_w2",  sg=0, window=2, negative=10, hs=0),
    dict(name="cbow_w5",  sg=0, window=5, negative=10, hs=0),
    dict(name="sg_w2",    sg=1, window=2, negative=10, hs=0),
    dict(name="sg_w5",    sg=1, window=5, negative=10, hs=0),
    dict(name="sg_w2_hs", sg=1, window=2, negative=0,  hs=1),
]


def load_inscriptions(path=CORPUS_PATH):
    """Mirrors indus_data.jl's load_inscriptions exactly: one
    inscription per line, whitespace-separated sign numbers, kept as
    strings (they're tokens, not numbers)."""
    inscriptions = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                inscriptions.append(line.split())
    return inscriptions


def main():
    inscriptions = load_inscriptions()
    n_tokens = sum(len(ins) for ins in inscriptions)
    n_types = len({s for ins in inscriptions for s in ins})
    print(f"{len(inscriptions)} inscriptions, {n_tokens} tokens, "
          f"{n_types} sign types ({n_tokens / n_types:.1f} tokens/type)")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    rows = []

    for cfg in CONFIGS:
        for seed in SEEDS:
            model = Word2Vec(
                sentences=inscriptions,
                vector_size=VECTOR_SIZE,
                window=cfg["window"],
                sg=cfg["sg"],
                hs=cfg["hs"],
                negative=cfg["negative"],
                sample=0,
                min_count=MIN_COUNT,
                epochs=EPOCHS,
                seed=seed,
                workers=1,
            )
            out_path = os.path.join(OUT_DIR, f"{cfg['name']}_seed{seed}.model")
            model.save(out_path)
            rows.append(dict(
                config=cfg["name"], seed=seed, sg=cfg["sg"],
                window=cfg["window"], negative=cfg["negative"],
                hs=cfg["hs"], vector_size=VECTOR_SIZE, epochs=EPOCHS,
                min_count=MIN_COUNT, path=out_path,
            ))
            print(f"  {cfg['name']} seed {seed}: saved -> {out_path}")

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} models trained. manifest -> {manifest_path}")
    print("Now run: python indus_w2v_analyse.py")


if __name__ == "__main__":
    main()