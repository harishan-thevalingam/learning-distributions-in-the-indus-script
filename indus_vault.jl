# =====================================================================
#  indus_vault.jl
#
#  Train the model ONCE, save it, reuse it everywhere.
#
#  Every analysis script so far retrained from scratch (~5 min each),
#  which is wasteful and, worse, means each analysis is looking at a
#  slightly different model. This saves the weights, the vocabulary and
#  the train/test split together, so everything downstream sees exactly
#  the same model and the same held-out set.
#
#  Defines FUNCTIONS ONLY -- no consts, no globals -- so it can be
#  re-included in a live session without redefinition errors.
#
#  USAGE
#      using Pkg; Pkg.activate(".")
#      include("indus_data.jl")
#      inscriptions = load_inscriptions()
#      include("indus_model_v2.jl")
#      include("indus_vault.jl")
#
#      M = indus_get(inscriptions)          # trains + saves first time,
#                                           # loads instantly thereafter
#
#  M is a NamedTuple:
#      M.model              the trained IndusGPT
#      M.itos, M.stoi       vocabulary maps
#      M.signs              real sign strings, no specials
#      M.pad, M.bos, M.eos  special token ids
#      M.nv, M.maxl         vocab size, block size
#      M.train, M.test      the inscription split
#      M.ppl                held-out perplexity of the saved model
#      M.epoch              epoch the saved weights came from
#      M.encode(t)          sign strings -> token ids
#
#  To force a fresh run (e.g. after changing architecture):
#      M = indus_get(inscriptions; force = true)
#
#  Different configurations can coexist in different files:
#      M16 = indus_get(inscriptions; path = "indus_model_w16.jld2", window = 16)
# =====================================================================
using Flux, MLUtils, Random, Statistics, Printf, JLD2

# ---------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------
function indus_vocab(inscriptions; min_count = 1)
    cnt = Dict{String,Int}()
    for ins in inscriptions, s in ins
        cnt[s] = get(cnt, s, 0) + 1
    end
    signs = sort([s for (s,c) in cnt if c >= min_count]; by = x -> parse(Int, x))
    # <unk> only when min_count > 1 can actually produce unknown signs.
    # Adding it unconditionally shifts every token index and changes the model.
    specials = min_count > 1 ? ["<pad>", "<bos>", "<eos>", "<unk>"] : ["<pad>", "<bos>", "<eos>"]
    itos  = vcat(specials, signs)
    stoi  = Dict(s => i for (i,s) in enumerate(itos))
    (signs = signs, itos = itos, stoi = stoi, counts = cnt)
end

# ---------------------------------------------------------------------
# split -- seed 1234, 15% test, matching every earlier script
# ---------------------------------------------------------------------
function indus_split(inscriptions; seed = 1234, frac = 0.15)
    # MUST use the global RNG via Random.seed!, exactly as indus_replicate.jl,
    # indus_probe_v3.jl and indus_reorder_test.jl do. Julia's global RNG is
    # Xoshiro, so shuffle(MersenneTwister(1234), ...) gives a DIFFERENT split
    # and therefore a different model (28.64 ppl instead of 24.43).
    Random.seed!(seed)
    perm  = shuffle(1:length(inscriptions))
    ntest = round(Int, frac * length(perm))
    (train = inscriptions[perm[ntest+1:end]],
     test  = inscriptions[perm[1:ntest]],
     perm  = perm, ntest = ntest)
end

function indus_XY(inss, stoi, maxl, pad, bos, eos, unk)
    ex = map(inss) do t
        ids = vcat(bos, [get(stoi, s, unk) for s in t], eos)
        length(ids) < maxl + 1 ? vcat(ids, fill(pad, maxl + 1 - length(ids))) : ids[1:maxl+1]
    end
    hcat([e[1:end-1] for e in ex]...), hcat([e[2:end] for e in ex]...)
end

# ---------------------------------------------------------------------
# loss / eval
# ---------------------------------------------------------------------
function indus_loss(m, x, y, nv, pad)
    logp = Flux.logsoftmax(m(x); dims = 1)
    S, B = size(x)
    lp  = reshape(logp, nv, S*B)
    y2  = reshape(y, S*B)
    msk = Float32.(y2 .!= pad)
    nll = -vec(sum(Flux.onehotbatch(y2, 1:nv) .* lp; dims = 1))
    sum(nll .* msk) / sum(msk)
end

function indus_nll(m, X, Y, nv, pad)
    Flux.testmode!(m)
    lp = reshape(Flux.logsoftmax(m(X); dims = 1), nv, prod(size(X)))
    y2 = reshape(Y, prod(size(Y)))
    tot = 0.0; n = 0
    for t in 1:length(y2)
        yt = y2[t]; yt == pad && continue
        tot += -lp[yt, t]; n += 1
    end
    Flux.trainmode!(m)
    tot / n
end

# ---------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------
"""
    indus_get(inscriptions; path="indus_model.jld2", force=false, kwargs...)

Load the saved model if `path` exists, otherwise train one and save it.
Returns a NamedTuple (see header). Pass `force=true` to retrain.
"""
function indus_get(inscriptions;
                   path      = "indus_model.jld2",
                   force     = false,
                   min_count = 1,
                   maxl      = 20,
                   d_model   = 64,
                   n_heads   = 4,
                   n_layers  = 3,
                   d_ff      = 256,
                   pdrop     = 0.1,
                   window    = nothing,      # defaults to maxl (unrestricted)
                   lr        = 2f-4,
                   batch     = 32,
                   epochs    = 120,
                   patience  = 12,
                   seed      = 7)

    win = window === nothing ? maxl : window

    V   = indus_vocab(inscriptions; min_count = min_count)
    sp  = indus_split(inscriptions)
    nv  = length(V.itos)
    pad, bos, eos = V.stoi["<pad>"], V.stoi["<bos>"], V.stoi["<eos>"]
    unk = get(V.stoi, "<unk>", 0)
    enc(t) = [get(V.stoi, s, unk) for s in t]

    mk() = IndusGPT(vocab_size = nv, d_model = d_model, n_heads = n_heads,
                    n_layers = n_layers, d_ff = d_ff, max_len = maxl,
                    pdrop = pdrop, window = win)

    pack(model, ppl, epoch) = (model = model, itos = V.itos, stoi = V.stoi,
                               signs = V.signs, counts = V.counts,
                               pad = pad, bos = bos, eos = eos, unk = unk,
                               nv = nv, maxl = maxl, window = win,
                               train = sp.train, test = sp.test,
                               ppl = ppl, epoch = epoch, encode = enc)

    # ---------------- load ----------------
    if isfile(path) && !force
        d = JLD2.load(path)
        if d["itos"] != V.itos || d["maxl"] != maxl || d["window"] != win ||
           d["d_model"] != d_model || d["n_layers"] != n_layers ||
           d["n_heads"] != n_heads || d["d_ff"] != d_ff
            @warn """Saved model in "$path" does not match the requested configuration.
                     Retraining. Use a different `path` to keep both."""
        else
            m = mk()
            Flux.loadmodel!(m, d["state"])
            Flux.testmode!(m)
            @printf("loaded %s  (held-out ppl %.2f, from epoch %d)\n", path, d["ppl"], d["epoch"])
            return pack(m, d["ppl"], d["epoch"])
        end
    end

    # ---------------- train ----------------
    println("training ($(d_model)/$(n_layers)/$(d_ff), $(n_heads) heads, window $(win)) ...")
    flush(stdout)
    Xtr, Ytr = indus_XY(sp.train, V.stoi, maxl, pad, bos, eos, unk)
    Xte, Yte = indus_XY(sp.test,  V.stoi, maxl, pad, bos, eos, unk)

    Random.seed!(seed)
    m   = mk()
    opt = Flux.setup(Adam(lr), m)
    loader = DataLoader((Xtr, Ytr); batchsize = batch, shuffle = true)

    best = Inf; bstate = deepcopy(Flux.state(m)); wait = 0; bep = 0
    t0 = time()
    for ep in 1:epochs
        Flux.trainmode!(m)
        for (xb, yb) in loader
            _, gs = Flux.withgradient(mm -> indus_loss(mm, xb, yb, nv, pad), m)
            Flux.update!(opt, m, gs[1])
        end
        e = indus_nll(m, Xte, Yte, nv, pad)
        if e < best - 1f-4
            best = e; bstate = deepcopy(Flux.state(m)); wait = 0; bep = ep
        else
            wait += 1
            wait >= patience && break
        end
        if ep % 20 == 0
            @printf("  ep %3d  held-out ppl %.2f  (%.0fs)\n", ep, exp(e), time()-t0)
            flush(stdout)
        end
    end
    Flux.loadmodel!(m, bstate)
    Flux.testmode!(m)
    ppl = exp(best)
    @printf("done: held-out ppl %.2f at epoch %d (%.0fs)\n", ppl, bep, time()-t0)

    JLD2.jldsave(path;
        state = bstate, itos = V.itos, maxl = maxl, window = win,
        d_model = d_model, n_layers = n_layers, n_heads = n_heads, d_ff = d_ff,
        min_count = min_count, ppl = ppl, epoch = bep, seed = seed)
    println("saved -> $path")

    pack(m, ppl, bep)
end

"""
    indus_loglik(M, texts)

Log-likelihood of each inscription under the saved model, in nats.
`texts` is a vector of sign-string vectors.
"""
function indus_loglik(M, texts)
    B = length(texts)
    X = fill(M.pad, M.maxl, B); Y = fill(M.pad, M.maxl, B)
    for (b, t) in enumerate(texts)
        full = vcat(M.bos, M.encode(t), M.eos)
        length(full) > M.maxl + 1 && (full = full[1:M.maxl+1])
        n = length(full) - 1
        X[1:n, b] = full[1:end-1]; Y[1:n, b] = full[2:end]
    end
    Flux.testmode!(M.model)
    lp = reshape(Flux.logsoftmax(M.model(X); dims = 1), M.nv, M.maxl * B)
    y2 = reshape(Y, M.maxl * B)
    out = zeros(Float64, B)
    for t in 1:length(y2)
        yt = y2[t]; yt == M.pad && continue
        out[div(t-1, M.maxl) + 1] += Float64(lp[yt, t])
    end
    out
end

println("indus_vault.jl loaded.  Call:  M = indus_get(inscriptions)")
