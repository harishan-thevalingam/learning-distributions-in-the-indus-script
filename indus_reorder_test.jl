# =====================================================================
#  indus_reorder_test.jl
#
#  PNAS 2009 (Rao et al.), the reordering claim, evaluated with the
#  TRANSFORMER instead of a bigram.
#
#  The paper's argument that sign order is rule-governed: take a real
#  inscription, reorder it, and its likelihood under the learned model
#  collapses. Their SI example moves the LAST sign to the FRONT and
#  reports a drop from 8.0e-6 to 5.2e-10, i.e. about 15,000-fold.
#
#  Our earlier run of this used a bigram, which only reproduced their
#  model. This scores everything under YOUR IndusGPT, so the comparison
#  is our transformer against their published bigram result.
#
#  Three perturbations, all on held-out inscriptions only:
#    (a) last sign moved to the front   <- the paper's own perturbation
#    (b) one random adjacent swap        <- smallest possible change
#    (c) full random shuffle             <- upper bound on the effect
#
#  Training protocol identical to indus_probe_v3.jl: same 85/15 split
#  (seed 1234), early stopping, 64 / 3 layers / 4 heads / 256,
#  unrestricted attention.
#
#  All globals suffixed _SH so nothing clashes with other scripts.
#
#      using Pkg; Pkg.activate(".")
#      include("indus_data.jl")
#      inscriptions = load_inscriptions()
#      include("indus_model_v2.jl")
#      include("indus_reorder_test.jl")
#
#  Runtime ~5 min (one training run; scoring is seconds).
# =====================================================================
using Flux, MLUtils, Random, Statistics, Printf

const T0_SH = time()
says(s) = (println(@sprintf("[%6.1fs]", time()-T0_SH), " ", s); flush(stdout))

const MAXL_SH   = 20
const DMODEL_SH = 64
const NHEADS_SH = 4
const NLAYER_SH = 3
const DFF_SH    = 256
const PDROP_SH  = 0.1
const LR_SH     = 2f-4
const BATCH_SH  = 32
const EPOCHS_SH = 120
const PATIENCE_SH = 12

# ---------------------------------------------------------------------
# vocab (min_count = 1, all 377 signs real tokens)
# ---------------------------------------------------------------------
let cnt = Dict{String,Int}()
    for ins in inscriptions, s in ins; cnt[s] = get(cnt, s, 0) + 1; end
    global SIGNS_SH = sort(collect(keys(cnt)); by = x -> parse(Int, x))
end
const ITOS_SH = vcat(["<pad>","<bos>","<eos>"], SIGNS_SH)
const STOI_SH = Dict(s => i for (i,s) in enumerate(ITOS_SH))
const NV_SH   = length(ITOS_SH)
const PAD_SH, BOS_SH, EOS_SH = STOI_SH["<pad>"], STOI_SH["<bos>"], STOI_SH["<eos>"]
encs(t) = [STOI_SH[s] for s in t]
says("vocab: $(length(SIGNS_SH)) signs + 3 specials = $NV_SH tokens")

# ---------------------------------------------------------------------
# split -- identical to the other scripts
# ---------------------------------------------------------------------
Random.seed!(1234)
const PERM_SH  = shuffle(1:length(inscriptions))
const NTEST_SH = round(Int, 0.15 * length(PERM_SH))
const TEST_SH  = inscriptions[PERM_SH[1:NTEST_SH]]
const TRAIN_SH = inscriptions[PERM_SH[NTEST_SH+1:end]]
says("split: $(length(TRAIN_SH)) train / $(length(TEST_SH)) test")

function buildXY_SH(inss)
    ex = map(inss) do t
        ids = vcat(BOS_SH, encs(t), EOS_SH)
        length(ids) < MAXL_SH+1 ? vcat(ids, fill(PAD_SH, MAXL_SH+1-length(ids))) : ids[1:MAXL_SH+1]
    end
    hcat([e[1:end-1] for e in ex]...), hcat([e[2:end] for e in ex]...)
end
const XTR_SH, YTR_SH = buildXY_SH(TRAIN_SH)
const XTE_SH, YTE_SH = buildXY_SH(TEST_SH)

function loss_SH(m, x, y)
    logp = Flux.logsoftmax(m(x); dims=1)
    S, B = size(x)
    lp = reshape(logp, NV_SH, S*B); y2 = reshape(y, S*B)
    msk = Float32.(y2 .!= PAD_SH)
    nll = -vec(sum(Flux.onehotbatch(y2, 1:NV_SH) .* lp; dims=1))
    sum(nll .* msk) / sum(msk)
end
function evalnll_SH(m, X, Y)
    Flux.testmode!(m)
    lp = reshape(Flux.logsoftmax(m(X); dims=1), NV_SH, prod(size(X)))
    y2 = reshape(Y, prod(size(Y)))
    tot = 0.0; n = 0
    for t in 1:length(y2)
        yt = y2[t]; yt == PAD_SH && continue
        tot += -lp[yt,t]; n += 1
    end
    Flux.trainmode!(m)
    tot/n
end

# ---------------------------------------------------------------------
# train
# ---------------------------------------------------------------------
says("training IndusGPT ($DMODEL_SH/$NLAYER_SH/$DFF_SH, $NHEADS_SH heads) ...")
Random.seed!(7)
model_SH = IndusGPT(vocab_size = NV_SH, d_model = DMODEL_SH, n_heads = NHEADS_SH,
                    n_layers = NLAYER_SH, d_ff = DFF_SH, max_len = MAXL_SH,
                    pdrop = PDROP_SH, window = MAXL_SH)
let opt = Flux.setup(Adam(LR_SH), model_SH),
    loader = DataLoader((XTR_SH, YTR_SH); batchsize = BATCH_SH, shuffle = true),
    best = Inf, bstate = deepcopy(Flux.state(model_SH)), wait = 0, bep = 0
    for ep in 1:EPOCHS_SH
        Flux.trainmode!(model_SH)
        for (xb, yb) in loader
            _, gs = Flux.withgradient(mm -> loss_SH(mm, xb, yb), model_SH)
            Flux.update!(opt, model_SH, gs[1])
        end
        e = evalnll_SH(model_SH, XTE_SH, YTE_SH)
        if e < best - 1f-4
            best = e; bstate = deepcopy(Flux.state(model_SH)); wait = 0; bep = ep
        else
            wait += 1; wait >= PATIENCE_SH && break
        end
        ep % 20 == 0 && says("   ep $ep  held-out ppl $(round(exp(e), digits=2))")
    end
    Flux.loadmodel!(model_SH, bstate)
    says("best held-out ppl $(round(exp(best), digits=2)) at epoch $bep")
end
Flux.testmode!(model_SH)

# ---------------------------------------------------------------------
# sequence log-likelihood under the transformer
#   log P(inscription) = sum over positions of log p(token | everything before)
#   includes the end token, so length is accounted for
# ---------------------------------------------------------------------
"Batched: takes a vector of sign-string vectors, returns their log-likelihoods."
function loglik_batch_SH(m, texts)
    B = length(texts)
    X = fill(PAD_SH, MAXL_SH, B); Y = fill(PAD_SH, MAXL_SH, B)
    for (b, t) in enumerate(texts)
        full = vcat(BOS_SH, encs(t), EOS_SH)
        length(full) > MAXL_SH+1 && (full = full[1:MAXL_SH+1])
        n = length(full) - 1
        X[1:n, b] = full[1:end-1]; Y[1:n, b] = full[2:end]
    end
    lp = reshape(Flux.logsoftmax(m(X); dims=1), NV_SH, MAXL_SH*B)
    y2 = reshape(Y, MAXL_SH*B)
    out = zeros(Float64, B)
    for t in 1:length(y2)
        yt = y2[t]; yt == PAD_SH && continue
        out[div(t-1, MAXL_SH)+1] += Float64(lp[yt, t])
    end
    out
end

# ---------------------------------------------------------------------
# the three perturbations
# ---------------------------------------------------------------------
move_last_to_front_SH(t) = vcat([t[end]], t[1:end-1])        # the paper's own

function adjacent_swap_SH(t, rng)
    s = copy(t)
    i = rand(rng, 1:length(s)-1)
    s[i], s[i+1] = s[i+1], s[i]
    s
end

full_shuffle_SH(t, rng) = shuffle(rng, copy(t))

# ---------------------------------------------------------------------
# run
# ---------------------------------------------------------------------
says("")
says("="^70)
says("REORDERING TEST -- TRANSFORMER ONLY")
says("="^70)

let usable = [t for t in TEST_SH if length(t) >= 3]
    rng = MersenneTwister(2024)
    orig = loglik_batch_SH(model_SH, usable)
    pert = Dict(
        "last sign moved to front" => loglik_batch_SH(model_SH, [move_last_to_front_SH(t) for t in usable]),
        "one adjacent swap"        => loglik_batch_SH(model_SH, [adjacent_swap_SH(t, rng) for t in usable]),
        "full random shuffle"      => loglik_batch_SH(model_SH, [full_shuffle_SH(t, rng) for t in usable]),
    )
    println()
    @printf("  held-out inscriptions of length >= 3: %d\n\n", length(usable))
    @printf("  %-26s %14s %14s %14s\n", "perturbation", "median drop", "mean drop", "% made worse")
    @printf("  %-26s %14s %14s %14s\n", "", "(fold)", "(fold)", "")
    for k in ["last sign moved to front", "one adjacent swap", "full random shuffle"]
        d = orig .- pert[k]                       # log-likelihood lost, in nats
        med = exp(median(d)); mn = exp(mean(d))
        worse = 100 * count(>(0), d) / length(d)
        @printf("  %-26s %14.1f %14.1f %13.1f%%\n", k, med, mn, worse)
    end
    println()
    println("  'drop' = how many times more likely the original is than the reordered")
    println("  version, under the transformer. 'made worse' = fraction of inscriptions")
    println("  where reordering reduced the likelihood at all.")
    println()
    println("  PAPER'S REFERENCE POINT (their bigram, their SI example):")
    println("    last sign moved to front, 8.0e-6 -> 5.2e-10, about 15,000-fold")
    println()
    println("  The first row is the like-for-like comparison. The other two bracket it:")
    println("  an adjacent swap is the smallest possible reordering, a full shuffle the")
    println("  largest, so the effect size should increase down the table.")

    # a worked example, to mirror the paper's SI figure
    println()
    println("  WORKED EXAMPLE (mirrors their SI Fig. S1):")
    i = argmax(length.(usable))
    t = usable[i]
    tp = move_last_to_front_SH(t)
    lo = loglik_batch_SH(model_SH, [t])[1]
    lp2 = loglik_batch_SH(model_SH, [tp])[1]
    @printf("    original : %-34s  likelihood %.2e\n", join(t, " "), exp(lo))
    @printf("    reordered: %-34s  likelihood %.2e\n", join(tp, " "), exp(lp2))
    @printf("    ratio    : %.0f-fold\n", exp(lo - lp2))
end

says("")
says("="^70)
says("DONE")
says("="^70)
