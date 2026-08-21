# =====================================================================
#  indus_replicate.jl
#
#  Replicates the MODEL results of
#    Yadav, Joglekar, Rao, Vahia, Adhikari & Mahadevan,
#    "Statistical Analysis of the Indus Script Using n-Grams",
#    PLoS ONE 5(3): e9506 (2010)
#  using YOUR IndusGPT from indus_model_v2.jl, at your sweep's best
#  configuration: d_model 64, 4 heads, 3 layers, d_ff 256.
#
#    R1  Table 1 : perplexity vs context order, and the n=4 saturation
#    R2  Table 5 : sign-restoration sensitivity (paper: 74% +- 2)
#    R3  Fig 13  : most probable texts of length 4, 5, 6
#    R4  Fig 11  : the seven named restoration examples
#    R5  Table 4 : entropy and mutual information
#
#  FRESH Julia session. Do NOT include indus_train_v3.jl first --
#  it defines PADi/BOSi/EOSi/MAXLEN/V as consts and they will clash.
#
#      using Pkg; Pkg.activate(".")
#      include("indus_data.jl")
#      inscriptions = load_inscriptions()
#      include("indus_model_v2.jl")     # the patched one, with `window`
#      include("indus_replicate.jl")
#
#  ~25-40 min. Every line is timestamped. If 3 minutes pass with no new
#  line, kill it -- that is a hang, not slowness.
# =====================================================================
using Flux, MLUtils, Random, Statistics, Printf

const T0 = time()
say(s) = (println(@sprintf("[%6.1fs]", time() - T0), " ", s); flush(stdout))

# ---------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------
const MIN_COUNT = 1     # 1 = keep all 377 signs, matching the paper's V.
                        # Your sweep used 2 (V=268), which makes perplexity
                        # mechanically lower and NOT comparable to their 25.26.
const MAXL   = 20       # your MAXLEN
const DMODEL = 64       # your sweep's best config
const NHEADS = 4
const NLAYER = 3
const DFF    = 256
const PDROP  = 0.1
const LR     = 2f-4
const BATCH  = 32
const EPOCHS = 120
const PATIENCE = 12

# ---------------------------------------------------------------------
#  VOCAB
# ---------------------------------------------------------------------
let cnt = Dict{String,Int}()
    for ins in inscriptions, s in ins; cnt[s] = get(cnt, s, 0) + 1; end
    global SIGNS = sort([s for (s,c) in cnt if c >= MIN_COUNT])
end
const ITOS  = vcat(["<pad>", "<bos>", "<eos>", "<unk>"], SIGNS)
const STOI  = Dict(s => i for (i,s) in enumerate(ITOS))
const NV    = length(ITOS)
const PAD, BOS, EOS, UNK = STOI["<pad>"], STOI["<bos>"], STOI["<eos>"], STOI["<unk>"]
const SIGN_IDS = [STOI[s] for s in SIGNS]
const NSIGN = length(SIGNS)

enc(t) = [get(STOI, s, UNK) for s in t]

say("vocab: $NSIGN signs (min_count=$MIN_COUNT) + 4 specials = $NV tokens   [paper: 377 signs]")

# ---------------------------------------------------------------------
#  FIXED SPLIT -- every model below is scored on this same split,
#  so all comparisons are paired.
# ---------------------------------------------------------------------
Random.seed!(1234)
const PERM  = shuffle(1:length(inscriptions))
const NTEST = round(Int, 0.15 * length(PERM))
const TEST  = inscriptions[PERM[1:NTEST]]
const TRAIN = inscriptions[PERM[NTEST+1:end]]
say("split: $(length(TRAIN)) train / $(length(TEST)) test")

function build_XY(inss)
    ex = map(inss) do t
        ids = vcat(BOS, enc(t), EOS)
        length(ids) < MAXL+1 ? vcat(ids, fill(PAD, MAXL+1-length(ids))) : ids[1:MAXL+1]
    end
    hcat([e[1:end-1] for e in ex]...), hcat([e[2:end] for e in ex]...)
end
const XTR, YTR = build_XY(TRAIN)
const XTE, YTE = build_XY(TEST)

# ---------------------------------------------------------------------
#  loss / eval  (identical to indus_train_v3.jl, retargeted to NV/PAD)
# ---------------------------------------------------------------------
function loss_fn(m, x, y)
    logp = Flux.logsoftmax(m(x); dims=1)
    S, B = size(x)
    lp = reshape(logp, NV, S*B); y2 = reshape(y, S*B)
    msk = Float32.(y2 .!= PAD)
    nll = -vec(sum(Flux.onehotbatch(y2, 1:NV) .* lp; dims=1))
    sum(nll .* msk) / sum(msk)
end

function eval_nll(m, X, Y)
    Flux.testmode!(m)
    lp = reshape(Flux.logsoftmax(m(X); dims=1), NV, prod(size(X)))
    y2 = reshape(Y, prod(size(Y)))
    tot = 0.0; n = 0; tots = 0.0; ns = 0; corr = 0
    for t in 1:length(y2)
        yt = y2[t]; yt == PAD && continue
        tot += -lp[yt,t]; n += 1
        corr += (argmax(view(lp, :, t)) == yt)
        if yt != EOS; tots += -lp[yt,t]; ns += 1; end
    end
    Flux.trainmode!(m)
    (all = tot/n, signs = tots/ns, acc = 100corr/n)
end

function train_model(window; seed = 7, label = "")
    Random.seed!(seed)
    m = IndusGPT(vocab_size = NV, d_model = DMODEL, n_heads = NHEADS,
                 n_layers = NLAYER, d_ff = DFF, max_len = MAXL,
                 pdrop = PDROP, window = window)
    opt = Flux.setup(Adam(LR), m)
    loader = DataLoader((XTR, YTR); batchsize = BATCH, shuffle = true)
    best = Inf; bstate = deepcopy(Flux.state(m)); wait = 0; bep = 0
    for ep in 1:EPOCHS
        Flux.trainmode!(m)
        for (xb, yb) in loader
            _, gs = Flux.withgradient(mm -> loss_fn(mm, xb, yb), m)
            Flux.update!(opt, m, gs[1])
        end
        e = eval_nll(m, XTE, YTE)
        if e.all < best - 1f-4
            best = e.all; bstate = deepcopy(Flux.state(m)); wait = 0; bep = ep
        else
            wait += 1; wait >= PATIENCE && break
        end
        ep % 20 == 0 && say("    $label ep $ep  ppl=$(round(exp(e.all), digits=2))")
    end
    Flux.loadmodel!(m, bstate)
    m, bep
end

# =====================================================================
#  N-GRAMS WITH WITTEN-BELL SMOOTHING  (the paper's own method)
# =====================================================================
struct NGram
    order::Int
    cnt::Vector{Dict{Vector{String},Dict{String,Int}}}
end
function NGram(texts, order::Int)
    cnt = [Dict{Vector{String},Dict{String,Int}}() for _ in 1:order]
    for t in texts
        seq = vcat(fill("#", order-1), t, "\$")
        for i in order:length(seq), c in 0:(order-1)
            d = get!(cnt[c+1], String.(seq[i-c:i-1]), Dict{String,Int}())
            d[seq[i]] = get(d, seq[i], 0) + 1
        end
    end
    NGram(order, cnt)
end
function wb(m::NGram, w::AbstractString, ctx::Vector{String})
    c = length(ctx)
    if c == 0
        d = get(m.cnt[1], String[], nothing)
        d === nothing && return 1.0/NSIGN
        n = sum(values(d)); T = length(d)
        return (get(d, String(w), 0) + T*(1.0/NSIGN)) / (n + T)
    end
    lower = wb(m, w, ctx[2:end])
    d = get(m.cnt[c+1], ctx, nothing)
    d === nothing && return lower
    n = sum(values(d)); T = length(d)
    (get(d, String(w), 0) + T*lower) / (n + T)
end
function ngram_eval(m::NGram, texts)
    tot = 0.0; n = 0
    for t in texts
        seq = vcat(fill("#", m.order-1), t, "\$")
        for i in m.order:length(seq)
            p = wb(m, seq[i], String.(seq[max(1,i-m.order+1):i-1]))
            tot += -log(p); n += 1
        end
    end
    tot/n
end

# =====================================================================
#  R5 -- Table 4
# =====================================================================
say(""); say("="^70); say("R5  --  TABLE 4 : ENTROPY AND MUTUAL INFORMATION"); say("="^70)
let all_t = [s for t in inscriptions for s in t]
    M = length(all_t)
    fr = Dict{String,Int}(); for s in all_t; fr[s] = get(fr,s,0)+1; end
    H = -sum((c/M)*log2(c/M) for c in values(fr))
    bg = Dict{Tuple{String,String},Int}()
    for t in inscriptions, (a,b) in zip(t, t[2:end]); bg[(a,b)] = get(bg,(a,b),0)+1; end
    B = sum(values(bg))
    pa = Dict{String,Int}(); pb = Dict{String,Int}()
    for ((a,b),c) in bg; pa[a]=get(pa,a,0)+c; pb[b]=get(pb,b,0)+c; end
    I = sum((c/B)*log2((c/B)/((pa[a]/B)*(pb[b]/B))) for ((a,b),c) in bg)
    @printf("  %-34s %8.2f   [paper: 8.56]\n", "uniform 377-sign source", log2(length(fr)))
    @printf("  %-34s %8.2f   [paper: 6.68]\n", "EBUDS entropy", H)
    @printf("  %-34s %8.2f   [paper: 2.24]\n", "bigram mutual information", I)
    println("  (plug-in MI is upward-biased here; theirs came off the smoothed matrix)")
end

# =====================================================================
#  R1 -- Table 1
# =====================================================================
say(""); say("="^70); say("R1  --  TABLE 1 : PERPLEXITY VS CONTEXT ORDER"); say("="^70)
const PAPER_PPL = Dict(1=>68.82, 2=>26.69, 3=>26.09, 4=>25.26, 5=>25.26)
const PAPER_H   = Dict(1=>6.10,  2=>4.74,  3=>4.71,  4=>4.66,  5=>4.66)

say("n-gram baselines (Witten-Bell, same held-out split):")
for n in 1:5
    l = ngram_eval(NGram(TRAIN, n), TEST)
    @printf("  n=%d  ppl=%7.2f  H=%5.2f bits   [paper: %6.2f / %.2f]\n",
            n, exp(l), l/log(2), PAPER_PPL[n], PAPER_H[n])
end

say("")
say("YOUR IndusGPT ($DMODEL/$NLAYER/$DFF, $NHEADS heads) with windowed attention:")
tf_ppl = Dict{Int,Float64}(); tf_models = Dict{Int,Any}()
for k in [2, 3, 4, 5, MAXL]
    lbl = k == MAXL ? "unrestricted" : "window=$k"
    say("  training $lbl ...")
    m, bep = train_model(k; label = lbl)
    e = eval_nll(m, XTE, YTE)
    tf_ppl[k] = exp(e.all); tf_models[k] = m
    ref = haskey(PAPER_PPL, k) ? @sprintf("  [paper n=%d: %.2f]", k, PAPER_PPL[k]) : "  [vs paper n=5: 25.26]"
    @printf("  %-13s ppl=%7.2f  H=%5.2f bits  acc=%5.2f%%  best_ep=%3d%s\n",
            lbl, tf_ppl[k], e.all/log(2), e.acc, bep, ref)
end

say(""); say("SATURATION CHECK -- does extra context still help?")
let ks = [2,3,4,5,MAXL], nm = k -> (k == MAXL ? "unrestricted" : "window=$k")
    for i in 2:length(ks)
        @printf("  %-13s -> %-13s   Delta ppl = %+6.2f\n",
                nm(ks[i-1]), nm(ks[i]), tf_ppl[ks[i-1]] - tf_ppl[ks[i]])
    end
    println("  Paper's claim: n-gram perplexity saturates at n=4 (25.26 = 25.26 at n=5).")
    println("  Deltas near zero past window=4 confirm it by an independent method.")
    println("  A clear gain at unrestricted contradicts it -- that is the real finding.")
end
const BEST = tf_models[MAXL]

# =====================================================================
#  R2 -- Table 5 : restoration
#  Each candidate v is scored by the log-probability of the WHOLE
#  inscription with v spliced into the gap. Tokens after the gap change
#  their probabilities, so right-hand context enters even though
#  attention is causal -- the analogue of the bigram's P(v|left)P(right|v).
# =====================================================================
say(""); say("="^70); say("R2  --  TABLE 5 : SIGN RESTORATION   [paper: 74% +- 2]"); say("="^70)

function tf_posterior(m, ids::Vector{Int}, j::Int)
    B = NSIGN
    X = fill(PAD, MAXL, B); Y = fill(PAD, MAXL, B)
    for (b, v) in enumerate(SIGN_IDS)
        s = copy(ids); s[j] = v
        full = vcat(BOS, s, EOS)
        length(full) > MAXL+1 && (full = full[1:MAXL+1])
        n = length(full) - 1
        X[1:n, b] = full[1:end-1]; Y[1:n, b] = full[2:end]
    end
    Flux.testmode!(m)
    lp = reshape(Flux.logsoftmax(m(X); dims=1), NV, MAXL*B)
    Flux.trainmode!(m)
    y2 = reshape(Y, MAXL*B); out = zeros(Float64, B)
    for t in 1:length(y2)
        yt = y2[t]; yt == PAD && continue
        out[div(t-1, MAXL)+1] += Float64(lp[yt, t])
    end
    out
end

const BIGRAM = NGram(TRAIN, 2)
function bg_posterior(t, j)
    left  = j > 1         ? String[t[j-1]] : String["#"]
    right = j < length(t) ? String(t[j+1]) : "\$"
    [log(wb(BIGRAM, w, left)) + log(wb(BIGRAM, right, String[w])) for w in SIGNS]
end

function nucleus(sc::Vector{Float64}, truth::Int; mass = 0.90)
    p = exp.(sc .- maximum(sc)); p ./= sum(p)
    ord = sortperm(p; rev=true)
    cum = 0.0; k = length(ord)
    for (i,idx) in enumerate(ord); cum += p[idx]; cum >= mass && (k = i; break); end
    r = findfirst(==(truth), ord)
    (r <= k, r, k)
end

let usable = [t for t in TEST if length(t) >= 2]
    th=0; tt1=0; tt5=0; tk=0.0; bh=0; bt1=0; bt5=0; bk=0.0; n=0
    rng = MersenneTwister(99)
    for (i,t) in enumerate(usable)
        j = rand(rng, 1:length(t))
        truth = findfirst(==(t[j]), SIGNS)
        truth === nothing && continue
        h1,r1,k1 = nucleus(tf_posterior(BEST, enc(t), j), truth)
        h2,r2,k2 = nucleus(bg_posterior(t, j), truth)
        th+=h1; tt1+=(r1==1); tt5+=(r1<=5); tk+=k1
        bh+=h2; bt1+=(r2==1); bt5+=(r2<=5); bk+=k2
        n += 1
        i % 50 == 0 && say("    restored $i / $(length(usable))")
    end
    println()
    @printf("  %-14s %12s %10s %10s %12s\n", "model", "sensitivity", "top-1", "top-5", "set size")
    @printf("  %-14s %11.1f%% %9.1f%% %9.1f%% %12.1f\n", "IndusGPT",    100th/n, 100tt1/n, 100tt5/n, tk/n)
    @printf("  %-14s %11.1f%% %9.1f%% %9.1f%% %12.1f\n", "Bigram (WB)", 100bh/n, 100bt1/n, 100bt5/n, bk/n)
    @printf("  %-14s %11s  %9.2f%%\n", "Random", "-", 100/NSIGN)
    println("  paper's bigram: 74% +- 2    ($n restorations, one per test text)")
    println("  NOTE: sensitivity is lenient -- truth need only land somewhere in a set")
    println("        of ~$(round(Int, bk/n)) candidates. Always quote top-1 beside it.")
end

# =====================================================================
#  R3 -- Fig 13 : most probable texts (beam search; 377^L is infeasible)
# =====================================================================
say(""); say("="^70); say("R3  --  FIG 13 : MOST PROBABLE TEXTS"); say("="^70)

function best_text(m, L::Int; beam = 120)
    beams = [(Int[], 0.0)]
    for _ in 1:L
        X = fill(PAD, MAXL, length(beams))
        for (b,(sq,_)) in enumerate(beams)
            pre = vcat(BOS, sq); X[1:length(pre), b] = pre
        end
        Flux.testmode!(m); lp = Flux.logsoftmax(m(X); dims=1); Flux.trainmode!(m)
        cand = Tuple{Vector{Int},Float64}[]
        for (b,(sq,sc)) in enumerate(beams), v in SIGN_IDS
            push!(cand, (vcat(sq, v), sc + Float64(lp[v, length(sq)+1, b])))
        end
        sort!(cand; by = x -> -x[2])
        beams = cand[1:min(beam, length(cand))]
    end
    X = fill(PAD, MAXL, length(beams))
    for (b,(sq,_)) in enumerate(beams)
        pre = vcat(BOS, sq); X[1:length(pre), b] = pre
    end
    Flux.testmode!(m); lp = Flux.logsoftmax(m(X); dims=1); Flux.trainmode!(m)
    sc = [(sq, s + Float64(lp[EOS, length(sq)+1, b])) for (b,(sq,s)) in enumerate(beams)]
    sort!(sc; by = x -> -x[2]); first(sc)
end

const CORPUS_SET = Set(Tuple.(inscriptions))
# Fig 13 targets, converted from the paper's printed (glyph-aligned) order
# into the reading order used by induscorpus.txt
const FIG13 = Dict(4 => reverse(["342","48","99","267"]),
                   5 => reverse(["342","8","171","99","267"]),
                   6 => reverse(["342","8","171","53","99","267"]))
for L in 4:6
    sq, sc = best_text(BEST, L)
    txt = [ITOS[i] for i in sq]
    @printf("  length %d\n", L)
    @printf("    IndusGPT : %-30s logp=%7.2f  in corpus: %s\n",
            join(txt," "), sc, Tuple(txt) in CORPUS_SET ? "YES" : "no")
    @printf("    paper    : %-30s %s\n",
            join(FIG13[L]," "), join(txt," ") == join(FIG13[L]," ") ? "<-- MATCH" : "")
end
println("  (paper found its length-4 and length-5 texts verbatim in the corpus;")
println("   its length-6 text occurs only as a variant with two insertions)")

# =====================================================================
#  R4 -- Fig 11 : the seven named restoration examples
#  Stored as the paper prints them, with the index of the deleted sign
#  in that same printed order; both are reversed to reading order below.
# =====================================================================
say(""); say("="^70); say("R4  --  FIG 11 : THE SEVEN NAMED EXAMPLES"); say("="^70)
const FIG11 = [
    ("4312", ["342","48","53","70"],                        2),
    ("4016", ["342","327","70","67","53","97","391"],       4),
    ("5237", ["342","135","67","99","267"],                 5),
    ("2653", ["342","347","127","48","99","267"],           2),
    ("5073", ["342","244","67","99","130","51","364"],      2),
    ("3360", ["169","87","211","18","194","112","59"],      5),
    ("9071", ["342","293","182","72","67","98","99","267"], 7),
]
@printf("  %-6s %-30s %-7s %-11s %-11s %s\n",
        "text", "inscription (reading order)", "deleted", "IndusGPT", "bigram", "rank t/b")
let tf_ok = 0, bg_ok = 0
    for (id, printed, pidx) in FIG11
        t = reverse(printed)
        j = length(printed) + 1 - pidx
        if !(Tuple(t) in CORPUS_SET)
            @printf("  %-6s NOT FOUND IN CORPUS\n", id); continue
        end
        truth = findfirst(==(t[j]), SIGNS)
        st = tf_posterior(BEST, enc(t), j); sb = bg_posterior(t, j)
        pt = SIGNS[argmax(st)]; pb = SIGNS[argmax(sb)]
        rt = findfirst(==(truth), sortperm(st; rev=true))
        rb = findfirst(==(truth), sortperm(sb; rev=true))
        tf_ok += (pt == t[j]); bg_ok += (pb == t[j])
        @printf("  %-6s %-30s %-7s %-11s %-11s %d / %d\n", id, join(t," "), t[j],
                pt == t[j] ? "$pt OK" : pt, pb == t[j] ? "$pb OK" : pb, rt, rb)
    end
    @printf("\n  IndusGPT restored %d/%d exactly;  bigram %d/%d   (paper: 7/7)\n",
            tf_ok, length(FIG11), bg_ok, length(FIG11))
end

say(""); say("="^70); say("DONE"); say("="^70)
