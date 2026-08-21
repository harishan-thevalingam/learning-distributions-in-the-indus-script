# =====================================================================
#  indus_circuits_control.jl
#
#  The control that makes indus_circuits.jl interpretable.
#
#  That run gave: composition median 0.1254 against a random-matrix
#  reference of 1/sqrt(64) = 0.125; QK effective rank 14.8-15.0 out of a
#  ceiling of 16; embedding effective rank 54.6 of 64 with a nearly flat
#  singular spectrum. All of that reads as "unstructured" -- but a
#  RANDOMLY INITIALISED model produces those same numbers by
#  construction. So the trained figures only mean something against an
#  explicit null.
#
#  THREE MODELS, identical architecture:
#    TRAINED   the saved model (held-out ppl 24.43)
#    RANDOM    same architecture, never trained
#    SHUFFLED  trained on sign-shuffled inscriptions: same unigram
#              statistics, sequential structure destroyed. This is the
#              stronger null -- it shows what training on genuinely
#              structureless data does, separating "training changes
#              nothing" from "there was nothing to learn".
#
#  If TRAINED sits between RANDOM and SHUFFLED, training did something.
#  If TRAINED, RANDOM and SHUFFLED are indistinguishable, the circuits
#  really are unstructured and the earlier reading stands.
#
#  Also tests whether the OV circuits are just tracking sign frequency:
#  several heads push toward sign 342, which is 10.2% of all tokens, and
#  that could be unigram bias rather than a learned ender circuit.
#
#  USAGE
#      using Pkg; Pkg.activate(".")
#      include("indus_data.jl")
#      inscriptions = load_inscriptions()
#      include("indus_model_v2.jl")
#      include("indus_vault.jl")
#      M = indus_get(inscriptions)
#      include("indus_circuits_control.jl")
#
#  Trains the shuffled-corpus model once (~4 min), then caches it in
#  indus_model_shuffled.jld2. Writes control_summary.csv.
# =====================================================================
using Flux, LinearAlgebra, Statistics, Random, Printf, DelimitedFiles

# ---------------------------------------------------------------------
# per-head weight slices (same packing as indus_model_v2.jl)
# ---------------------------------------------------------------------
ctl_qk(mha, h) = begin
    dh = mha.d_head; r = (h-1)*dh+1 : h*dh
    (Float64.(mha.Wq.weight[r, :]), Float64.(mha.Wk.weight[r, :]))
end
ctl_vo(mha, h) = begin
    dh = mha.d_head; r = (h-1)*dh+1 : h*dh
    (Float64.(mha.Wv.weight[r, :]), Float64.(mha.Wo.weight[:, r]))
end

function eff_rank(A)
    s = svdvals(A); s = s[s .> 1e-12]
    isempty(s) && return 0.0
    p = s ./ sum(s)
    exp(-sum(p .* log.(p)))
end
frob(A) = sqrt(sum(abs2, A))
compscore(Wl, We) = frob(Wl * We) / (frob(Wl) * frob(We) + 1e-12)

# ---------------------------------------------------------------------
# summarise one model
# ---------------------------------------------------------------------
function circuit_summary(model, keep_idx)
    nl = length(model.blocks); nh = model.blocks[1].attn.n_heads
    qkr = Float64[]; ovr = Float64[]
    for l in 1:nl, h in 1:nh
        Wq, Wk = ctl_qk(model.blocks[l].attn, h)
        Wv, Wo = ctl_vo(model.blocks[l].attn, h)
        push!(qkr, eff_rank(Wq' * Wk))
        push!(ovr, eff_rank(Wo * Wv))
    end
    comps = Float64[]
    for l1 in 1:nl-1, h1 in 1:nh
        Wv1, Wo1 = ctl_vo(model.blocks[l1].attn, h1)
        OV1 = Wo1 * Wv1
        for l2 in l1+1:nl, h2 in 1:nh
            Wq2, Wk2 = ctl_qk(model.blocks[l2].attn, h2)
            Wv2, _   = ctl_vo(model.blocks[l2].attn, h2)
            push!(comps, compscore(Wq2, OV1))
            push!(comps, compscore(Wk2, OV1))
            push!(comps, compscore(Wv2, OV1))
        end
    end
    E = Float64.(model.tok_emb.weight)[:, keep_idx]
    sE = svdvals(E)
    (qk_rank = mean(qkr), ov_rank = mean(ovr),
     comp_med = median(comps), comp_max = maximum(comps),
     emb_rank = eff_rank(E),
     emb_top8 = sum(sE[1:min(8,length(sE))]) / sum(sE),
     emb_decay = sE[1] / sE[min(10, length(sE))])
end

const KEEPC = [M.stoi[s] for s in M.signs if get(M.counts, s, 0) >= 20]
const DM    = size(M.model.tok_emb.weight, 1)

# ---------------------------------------------------------------------
# RANDOM: same architecture, untrained
# ---------------------------------------------------------------------
println("building untrained control ...")
Random.seed!(999)
rand_model = IndusGPT(vocab_size = M.nv, d_model = DM, n_heads = 4,
                      n_layers = 3, d_ff = 256, max_len = M.maxl,
                      pdrop = 0.1, window = M.window)
Flux.testmode!(rand_model)

# ---------------------------------------------------------------------
# SHUFFLED: trained on order-destroyed inscriptions
# ---------------------------------------------------------------------
println("preparing shuffled corpus ...")
Random.seed!(4321)
shuffled = [shuffle(copy(t)) for t in inscriptions]
# sanity: unigram counts must be unchanged, bigram structure must not be
c1 = Dict{String,Int}(); c2 = Dict{String,Int}()
for t in inscriptions, s in t; c1[s] = get(c1,s,0)+1; end
for t in shuffled,     s in t; c2[s] = get(c2,s,0)+1; end
@printf("  unigram counts identical: %s\n", c1 == c2 ? "yes" : "NO -- bug")

shuf_path = "indus_model_shuffled.jld2"
MS = indus_get(shuffled; path = shuf_path)
@printf("  shuffled-corpus model held-out ppl %.2f  (real corpus %.2f)\n",
        MS.ppl, M.ppl)
println("  a large gap here means the real corpus has sequential structure;")
println("  a small gap would mean the model was never using it.\n")

# ---------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------
S_tr = circuit_summary(M.model,      KEEPC)
S_rd = circuit_summary(rand_model,   KEEPC)
S_sh = circuit_summary(MS.model,     [MS.stoi[s] for s in MS.signs if get(MS.counts,s,0) >= 20])

println("="^76)
println("CIRCUIT STRUCTURE: TRAINED vs UNTRAINED vs TRAINED-ON-SHUFFLED")
println("="^76)
@printf("%-34s %11s %11s %11s\n", "", "TRAINED", "RANDOM", "SHUFFLED")
rows = [
 ("QK effective rank (max 16)",        S_tr.qk_rank,  S_rd.qk_rank,  S_sh.qk_rank),
 ("OV effective rank (max 16)",        S_tr.ov_rank,  S_rd.ov_rank,  S_sh.ov_rank),
 ("composition, median",               S_tr.comp_med, S_rd.comp_med, S_sh.comp_med),
 ("composition, max",                  S_tr.comp_max, S_rd.comp_max, S_sh.comp_max),
 ("embedding effective rank (max $DM)",S_tr.emb_rank, S_rd.emb_rank, S_sh.emb_rank),
 ("embedding top-8 spectral share",    S_tr.emb_top8, S_rd.emb_top8, S_sh.emb_top8),
 ("embedding sigma1/sigma10",          S_tr.emb_decay,S_rd.emb_decay,S_sh.emb_decay),
]
for (n,a,b,c) in rows
    @printf("%-34s %11.4f %11.4f %11.4f\n", n, a, b, c)
end
@printf("\nrandom-matrix reference for composition: 1/sqrt(d_model) = %.4f\n",
        1/sqrt(DM))

open("control_summary.csv","w") do io
    println(io, "quantity,trained,random,shuffled")
    for (n,a,b,c) in rows
        @printf(io, "%s,%.6f,%.6f,%.6f\n", replace(n, "," => ";"), a, b, c)
    end
end

println("""
Reading this. If TRAINED matches RANDOM on every row, training left the
circuit structure untouched and the model is doing its work somewhere
other than in structured attention circuits. If TRAINED separates from
RANDOM but matches SHUFFLED, the structure it acquired is whatever
training on ANY corpus with these unigram statistics produces, not
anything about Indus sequence. Only TRAINED separating from BOTH is
evidence of learned sequential structure.
""")

# ---------------------------------------------------------------------
# is the OV circuit just tracking sign frequency?
#
# Several heads push toward 342, which is 10.2% of all tokens. That may
# be a learned ender circuit or may be unigram bias.
# ---------------------------------------------------------------------
println("="^76)
println("OV OUTPUT MASS vs SIGN FREQUENCY")
println("="^76)
keepsigns = [s for s in M.signs if get(M.counts, s, 0) >= 20]
kidx = [M.stoi[s] for s in keepsigns]
freq = Float64[M.counts[s] for s in keepsigns]; freq ./= sum(freq)
WU = Float64.(M.model.head.weight)

function ov_out_mass(model, l, h, kidx)
    blk = model.blocks[l]
    Eln = Float64.(blk.ln1(model.tok_emb.weight))[:, kidx]
    Wv, Wo = ctl_vo(blk.attn, h)
    OV = WU * (Wo * (Wv * Eln))         # (vocab, nkeep)
    vec(mean(OV[kidx, :], dims = 2))    # mean logit pushed onto each output sign
end

cors = Float64[]
for l in 1:3, h in 1:4
    m = ov_out_mass(M.model, l, h, kidx)
    push!(cors, cor(m, freq))
end
@printf("correlation between a head's mean OV output logit and sign frequency:\n")
for (i,(l,h)) in enumerate(Iterators.product(1:3,1:4) |> collect |> vec)
    @printf("  L%dH%d  %+.3f\n", l, h, cors[i])
end
@printf("\n  mean |correlation| across heads: %.3f\n", mean(abs.(cors)))
println("""  High correlation means the OV circuits are reproducing unigram
  frequency rather than a learned ender rule, and the repeated
  '=> 342' entries in the circuit listing are frequency bias.""")
