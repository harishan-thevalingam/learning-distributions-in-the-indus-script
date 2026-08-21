# =====================================================================
#  indus_mlp.jl
#
#  WHAT ARE THE MLPs DOING?
#
#  By elimination this is where the model's remaining knowledge sits.
#  Attention returns the initialisation value on every probe tried (rank,
#  bigram correlation in both directions, composition, additive
#  decomposition, content-vs-position against a random control), and an
#  attention-free architecture trained from scratch reaches 23.96 against
#  the full model's 24.43. The direct path W_U W_E carries part of the
#  bigram at +0.219 against +0.019 random. Everything else must be in the
#  feed-forward layers, which nothing has examined.
#
#  WHY NEURONS ARE READABLE. Each MLP maps 64 -> 256 -> 64. Each of the
#  256 hidden units, 768 across three layers, can be read from both ends:
#      input side  W_in[i,:]  which signs make neuron i fire
#      output side W_out[:,i] which signs neuron i then promotes,
#                             once pushed through the unembedding
#  That pair IS a transition rule. If the model implements its bigram in
#  the MLPs, neurons should be legible as individual rules of the form
#  "fires on sign A, promotes the signs that follow A".
#
#  FOUR ANALYSES
#   (1) activation profiles measured on real data, so LayerNorm and the
#       nonlinearity are handled empirically rather than approximated
#   (2) per-neuron bigram alignment: does the outer product of a neuron's
#       firing profile and its promotion profile match real transitions?
#   (3) the MLPs' aggregate effective transition matrix, against the
#       empirical bigram and against the direct path's +0.219
#   (4) neuron ablation, to see whether a few neurons carry the model or
#       the work is spread thin
#
#  Run after indus_vault.jl in the same session (needs M).
#      include("indus_mlp.jl")
#
#  Writes mlp_neurons.csv, mlp_summary.csv
# =====================================================================
using Flux, LinearAlgebra, Statistics, Printf, DelimitedFiles

const SG_MLP = [s for s in M.signs if get(M.counts, s, 0) >= 20]
const NS_MLP = length(SG_MLP)
const IXS_MLP = Dict(s => i for (i,s) in enumerate(SG_MLP))
const VIDX_MLP = [M.stoi[s] for s in SG_MLP]
const NLAY = length(M.model.blocks)
const DFF_MLP = size(M.model.blocks[1].ff[1].weight, 1)

@printf("%d layers x %d neurons = %d neurons total\n", NLAY, DFF_MLP, NLAY*DFF_MLP)
@printf("analysis on %d signs with count >= 20\n\n", NS_MLP)

# empirical bigram over the kept signs
bi_MLP = zeros(Float64, NS_MLP, NS_MLP)
for t in inscriptions, (a,b) in zip(t, t[2:end])
    (haskey(IXS_MLP,a) && haskey(IXS_MLP,b)) || continue
    bi_MLP[IXS_MLP[a], IXS_MLP[b]] += 1
end
occ_MLP = bi_MLP .> 0
logp_MLP = similar(bi_MLP)
for i in 1:NS_MLP
    r = bi_MLP[i,:] .+ 1.0
    logp_MLP[i,:] = log.(r ./ sum(r))
end

# ---------------------------------------------------------------------
# (1) activation profiles, measured on real data
# ---------------------------------------------------------------------
println("="^76)
println("(1) NEURON ACTIVATION PROFILES")
println("="^76)
println("Mean activation of each neuron at positions holding each sign,")
println("recorded from real forward passes so LayerNorm and the")
println("nonlinearity are handled exactly rather than approximated.\n")

Flux.testmode!(M.model)

# act[l][neuron, sign] = mean activation of that neuron when the current sign is that one
act_MLP = [zeros(Float64, DFF_MLP, NS_MLP) for _ in 1:NLAY]
cnt_MLP = zeros(Int, NS_MLP)

for ins in inscriptions
    ids = vcat(M.bos, M.encode(ins))
    S = length(ids); S > M.maxl && (ids = ids[1:M.maxl]; S = M.maxl)
    S < 2 && continue
    x = reshape(ids, S, 1)
    h = M.model.tok_emb(x) .+ M.model.pe[:, 1:S]
    mask = causal_mask(S, M.model.window)
    for (l, blk) in enumerate(M.model.blocks)
        h = h .+ blk.attn(blk.ln1(h), mask)
        pre = blk.ln2(h)
        a = blk.ff[1](pre)                      # (d_ff, S, 1), post-gelu
        for p in 1:S
            tok = ids[p]
            tok in (M.bos, M.eos, M.pad) && continue
            s = M.itos[tok]
            haskey(IXS_MLP, s) || continue
            j = IXS_MLP[s]
            @views act_MLP[l][:, j] .+= Float64.(a[:, p, 1])
            l == 1 && (cnt_MLP[j] += 1)
        end
        h = h .+ blk.ff(blk.ln2(h))
    end
end
for l in 1:NLAY, j in 1:NS_MLP
    cnt_MLP[j] > 0 && (@views act_MLP[l][:, j] ./= cnt_MLP[j])
end

for l in 1:NLAY
    sel = [std(act_MLP[l][i, :]) for i in 1:DFF_MLP]
    @printf("  layer %d: neuron selectivity across signs, mean sd %.4f, max %.4f\n",
            l, mean(sel), maximum(sel))
end
println("  (a neuron that fires equally for every sign has sd 0 and is")
println("   carrying no sign-specific information)")

# ---------------------------------------------------------------------
# (2) per-neuron promotion, and bigram alignment
# ---------------------------------------------------------------------
println()
println("="^76)
println("(2) WHICH SIGNS EACH NEURON PROMOTES, AND WHETHER THAT MATCHES REAL TRANSITIONS")
println("="^76)

WU_MLP = Float64.(M.model.head.weight)          # (vocab, d_model)

# promote[l][neuron, sign] = logit this neuron pushes onto that output sign
promote_MLP = Vector{Matrix{Float64}}(undef, NLAY)
for l in 1:NLAY
    Wout = Float64.(M.model.blocks[l].ff[2].weight)      # (d_model, d_ff)
    promote_MLP[l] = (WU_MLP * Wout)[VIDX_MLP, :]'       # (d_ff, nsign)
end

# a neuron's implied rule: outer product of firing profile and promotion
# profile. Correlate that against the empirical transition log-probabilities.
align_MLP = [zeros(Float64, DFF_MLP) for _ in 1:NLAY]
for l in 1:NLAY, i in 1:DFF_MLP
    a = act_MLP[l][i, :]; p = promote_MLP[l][i, :]
    (std(a) < 1e-9 || std(p) < 1e-9) && continue
    R = a * p'                                            # (fires on, promotes)
    align_MLP[l][i] = cor(vec(R[occ_MLP]), vec(logp_MLP[occ_MLP]))
end

for l in 1:NLAY
    v = align_MLP[l]
    @printf("  layer %d: bigram alignment  mean %+.4f  max %+.4f  min %+.4f\n",
            l, mean(v), maximum(v), minimum(v))
end

println("\n  Most bigram-aligned neurons, with the signs they fire on and promote:\n")
for l in 1:NLAY
    ord = sortperm(align_MLP[l]; rev = true)[1:3]
    for i in ord
        a = act_MLP[l][i, :]; p = promote_MLP[l][i, :]
        fires = SG_MLP[sortperm(a; rev = true)[1:4]]
        proms = SG_MLP[sortperm(p; rev = true)[1:4]]
        @printf("  L%d N%-3d  align %+.3f | fires on %-22s | promotes %s\n",
                l, i, align_MLP[l][i], join(fires, ","), join(proms, ","))
        # does it promote what actually follows the signs it fires on?
        top_fire = SG_MLP[argmax(a)]
        actual = SG_MLP[sortperm(bi_MLP[IXS_MLP[top_fire], :]; rev = true)[1:3]]
        @printf("           signs that actually follow %s: %s\n",
                top_fire, join(actual, ","))
    end
end

# ---------------------------------------------------------------------
# (3) aggregate: the MLPs' effective transition matrix
# ---------------------------------------------------------------------
println()
println("="^76)
println("(3) AGGREGATE MLP TRANSITION MATRIX vs EMPIRICAL BIGRAM")
println("="^76)
println("Summing every neuron's implied rule gives the MLPs' overall")
println("contribution to predicting the next sign.\n")

for l in 1:NLAY
    T = act_MLP[l]' * promote_MLP[l]          # (fires-on sign, promoted sign)
    c = cor(vec(T[occ_MLP]), vec(logp_MLP[occ_MLP]))
    @printf("  layer %d MLP:  correlation with log P(next | current)  %+.4f\n", l, c)
end
Tall = sum(act_MLP[l]' * promote_MLP[l] for l in 1:NLAY)
c_all = cor(vec(Tall[occ_MLP]), vec(logp_MLP[occ_MLP]))
@printf("  all MLPs:     correlation with log P(next | current)  %+.4f\n", c_all)
println()
println("  for comparison, from earlier runs:")
println("    direct path W_U W_E                          +0.219")
println("    QK attention circuits                        +0.021 (random +0.016)")

# ---------------------------------------------------------------------
# (4) neuron ablation
# ---------------------------------------------------------------------
println()
println("="^76)
println("(4) NEURON ABLATION")
println("="^76)
println("Zeroing one neuron's output at a time. Ablating the 20 with the")
println("largest promotion norm, and 20 chosen at random for comparison.\n")

function ppl_MLP(model)
    Flux.testmode!(model)
    ex = map(M.test) do t
        ids = vcat(M.bos, M.encode(t), M.eos)
        length(ids) < M.maxl+1 ? vcat(ids, fill(M.pad, M.maxl+1-length(ids))) : ids[1:M.maxl+1]
    end
    X = hcat([e[1:end-1] for e in ex]...); Y = hcat([e[2:end] for e in ex]...)
    lp = reshape(Flux.logsoftmax(model(X); dims=1), M.nv, prod(size(X)))
    y2 = reshape(Y, prod(size(Y)))
    tot = 0.0; n = 0
    for t in 1:length(y2)
        yt = y2[t]; yt == M.pad && continue
        tot += -lp[yt,t]; n += 1
    end
    exp(tot/n)
end

base_MLP = ppl_MLP(M.model)
@printf("  intact: %.2f\n\n", base_MLP)

norms_MLP = [(l, i, norm(promote_MLP[l][i, :])) for l in 1:NLAY for i in 1:DFF_MLP]
sort!(norms_MLP; by = x -> -x[3])

costs_top = Float64[]
for (l, i, _) in norms_MLP[1:20]
    W = M.model.blocks[l].ff[2].weight
    saved = copy(W[:, i]); W[:, i] .= 0
    push!(costs_top, ppl_MLP(M.model) - base_MLP)
    W[:, i] .= saved
end
using Random
Random.seed!(11)
rand_sel = norms_MLP[randperm(length(norms_MLP))[1:20]]
costs_rand = Float64[]
for (l, i, _) in rand_sel
    W = M.model.blocks[l].ff[2].weight
    saved = copy(W[:, i]); W[:, i] .= 0
    push!(costs_rand, ppl_MLP(M.model) - base_MLP)
    W[:, i] .= saved
end

@printf("  20 highest-norm neurons: mean cost %+.3f, max %+.3f\n",
        mean(costs_top), maximum(costs_top))
@printf("  20 random neurons:       mean cost %+.3f, max %+.3f\n",
        mean(costs_rand), maximum(costs_rand))
println("\n  Largest individual costs among the high-norm set:")
ordc = sortperm(costs_top; rev = true)[1:5]
for k in ordc
    (l, i, nn) = norms_MLP[k]
    @printf("    L%d N%-3d  %+.3f\n", l, i, costs_top[k])
end
println("""
  For reference, ablating ALL 12 attention heads cost +6.78. If no single
  neuron approaches that, the MLP computation is distributed rather than
  carried by a few interpretable units -- which would mean there are no
  clean 'bigram rule' neurons to read off, even though the MLPs
  collectively hold the structure.""")

open("mlp_neurons.csv","w") do io
    println(io, "layer,neuron,selectivity,promote_norm,bigram_align")
    for l in 1:NLAY, i in 1:DFF_MLP
        @printf(io, "%d,%d,%.6f,%.6f,%.6f\n", l, i,
                std(act_MLP[l][i,:]), norm(promote_MLP[l][i,:]), align_MLP[l][i])
    end
end
open("mlp_summary.csv","w") do io
    println(io, "quantity,value")
    for l in 1:NLAY
        T = act_MLP[l]' * promote_MLP[l]
        @printf(io, "layer%d_bigram_corr,%.6f\n", l, cor(vec(T[occ_MLP]), vec(logp_MLP[occ_MLP])))
    end
    @printf(io, "all_mlp_bigram_corr,%.6f\n", c_all)
    @printf(io, "intact_ppl,%.6f\n", base_MLP)
    @printf(io, "mean_cost_top20,%.6f\n", mean(costs_top))
    @printf(io, "mean_cost_random20,%.6f\n", mean(costs_rand))
end
println("\nwrote mlp_neurons.csv, mlp_summary.csv")
