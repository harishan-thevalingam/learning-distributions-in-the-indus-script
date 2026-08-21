# =====================================================================
#  indus_attn_long.jl
#
#  Attention-head diagnostics restricted to LONGER inscriptions.
#
#  Motivation: in the full-corpus run every head came out at chance on
#  mean attended distance (chance 1.325, observed 0.96-1.33). That is
#  largely a dilution artefact. Inscriptions average 4.52 signs, so
#  3,096 of the 8,548 attention rows sit at position 1 or 2, where a
#  head has one or two places to look and cannot express a preference.
#  Those rows carry no information but half the weight.
#
#  TWO filters, because filtering on length alone is not enough: even a
#  14-sign inscription still has a position 1 and a position 2.
#
#      MIN_LEN  keep only inscriptions with at least this many signs
#      MIN_POS  keep only query positions at or beyond this index
#
#  The chance levels are RECOMPUTED on exactly the surviving rows. The
#  published 1.325 / 1.096 came from the full set and does not apply
#  here -- longer contexts mean a uniform head scores higher, so reusing
#  the old baseline would manufacture a false positive.
#
#  Loads the saved model. No training.
#
#  USAGE
#      using Pkg; Pkg.activate(".")
#      include("indus_data.jl")
#      inscriptions = load_inscriptions()
#      include("indus_model_v2.jl")
#      include("indus_vault.jl")
#      M = indus_get(inscriptions)
#      include("indus_attn_long.jl")
#
#  Writes head_stats_long.csv and prints a comparison table.
#  Runtime: seconds.
# =====================================================================
using Flux, Statistics, Printf, DelimitedFiles

# ---------------------------------------------------------------------
# recover per-head attention weights by replaying MHA's own arithmetic
# up to the softmax, using the model's learned Wq and Wk
# ---------------------------------------------------------------------
function attn_weights_layer(mha::MHA, x, mask)
    d_model, seq_len, batch = size(x)
    H, dh = mha.n_heads, mha.d_head
    q, k = mha.Wq(x), mha.Wk(x)
    toh(t) = reshape(permutedims(reshape(t, dh, H, seq_len, batch), (1,3,2,4)),
                     dh, seq_len, H*batch)
    q, k = toh(q), toh(k)
    scores = Flux.batched_mul(Flux.batched_transpose(q), k) .* Float32(1/sqrt(dh))
    scores = scores .+ mask
    Flux.softmax(scores; dims = 2)          # (seq, seq, H*batch)
end

function attn_maps(model, ids::Vector{Int})
    S = length(ids)
    x = reshape(ids, S, 1)
    h = model.tok_emb(x) .+ model.pe[:, 1:S]
    h = model.drop(h)
    mask = causal_mask(S, model.window)
    maps = Array{Float32,3}[]
    for blk in model.blocks
        xin = blk.ln1(h)
        push!(maps, attn_weights_layer(blk.attn, xin, mask))
        h = h .+ blk.drop(blk.attn(xin, mask))
        h = h .+ blk.drop(blk.ff(blk.ln2(h)))
    end
    maps
end

# ---------------------------------------------------------------------
# diagnostics + matching chance level over one filtered row set
# ---------------------------------------------------------------------
"""
    attn_diagnostics(M, inscriptions; min_len, min_pos)

Per-head mean entropy and mean attended distance, over query positions
`min_pos`..end of inscriptions with at least `min_len` signs. Returns the
two head matrices, the chance levels for the same rows, and the row count.
"""
function attn_diagnostics(M, inscriptions; min_len::Int = 1, min_pos::Int = 1)
    model = M.model
    Flux.testmode!(model)
    nl = length(model.blocks)
    nh = model.blocks[1].attn.n_heads

    ent  = zeros(Float64, nl, nh)
    dist = zeros(Float64, nl, nh)
    # chance: a head spreading weight uniformly over the i positions it can see
    ch_e = 0.0; ch_d = 0.0
    nrows = 0

    for ins in inscriptions
        length(ins) < min_len && continue
        ids = vcat(M.bos, M.encode(ins))
        S = length(ids)
        S > M.maxl && (ids = ids[1:M.maxl]; S = M.maxl)
        S < min_pos && continue
        maps = attn_maps(model, ids)
        for i in min_pos:S
            for (l, A) in enumerate(maps), hh in 1:nh
                e = 0.0; d = 0.0
                for j in 1:i
                    w = Float64(A[i, j, hh])
                    w > 1e-12 && (e -= w * log(w))
                    d += w * (i - j)
                end
                ent[l,hh]  += e
                dist[l,hh] += d
            end
            ch_e += log(i)          # entropy of uniform over i options
            ch_d += (i - 1) / 2     # mean |i-j| for uniform over 1..i
            nrows += 1
        end
    end

    if nrows == 0
        error("no rows survived the filter (min_len=$min_len, min_pos=$min_pos)")
    end
    (entropy = ent ./ nrows, distance = dist ./ nrows,
     chance_entropy = ch_e / nrows, chance_distance = ch_d / nrows,
     rows = nrows, nl = nl, nh = nh)
end

function report(tag, R)
    println()
    println("="^72)
    @printf("%s   (%d attention rows)\n", tag, R.rows)
    @printf("chance level:  entropy %.3f    distance %.3f\n",
            R.chance_entropy, R.chance_distance)
    println("="^72)
    @printf("%6s %5s %10s %10s %12s %10s\n",
            "layer", "head", "entropy", "vs chance", "distance", "vs chance")
    for l in 1:R.nl, hh in 1:R.nh
        e = R.entropy[l,hh]; d = R.distance[l,hh]
        @printf("%6d %5d %10.3f %10.3f %12.3f %10.3f\n",
                l, hh, e, e - R.chance_entropy, d, d - R.chance_distance)
    end
    # effective positions attended: exp(entropy) is the readable form
    @printf("\n  effective positions attended (exp of entropy): %.2f to %.2f, chance %.2f\n",
            exp(minimum(R.entropy)), exp(maximum(R.entropy)), exp(R.chance_entropy))
    # how far from chance is the most extreme head, in each measure
    de = extrema(R.entropy .- R.chance_entropy)
    dd = extrema(R.distance .- R.chance_distance)
    @printf("  entropy  deviation from chance: %+.3f to %+.3f\n", de[1], de[2])
    @printf("  distance deviation from chance: %+.3f to %+.3f\n", dd[1], dd[2])
end

# ---------------------------------------------------------------------
# run: full corpus, then progressively stricter filters
# ---------------------------------------------------------------------
lens = length.(inscriptions)
@printf("corpus: %d inscriptions, mean length %.2f\n", length(inscriptions), mean(lens))
for L in [6, 8, 10]
    @printf("  inscriptions with >= %2d signs: %4d (%.1f%%)\n",
            L, count(>=(L), lens), 100*count(>=(L), lens)/length(lens))
end

R_all  = attn_diagnostics(M, inscriptions)
R_len6 = attn_diagnostics(M, inscriptions; min_len = 6)
R_l6p4 = attn_diagnostics(M, inscriptions; min_len = 6, min_pos = 4)
R_l8p4 = attn_diagnostics(M, inscriptions; min_len = 8, min_pos = 4)

report("ALL INSCRIPTIONS, ALL POSITIONS  (the published run)", R_all)
report("INSCRIPTIONS >= 6 SIGNS, ALL POSITIONS", R_len6)
report("INSCRIPTIONS >= 6 SIGNS, POSITIONS >= 4", R_l6p4)
report("INSCRIPTIONS >= 8 SIGNS, POSITIONS >= 4", R_l8p4)

# ---------------------------------------------------------------------
# export the main filtered condition
# ---------------------------------------------------------------------
open("head_stats_long.csv", "w") do io
    println(io, "condition,layer,head,entropy,distance,chance_entropy,chance_distance,rows")
    for (tag, R) in [("all", R_all), ("len6", R_len6),
                     ("len6_pos4", R_l6p4), ("len8_pos4", R_l8p4)]
        for l in 1:R.nl, hh in 1:R.nh
            @printf(io, "%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%d\n",
                    tag, l, hh, R.entropy[l,hh], R.distance[l,hh],
                    R.chance_entropy, R.chance_distance, R.rows)
        end
    end
end
println("\nwrote head_stats_long.csv")

println("""
Reading this: the chance level RISES as the filter tightens, because
longer contexts give a uniform head more room. The question is whether
any head pulls further BELOW chance on entropy, or away from chance on
distance, once the uninformative short rows are removed. If the gaps in
the "vs chance" columns stay flat as the filter tightens, the heads have
genuinely learned nothing positional and the earlier result was not a
dilution artefact after all.
""")
