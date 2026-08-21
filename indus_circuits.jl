# =====================================================================
#  indus_circuits.jl
#
#  Structural analysis of the trained transformer: what the WEIGHTS say,
#  rather than what the attention maps do.
#
#  Everything so far has been descriptive -- entropy, attended distance,
#  neighbour tests. Those describe behaviour on particular inscriptions.
#  This reads the learned matrices directly, following the framework in
#  Elhage et al., "A Mathematical Framework for Transformer Circuits".
#
#  FOUR ANALYSES
#
#  (1) QK CIRCUITS. Attention score between two positions is
#          x_i' Wq' Wk x_j / sqrt(d_head)
#      so folding in the embeddings gives, per head, a sign-by-sign
#      matrix W_QK = E' Wq' Wk E / sqrt(d_head), 377 x 377. Entry (i,j)
#      is how much sign i, as a query, wants to attend to sign j, as a
#      key -- independent of any particular inscription. This is the
#      head's preference, read off the weights.
#
#  (2) OV CIRCUITS. Having attended to sign j, what does the head make
#      the model predict? W_OV = W_U Wo Wv E, again 377 x 377. Entry
#      (o,j) is the logit pushed onto output sign o when the head
#      attends to sign j.
#
#      Both are basis-free: they are invariant to the internal
#      coordinate choices that made cosine on raw embeddings
#      uninterpretable.
#
#  (3) RANK. d_head = 16, so each head's QK form has rank at most 16 on
#      a 377-dimensional sign space -- an architectural ceiling on the
#      structure a head can express. Singular value spectra say how much
#      of that budget is actually used, and the same for the embedding
#      matrix.
#
#  (4) COMPOSITION. Do layer-2 and layer-3 heads read what earlier heads
#      wrote? Q-, K- and V-composition scores per head pair. If these are
#      near zero the model is effectively three parallel one-layer models
#      rather than a depth-3 computation, which would explain the flat
#      window sweep from the model side.
#
#  CAVEATS, stated up front:
#   - Positional encodings are excluded. These are the content-based
#     (token-to-token) circuits; a head that is purely positional will
#     look empty here, which is itself informative but must not be read
#     as "this head does nothing".
#   - LayerNorm is applied numerically to the embeddings before folding,
#     which captures the gain but not the input-dependent scaling.
#   - Softmax is monotonic, so relative ordering within a row of W_QK is
#     exact even though the absolute values are pre-softmax logits.
#
#  USAGE (model comes from the vault, no training)
#      using Pkg; Pkg.activate(".")
#      include("indus_data.jl")
#      inscriptions = load_inscriptions()
#      include("indus_model_v2.jl")
#      include("indus_vault.jl")
#      M = indus_get(inscriptions)
#      include("indus_circuits.jl")
#
#  Writes qk_top.csv, ov_top.csv, rank_spectra.csv, composition.csv
# =====================================================================
using Flux, LinearAlgebra, Statistics, Printf, DelimitedFiles

# ---------------------------------------------------------------------
# pull per-head weight slices out of the packed Dense layers
#
# MHA packs heads along the feature dimension: the reshape in
# indus_model_v2.jl is reshape(t, dh, H, seq, batch), so head h occupies
# rows (h-1)*dh+1 : h*dh of Wq/Wk/Wv, and the matching COLUMNS of Wo.
# ---------------------------------------------------------------------
qk_slice(mha, h) = begin
    dh = mha.d_head; r = (h-1)*dh+1 : h*dh
    (mha.Wq.weight[r, :], mha.Wk.weight[r, :])
end
v_slice(mha, h) = begin
    dh = mha.d_head; r = (h-1)*dh+1 : h*dh
    (mha.Wv.weight[r, :], mha.Wo.weight[:, r])
end

const MODEL   = M.model
const NL      = length(MODEL.blocks)
const NH      = MODEL.blocks[1].attn.n_heads
const DH      = MODEL.blocks[1].attn.d_head
const WE      = MODEL.tok_emb.weight            # (d_model, vocab)
const WU      = MODEL.head.weight               # (vocab, d_model)
const DMODEL  = size(WE, 1)

# analysis restricted to signs the model actually saw enough of
const KEEP    = [s for s in M.signs if get(M.counts, s, 0) >= 20]
const KIDX    = [M.stoi[s] for s in KEEP]
const NK      = length(KEEP)

@printf("model: %d layers x %d heads, d_head %d, d_model %d, vocab %d\n",
        NL, NH, DH, DMODEL, M.nv)
@printf("circuit analysis on %d signs with count >= 20\n\n", NK)

# ---------------------------------------------------------------------
# (1) + (2)  QK and OV circuits
# ---------------------------------------------------------------------
println("="^72)
println("(1) QK CIRCUITS -- which sign attends to which, from the weights")
println("="^72)
println("Top pairs per head. 'query -> key' means: when the model is at the")
println("query sign, this head is pulled toward the key sign. Scores are")
println("pre-softmax logits; only their ordering within a head is meaningful.\n")

qk_rows = Vector{Any}()
ov_rows = Vector{Any}()
qk_mats = Dict{Tuple{Int,Int},Matrix{Float64}}()

for l in 1:NL
    blk = MODEL.blocks[l]
    # LayerNorm the embeddings the way the block would see them
    Eln = Float64.(blk.ln1(WE))[:, KIDX]        # (d_model, NK)
    for h in 1:NH
        Wq_h, Wk_h = qk_slice(blk.attn, h)
        QK = (Eln' * Float64.(Wq_h)') * (Float64.(Wk_h) * Eln) ./ sqrt(DH)
        qk_mats[(l,h)] = QK

        # strongest query->key preferences, excluding the diagonal
        off = copy(QK); for i in 1:NK; off[i,i] = -Inf; end
        ord = sortperm(vec(off); rev = true)[1:6]
        pairs = [(KEEP[((o-1) % NK) + 1], KEEP[div(o-1, NK) + 1], off[o]) for o in ord]
        # diagonal strength: does this head like attending to the SAME sign?
        # relevant here because sign 245 repeats adjacently in all 33
        # inscriptions where it repeats at all
        dg = mean(diag(QK)); offm = mean(off[isfinite.(off)])
        @printf("  L%dH%d  same-sign bias %+.2f  |  ", l, h, dg - offm)
        println(join([@sprintf("%s->%s(%.1f)", a, b, v) for (a,b,v) in pairs[1:4]], "  "))
        for (a,b,v) in pairs
            push!(qk_rows, (l, h, a, b, v))
        end
    end
end

println()
println("="^72)
println("(2) OV CIRCUITS -- what attending to a sign makes the model predict")
println("="^72)
println("'key => output' means: attending to the key sign pushes the model")
println("toward predicting the output sign.\n")

for l in 1:NL
    blk = MODEL.blocks[l]
    Eln = Float64.(blk.ln1(WE))[:, KIDX]
    for h in 1:NH
        Wv_h, Wo_h = v_slice(blk.attn, h)
        OV = Float64.(WU) * (Float64.(Wo_h) * (Float64.(Wv_h) * Eln))   # (vocab, NK)
        OVk = OV[KIDX, :]                                                # (NK, NK)
        ord = sortperm(vec(OVk); rev = true)[1:6]
        pairs = [(KEEP[div(o-1, NK) + 1], KEEP[((o-1) % NK) + 1], OVk[o]) for o in ord]
        @printf("  L%dH%d  ", l, h)
        println(join([@sprintf("%s=>%s(%.1f)", k, o, v) for (k,o,v) in pairs[1:4]], "  "))
        for (k,o,v) in pairs
            push!(ov_rows, (l, h, k, o, v))
        end
    end
end

open("qk_top.csv","w") do io
    println(io, "layer,head,query_sign,key_sign,score")
    for (l,h,a,b,v) in qk_rows; @printf(io, "%d,%d,%s,%s,%.6f\n", l,h,a,b,v); end
end
open("ov_top.csv","w") do io
    println(io, "layer,head,key_sign,output_sign,logit")
    for (l,h,k,o,v) in ov_rows; @printf(io, "%d,%d,%s,%s,%.6f\n", l,h,k,o,v); end
end

# ---------------------------------------------------------------------
# (3) RANK -- how much of the architectural budget is used
# ---------------------------------------------------------------------
println()
println("="^72)
println("(3) RANK -- capacity used against capacity available")
println("="^72)
println("Effective rank is exp(entropy of the normalised singular value")
println("spectrum): the number of directions genuinely in use. A head's QK")
println("form can have rank at most d_head = $DH; the embedding matrix at")
println("most d_model = $DMODEL.\n")

function eff_rank(A)
    s = svdvals(A)
    s = s[s .> 1e-12]
    isempty(s) && return 0.0
    p = s ./ sum(s)
    exp(-sum(p .* log.(p)))
end

# embedding matrix
sE = svdvals(Float64.(WE)[:, KIDX])
@printf("  embedding matrix (%d x %d): effective rank %.1f of %d possible\n",
        DMODEL, NK, eff_rank(Float64.(WE)[:, KIDX]), min(DMODEL, NK))
@printf("    top 10 singular values: %s\n",
        join([@sprintf("%.2f", v) for v in sE[1:min(10,length(sE))]], " "))
@printf("    fraction of spectral mass in top 8: %.1f%%\n\n",
        100*sum(sE[1:min(8,length(sE))])/sum(sE))

rank_rows = Vector{Any}()
@printf("  %6s %10s %14s %14s\n", "head", "QK e-rank", "QK top-1 share", "OV e-rank")
for l in 1:NL
    blk = MODEL.blocks[l]
    for h in 1:NH
        Wq_h, Wk_h = qk_slice(blk.attn, h)
        Wv_h, Wo_h = v_slice(blk.attn, h)
        QKw = Float64.(Wq_h)' * Float64.(Wk_h)      # (d_model, d_model), rank <= DH
        OVw = Float64.(Wo_h) * Float64.(Wv_h)
        sq = svdvals(QKw); so = svdvals(OVw)
        er_q = eff_rank(QKw); er_o = eff_rank(OVw)
        top1 = sq[1] / sum(sq)
        @printf("  L%dH%d  %10.2f %13.1f%% %14.2f\n", l, h, er_q, 100*top1, er_o)
        push!(rank_rows, (l, h, er_q, er_o, top1))
    end
end

open("rank_spectra.csv","w") do io
    println(io, "layer,head,qk_effective_rank,ov_effective_rank,qk_top1_share")
    for (l,h,a,b,c) in rank_rows; @printf(io, "%d,%d,%.6f,%.6f,%.6f\n", l,h,a,b,c); end
end

# ---------------------------------------------------------------------
# (4) COMPOSITION -- does depth do anything?
# ---------------------------------------------------------------------
println()
println("="^72)
println("(4) COMPOSITION -- do later heads read what earlier heads wrote?")
println("="^72)
println("Score is ||W_later * W_earlier||_F / (||W_later||_F ||W_earlier||_F),")
println("following Elhage et al. Q-composition: the later head's QUERY reads")
println("the earlier head's output. K-composition: its KEY does. V-composition:")
println("its VALUE does. Near-zero across the board means the layers are not")
println("cooperating and the model is effectively three parallel 1-layer models.\n")

frob(A) = sqrt(sum(abs2, A))
comp(Wlater, Wearlier) = frob(Wlater * Wearlier) / (frob(Wlater) * frob(Wearlier) + 1e-12)

comp_rows = Vector{Any}()
@printf("  %-14s %8s %8s %8s\n", "earlier -> later", "Q-comp", "K-comp", "V-comp")
for l1 in 1:NL-1, h1 in 1:NH
    Wv1, Wo1 = v_slice(MODEL.blocks[l1].attn, h1)
    OV1 = Float64.(Wo1) * Float64.(Wv1)            # what this head writes
    for l2 in l1+1:NL, h2 in 1:NH
        Wq2, Wk2 = qk_slice(MODEL.blocks[l2].attn, h2)
        Wv2, _   = v_slice(MODEL.blocks[l2].attn, h2)
        q = comp(Float64.(Wq2), OV1)
        k = comp(Float64.(Wk2), OV1)
        v = comp(Float64.(Wv2), OV1)
        push!(comp_rows, (l1,h1,l2,h2,q,k,v))
    end
end

# report the strongest, plus the overall distribution
sort!(comp_rows; by = r -> -max(r[5], r[6], r[7]))
for r in comp_rows[1:min(10, length(comp_rows))]
    @printf("  L%dH%d -> L%dH%d  %8.4f %8.4f %8.4f\n", r[1],r[2],r[3],r[4],r[5],r[6],r[7])
end
qs = [r[5] for r in comp_rows]; ks = [r[6] for r in comp_rows]; vs = [r[7] for r in comp_rows]
@printf("\n  across all %d head pairs:\n", length(comp_rows))
@printf("    Q-composition  median %.4f  max %.4f\n", median(qs), maximum(qs))
@printf("    K-composition  median %.4f  max %.4f\n", median(ks), maximum(ks))
@printf("    V-composition  median %.4f  max %.4f\n", median(vs), maximum(vs))
println("""
  Reference: for random matrices of this shape the score sits near
  1/sqrt(d_model) = $(round(1/sqrt(DMODEL), digits=3)). Scores at or below that
  indicate no genuine composition -- the later head is not reading the
  earlier one's output in any structured way.""")

open("composition.csv","w") do io
    println(io, "l1,h1,l2,h2,q_comp,k_comp,v_comp")
    for r in comp_rows
        @printf(io, "%d,%d,%d,%d,%.6f,%.6f,%.6f\n", r[1],r[2],r[3],r[4],r[5],r[6],r[7])
    end
end

println("\nwrote qk_top.csv, ov_top.csv, rank_spectra.csv, composition.csv")
