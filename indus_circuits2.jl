# =====================================================================
#  indus_circuits2.jl
#
#  Follow-up to indus_circuits_control.jl, which returned a null on every
#  metric -- QK/OV effective rank, composition, embedding spectrum all
#  identical between TRAINED, RANDOM and SHUFFLED.
#
#  That null cannot be taken at face value. The same run showed held-out
#  perplexity 24.43 on the real corpus against 59.30 on the shuffled one:
#  the model learned a great deal of sequential structure. Metrics that
#  cannot separate trained from random are therefore insensitive, not
#  evidence of absence.
#
#  Two reasons they failed, and the fixes:
#
#  1. WRONG SPACE. Those metrics used weight-space objects (Wq'Wk, 64x64).
#     What the model applies to signs is the sign-space form
#     E' Wq' Wk E. At lr 2e-4 for ~50 epochs the weights move only a
#     small distance relative to their initialisation scale, so
#     weight-space spectra stay dominated by the init bulk while the
#     sign-space structure changes a lot. Everything below is computed in
#     sign space.
#
#  2. NO TARGET. Effective rank asks "is this structured?" in the
#     abstract. The sharper question is "structured HOW". Since the model
#     is behaviourally close to a bigram (window 2 already matches
#     unrestricted attention), the specific hypothesis is that the QK
#     circuits have absorbed bigram statistics. That is directly
#     testable: correlate W_QK[i,j] against the empirical
#     log P(sign j follows sign i).
#
#  ANALYSES
#   (a) sign-space QK and OV effective rank, trained vs random vs shuffled
#   (b) correlation of QK circuits with empirical bigram log-probabilities
#   (c) correlation of OV circuits with the same, and with unigram
#       frequency -- the repeated "=> 342" entries in the earlier listing
#       may just be frequency bias, since 342 is 10.2% of all tokens
#   (d) per-head Delta from the random model in sign space, to see which
#       heads training actually moved
#
#  Needs M (trained) and MS (shuffled) in scope, so run this AFTER
#  indus_circuits_control.jl in the same session.
#
#      include("indus_circuits2.jl")
#
#  Writes circuits2_summary.csv, qk_bigram_corr.csv
# =====================================================================
using Flux, LinearAlgebra, Statistics, Random, Printf, DelimitedFiles

# all names suffixed _C2 to avoid clashing with consts from earlier scripts
c2_qk(mha, h) = begin
    dh = mha.d_head; r = (h-1)*dh+1 : h*dh
    (Float64.(mha.Wq.weight[r, :]), Float64.(mha.Wk.weight[r, :]))
end
c2_vo(mha, h) = begin
    dh = mha.d_head; r = (h-1)*dh+1 : h*dh
    (Float64.(mha.Wv.weight[r, :]), Float64.(mha.Wo.weight[:, r]))
end
c2_effrank(A) = begin
    s = svdvals(A); s = s[s .> 1e-12]
    isempty(s) && return 0.0
    p = s ./ sum(s); exp(-sum(p .* log.(p)))
end

# common sign set: signs with count >= 20 in the REAL corpus
const SIGNS_C2 = [s for s in M.signs if get(M.counts, s, 0) >= 20]
const NC2 = length(SIGNS_C2)
@printf("sign-space analysis on %d signs (count >= 20)\n\n", NC2)

"Sign-space QK matrix for one head: entry (i,j) = query sign i attending to key sign j."
function qk_signspace(model, l, h, stoi)
    blk = model.blocks[l]
    idx = [stoi[s] for s in SIGNS_C2]
    Eln = Float64.(blk.ln1(model.tok_emb.weight))[:, idx]
    Wq, Wk = c2_qk(blk.attn, h)
    (Eln' * Wq') * (Wk * Eln) ./ sqrt(blk.attn.d_head)
end

"Sign-space OV matrix: entry (o,j) = logit pushed onto output sign o by attending to key sign j."
function ov_signspace(model, l, h, stoi)
    blk = model.blocks[l]
    idx = [stoi[s] for s in SIGNS_C2]
    Eln = Float64.(blk.ln1(model.tok_emb.weight))[:, idx]
    Wv, Wo = c2_vo(blk.attn, h)
    OV = Float64.(model.head.weight) * (Wo * (Wv * Eln))
    OV[idx, :]
end

# ---------------------------------------------------------------------
# a random model to compare against
# ---------------------------------------------------------------------
Random.seed!(999)
rnd_C2 = IndusGPT(vocab_size = M.nv, d_model = size(M.model.tok_emb.weight,1),
                  n_heads = 4, n_layers = 3, d_ff = 256, max_len = M.maxl,
                  pdrop = 0.1, window = M.window)
Flux.testmode!(rnd_C2)

# ---------------------------------------------------------------------
# (a) sign-space effective rank
# ---------------------------------------------------------------------
println("="^76)
println("(a) EFFECTIVE RANK IN SIGN SPACE  (max $NC2, not 16)")
println("="^76)
println("This is the object the model actually applies to signs, unlike the")
println("weight-space form used before.\n")

function meanranks(model, stoi)
    q = Float64[]; o = Float64[]
    for l in 1:3, h in 1:4
        push!(q, c2_effrank(qk_signspace(model, l, h, stoi)))
        push!(o, c2_effrank(ov_signspace(model, l, h, stoi)))
    end
    (mean(q), mean(o), q, o)
end
qt, ot, qtv, otv = meanranks(M.model,  M.stoi)
qr, orr, qrv, orv = meanranks(rnd_C2,  M.stoi)
qs, os, qsv, osv = meanranks(MS.model, MS.stoi)

@printf("%-28s %11s %11s %11s\n", "", "TRAINED", "RANDOM", "SHUFFLED")
@printf("%-28s %11.2f %11.2f %11.2f\n", "QK sign-space eff. rank", qt, qr, qs)
@printf("%-28s %11.2f %11.2f %11.2f\n", "OV sign-space eff. rank", ot, orr, os)
println()
@printf("per-head QK rank, trained:  %s\n", join([@sprintf("%.1f", v) for v in qtv], " "))
@printf("per-head QK rank, random:   %s\n", join([@sprintf("%.1f", v) for v in qrv], " "))

# ---------------------------------------------------------------------
# (b) do the QK circuits encode bigram statistics?
# ---------------------------------------------------------------------
println()
println("="^76)
println("(b) QK CIRCUITS vs EMPIRICAL BIGRAM STATISTICS")
println("="^76)
println("Correlation between W_QK[i,j] and log P(sign j follows sign i),")
println("computed over sign pairs that actually occur. If the model has")
println("absorbed bigram structure into its attention, this is where it")
println("would show up.\n")

# empirical bigram log-probabilities, add-one smoothed over the kept signs
bi_C2 = zeros(Float64, NC2, NC2)
ix_C2 = Dict(s => i for (i,s) in enumerate(SIGNS_C2))
for t in inscriptions, (a,b) in zip(t, t[2:end])
    (haskey(ix_C2,a) && haskey(ix_C2,b)) || continue
    bi_C2[ix_C2[a], ix_C2[b]] += 1
end
occ_C2 = bi_C2 .> 0                      # pairs actually attested
logp_C2 = similar(bi_C2)
for i in 1:NC2
    r = bi_C2[i,:] .+ 1.0
    logp_C2[i,:] = log.(r ./ sum(r))
end

function qk_bigram_corr(model, stoi)
    out = Float64[]
    for l in 1:3, h in 1:4
        Q = qk_signspace(model, l, h, stoi)
        push!(out, cor(vec(Q[occ_C2]), vec(logp_C2[occ_C2])))
    end
    out
end
ct = qk_bigram_corr(M.model,  M.stoi)
cr = qk_bigram_corr(rnd_C2,   M.stoi)
cs = qk_bigram_corr(MS.model, MS.stoi)

@printf("  %6s %11s %11s %11s\n", "head", "TRAINED", "RANDOM", "SHUFFLED")
i = 1
for l in 1:3, h in 1:4
    @printf("  L%dH%d   %+11.4f %+11.4f %+11.4f\n", l, h, ct[i], cr[i], cs[i])
    global i += 1
end
@printf("\n  mean |corr|:  trained %.4f   random %.4f   shuffled %.4f\n",
        mean(abs.(ct)), mean(abs.(cr)), mean(abs.(cs)))
println("  A clear trained-vs-random gap means the attention circuits encode")
println("  which sign follows which. No gap means the bigram behaviour is")
println("  implemented somewhere else -- most likely the feed-forward layers.")

# ---------------------------------------------------------------------
# (c) OV circuits: bigram structure, or just unigram frequency?
# ---------------------------------------------------------------------
println()
println("="^76)
println("(c) OV CIRCUITS: BIGRAM STRUCTURE vs UNIGRAM FREQUENCY")
println("="^76)
freq_C2 = Float64[M.counts[s] for s in SIGNS_C2]; freq_C2 ./= sum(freq_C2)
lfreq_C2 = log.(freq_C2)

@printf("  %6s %14s %16s\n", "head", "corr w/ bigram", "corr w/ unigram")
ovb = Float64[]; ovu = Float64[]
for l in 1:3, h in 1:4
    O = ov_signspace(M.model, l, h, M.stoi)      # (out, key)
    # bigram target: attending to key j should push output o if o follows j
    cb = cor(vec(O'[occ_C2]), vec(logp_C2[occ_C2]))
    # unigram: mean logit pushed onto each output sign, vs its frequency
    cu = cor(vec(mean(O, dims = 2)), lfreq_C2)
    push!(ovb, cb); push!(ovu, cu)
    @printf("  L%dH%d  %+14.4f %+16.4f\n", l, h, cb, cu)
end
@printf("\n  mean |corr| with bigram  %.4f\n", mean(abs.(ovb)))
@printf("  mean |corr| with unigram %.4f\n", mean(abs.(ovu)))
println("  If the unigram column dominates, the repeated '=> 342' entries in")
println("  the circuit listing are frequency bias, not a learned ender rule.")

# ---------------------------------------------------------------------
# (d) which heads did training actually move?
# ---------------------------------------------------------------------
println()
println("="^76)
println("(d) HOW FAR TRAINING MOVED EACH HEAD, IN SIGN SPACE")
println("="^76)
println("Relative Frobenius distance between the trained and random")
println("sign-space QK matrices. Larger = training reshaped that head more.\n")

mv = Float64[]
for l in 1:3, h in 1:4
    A = qk_signspace(M.model, l, h, M.stoi)
    B = qk_signspace(rnd_C2,  l, h, M.stoi)
    d = sqrt(sum(abs2, A .- B)) / sqrt(sum(abs2, B))
    push!(mv, d)
    @printf("  L%dH%d  %.4f\n", l, h, d)
end
@printf("\n  mean %.4f, range %.4f to %.4f\n", mean(mv), minimum(mv), maximum(mv))

open("circuits2_summary.csv","w") do io
    println(io, "head,qk_rank_trained,qk_rank_random,qk_bigram_corr_trained,qk_bigram_corr_random,ov_bigram_corr,ov_unigram_corr,move_from_random")
    i = 1
    for l in 1:3, h in 1:4
        @printf(io, "L%dH%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                l,h,qtv[i],qrv[i],ct[i],cr[i],ovb[i],ovu[i],mv[i])
        i += 1
    end
end
println("\nwrote circuits2_summary.csv")
