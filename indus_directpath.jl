# =====================================================================
#  indus_directpath.jl
#
#  WHERE IS THE SEQUENTIAL STRUCTURE?
#
#  Established so far:
#    - the model learns a great deal: held-out perplexity 24.43 on the
#      real corpus against 59.30 trained on shuffled inscriptions with
#      identical unigram statistics
#    - yet the attention circuits show nothing: sign-space QK effective
#      rank is identical to a random model (13.93 vs 14.06), and the
#      correlation between QK circuits and empirical bigram
#      log-probabilities is flat (trained 0.021, random 0.016, shuffled
#      0.023 -- the shuffled model actually scores higher)
#    - OV circuits track unigram frequency (0.134) far more than bigram
#      structure (0.044), so the repeated "=> 342" entries are frequency
#      bias rather than a learned ender rule
#
#  So the structure is being learned somewhere other than attention.
#
#  THE HYPOTHESIS. In a residual transformer the DIRECT PATH carries the
#  token embedding straight to the unembedding, bypassing every attention
#  head and MLP. Its contribution to the logits is W_U W_E, a vocab x
#  vocab matrix whose entry (o,i) is the logit for output o given current
#  token i. That is structurally a bigram. A transformer can therefore
#  implement bigram statistics entirely in the direct path, with
#  attention contributing nothing -- which would explain every null
#  collected so far in one stroke.
#
#  TWO TESTS
#
#  (1) CORRELATIONAL. Does W_U W_E correlate with the empirical
#      log P(next = o | current = i)? Compared against the random and
#      shuffled models.
#
#  (2) CAUSAL, and decisive. Ablate components and re-measure held-out
#      perplexity on the SAVED weights, without retraining:
#        - all attention zeroed        (direct path + MLPs only)
#        - all MLPs zeroed             (direct path + attention only)
#        - both zeroed                 (direct path alone)
#      If perplexity barely moves when attention is ablated, attention is
#      decorative and the bigram lives in the direct path and MLPs. The
#      reference points are 24.43 for the intact model and 59.30 for a
#      model trained on order-destroyed text.
#
#  Run after indus_circuits_control.jl in the same session (needs M, MS).
#      include("indus_directpath.jl")
#
#  Writes directpath_summary.csv
# =====================================================================
using Flux, LinearAlgebra, Statistics, Random, Printf, DelimitedFiles

const SG_DP = [s for s in M.signs if get(M.counts, s, 0) >= 20]
const ND_DP = length(SG_DP)
const IX_DP = Dict(s => i for (i,s) in enumerate(SG_DP))
@printf("direct-path analysis on %d signs (count >= 20)\n\n", ND_DP)

# empirical bigram log-probabilities over the kept signs
bi_DP = zeros(Float64, ND_DP, ND_DP)
for t in inscriptions, (a,b) in zip(t, t[2:end])
    (haskey(IX_DP,a) && haskey(IX_DP,b)) || continue
    bi_DP[IX_DP[a], IX_DP[b]] += 1
end
occ_DP = bi_DP .> 0
logp_DP = similar(bi_DP)
for i in 1:ND_DP
    r = bi_DP[i,:] .+ 1.0
    logp_DP[i,:] = log.(r ./ sum(r))
end
@printf("attested sign pairs among these signs: %d of %d cells\n\n",
        count(occ_DP), ND_DP^2)

# ---------------------------------------------------------------------
# (1) direct path W_U W_E
# ---------------------------------------------------------------------
println("="^76)
println("(1) DIRECT PATH  W_U W_E  vs EMPIRICAL BIGRAM")
println("="^76)
println("Entry (o,i) of W_U W_E is the logit pushed onto output sign o when")
println("the current sign is i, through the residual stream alone, with no")
println("attention head or MLP involved.\n")

"Direct-path logit matrix restricted to the kept signs: rows = output, cols = current."
function direct_path(model, stoi)
    idx = [stoi[s] for s in SG_DP]
    E   = Float64.(model.tok_emb.weight)[:, idx]      # (d_model, nsign)
    Eln = Float64.(model.ln_f(Float32.(E)))           # final LN before unembedding
    D   = Float64.(model.head.weight) * Eln           # (vocab, nsign)
    D[idx, :]                                          # (nsign, nsign)
end

Random.seed!(999)
rnd_DP = IndusGPT(vocab_size = M.nv, d_model = size(M.model.tok_emb.weight,1),
                  n_heads = 4, n_layers = 3, d_ff = 256, max_len = M.maxl,
                  pdrop = 0.1, window = M.window)
Flux.testmode!(rnd_DP)

for (tag, mdl, st) in [("TRAINED", M.model, M.stoi),
                       ("RANDOM",  rnd_DP,  M.stoi),
                       ("SHUFFLED", MS.model, MS.stoi)]
    D = direct_path(mdl, st)
    c = cor(vec(D'[occ_DP]), vec(logp_DP[occ_DP]))
    @printf("  %-9s correlation with log P(next | current):  %+.4f\n", tag, c)
end
println()
println("  Compare with the QK circuits, which gave 0.021 / 0.016 / 0.023.")
println("  A large trained value here locates the bigram in the direct path.")

# ---------------------------------------------------------------------
# (2) ablation -- the causal test
# ---------------------------------------------------------------------
println()
println("="^76)
println("(2) ABLATION: held-out perplexity with components switched off")
println("="^76)

# forward pass with optional attention / MLP ablation, mirroring
# IndusGPT's own forward in indus_model_v2.jl
function fwd_ablate(model, x::AbstractMatrix{Int}; use_attn = true, use_mlp = true)
    seq_len, batch = size(x)
    h = model.tok_emb(x) .+ model.pe[:, 1:seq_len]
    mask = causal_mask(seq_len, model.window)
    for blk in model.blocks
        if use_attn
            h = h .+ blk.attn(blk.ln1(h), mask)
        end
        if use_mlp
            h = h .+ blk.ff(blk.ln2(h))
        end
    end
    model.head(model.ln_f(h))
end

function ppl_ablate(Mx, texts; use_attn = true, use_mlp = true)
    Flux.testmode!(Mx.model)
    ex = map(texts) do t
        ids = vcat(Mx.bos, Mx.encode(t), Mx.eos)
        length(ids) < Mx.maxl + 1 ? vcat(ids, fill(Mx.pad, Mx.maxl + 1 - length(ids))) :
                                    ids[1:Mx.maxl+1]
    end
    X = hcat([e[1:end-1] for e in ex]...)
    Y = hcat([e[2:end]   for e in ex]...)
    logits = fwd_ablate(Mx.model, X; use_attn = use_attn, use_mlp = use_mlp)
    lp = reshape(Flux.logsoftmax(logits; dims = 1), Mx.nv, prod(size(X)))
    y2 = reshape(Y, prod(size(Y)))
    tot = 0.0; n = 0
    for t in 1:length(y2)
        yt = y2[t]; yt == Mx.pad && continue
        tot += -lp[yt, t]; n += 1
    end
    exp(tot / n)
end

p_full    = ppl_ablate(M, M.test)
p_no_attn = ppl_ablate(M, M.test; use_attn = false)
p_no_mlp  = ppl_ablate(M, M.test; use_mlp  = false)
p_direct  = ppl_ablate(M, M.test; use_attn = false, use_mlp = false)

@printf("  %-32s %10s %12s\n", "configuration", "perplexity", "vs intact")
@printf("  %-32s %10.2f %12s\n", "intact model",                p_full, "-")
@printf("  %-32s %10.2f %+11.2f\n", "attention ablated",        p_no_attn, p_no_attn - p_full)
@printf("  %-32s %10.2f %+11.2f\n", "MLPs ablated",             p_no_mlp,  p_no_mlp  - p_full)
@printf("  %-32s %10.2f %+11.2f\n", "both ablated (direct path)", p_direct, p_direct - p_full)
println()
@printf("  reference: trained on shuffled inscriptions  %10.2f\n", MS.ppl)
@printf("  reference: uniform over %d signs             %10.2f\n", length(M.signs), Float64(length(M.signs)))
println("""
  If attention ablation costs little, the attention heads are decorative
  and the model's sequential competence sits in the direct path and MLPs.
  If the direct path alone already beats the shuffled-corpus model, the
  bigram is in W_U W_E, which is the structural answer to where the model
  keeps what it learned about the Indus script.""")

# ---------------------------------------------------------------------
# per-head ablation, for completeness
# ---------------------------------------------------------------------
println()
println("="^76)
println("PER-HEAD ABLATION")
println("="^76)
println("Zeroing one head's contribution at a time, by masking its slice of")
println("the Wo columns. Cost in held-out perplexity.\n")

rows_DP = Vector{Any}()
for l in 1:3, h in 1:4
    blk = M.model.blocks[l]
    dh = blk.attn.d_head; r = (h-1)*dh+1 : h*dh
    saved = copy(blk.attn.Wo.weight[:, r])
    blk.attn.Wo.weight[:, r] .= 0
    p = ppl_ablate(M, M.test)
    blk.attn.Wo.weight[:, r] .= saved
    @printf("  L%dH%d  %8.2f   %+6.2f\n", l, h, p, p - p_full)
    push!(rows_DP, (l, h, p, p - p_full))
end
@printf("\n  intact %.2f;  largest single-head cost %+.2f\n",
        p_full, maximum(r[4] for r in rows_DP))

open("directpath_summary.csv","w") do io
    println(io, "config,perplexity")
    @printf(io, "intact,%.4f\n", p_full)
    @printf(io, "no_attention,%.4f\n", p_no_attn)
    @printf(io, "no_mlp,%.4f\n", p_no_mlp)
    @printf(io, "direct_path_only,%.4f\n", p_direct)
    @printf(io, "shuffled_corpus_model,%.4f\n", MS.ppl)
    for (l,h,p,d) in rows_DP
        @printf(io, "ablate_L%dH%d,%.4f\n", l, h, p)
    end
end
println("\nwrote directpath_summary.csv")
