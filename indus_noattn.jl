# =====================================================================
#  indus_noattn.jl
#
#  IS ATTENTION NECESSARY AT ALL?
#
#  Established: ablating all 12 heads from the trained model costs only
#  6.78 perplexity (24.43 -> 31.21), about 15% of everything gained over
#  the ~69 unigram baseline, and the bigram structure sits in the direct
#  path (W_U W_E correlates with log P(next|current) at +0.219 against
#  +0.019 random). But post-hoc ablation runs the network outside its
#  training regime, so some of that 6.78 is distribution shift rather
#  than lost function, and the objection cannot be answered by ablation.
#
#  This trains attention-free architectures FROM SCRATCH. No ablation, no
#  distribution shift, no objection available.
#
#  WHAT AN ATTENTION-FREE MODEL IS HERE. With attention removed no
#  position can see any other, so at position i the model sees only
#  emb(x_i) + pe(i) and must predict x_{i+1}. That is structurally a
#  bigram conditioned on position. If it matches the full transformer,
#  the transformer was never doing more than that.
#
#  FOUR CONDITIONS, same split, same vocabulary, same training protocol:
#    1. full transformer            (the saved model, for reference)
#    2. no attention, with PE       bigram + position
#    3. no attention, no PE         pure bigram, position-blind
#    4. no attention, wide MLP      parameter-matched to the full model,
#                                   so any gap is not just capacity
#
#  Condition 3 is the cleanest theoretical object: it CANNOT use position
#  or context, only the current sign. Its perplexity is an empirical
#  ceiling on what a bigram can achieve on this corpus with a neural
#  parameterisation, directly comparable to the Witten-Bell bigram's
#  25.50 from the replication run.
#
#  Run after indus_vault.jl in the same session (needs M for the split
#  and vocabulary).
#      include("indus_noattn.jl")
#
#  Trains three models, roughly 3-4 minutes each. Writes noattn_summary.csv
# =====================================================================
using Flux, MLUtils, Random, Statistics, Printf, DelimitedFiles, JLD2

# ---------------------------------------------------------------------
# attention-free block: LayerNorm + feed-forward, residual, no mixing
# across positions at all
# ---------------------------------------------------------------------
struct FFBlock
    ln::LayerNorm
    ff::Chain
    drop::Dropout
end
Flux.@functor FFBlock
FFBlock(d_model, d_ff, pdrop) =
    FFBlock(LayerNorm(d_model),
            Chain(Dense(d_model, d_ff, gelu), Dense(d_ff, d_model)),
            Dropout(pdrop))
(b::FFBlock)(x) = x .+ b.drop(b.ff(b.ln(x)))

struct NoAttnGPT
    tok_emb::Flux.Embedding
    pe::Matrix{Float32}
    use_pe::Bool
    drop::Dropout
    blocks::Vector{FFBlock}
    ln_f::LayerNorm
    head::Dense
    max_len::Int
end
Flux.@functor NoAttnGPT (tok_emb, drop, blocks, ln_f, head,)

function NoAttnGPT(; vocab_size, d_model = 64, n_layers = 3, d_ff = 256,
                     max_len = 20, pdrop = 0.1, use_pe = true)
    NoAttnGPT(Flux.Embedding(vocab_size, d_model),
              sinusoidal_pe(max_len, d_model),
              use_pe,
              Dropout(pdrop),
              [FFBlock(d_model, d_ff, pdrop) for _ in 1:n_layers],
              LayerNorm(d_model),
              Dense(d_model, vocab_size),
              max_len)
end

function (m::NoAttnGPT)(x::AbstractMatrix{Int})
    seq_len, batch = size(x)
    h = m.tok_emb(x)
    m.use_pe && (h = h .+ m.pe[:, 1:seq_len])
    h = m.drop(h)
    for blk in m.blocks; h = blk(h); end
    m.head(m.ln_f(h))
end

nparams(m) = sum(length, Flux.trainables(m))

# ---------------------------------------------------------------------
# training, reusing the split and vocabulary already in M
# ---------------------------------------------------------------------
function build_XY_NA(inss, M)
    ex = map(inss) do t
        ids = vcat(M.bos, M.encode(t), M.eos)
        length(ids) < M.maxl + 1 ? vcat(ids, fill(M.pad, M.maxl + 1 - length(ids))) :
                                   ids[1:M.maxl+1]
    end
    hcat([e[1:end-1] for e in ex]...), hcat([e[2:end] for e in ex]...)
end

const XTR_NA, YTR_NA = build_XY_NA(M.train, M)
const XTE_NA, YTE_NA = build_XY_NA(M.test,  M)

function train_noattn(; d_ff = 256, use_pe = true, label = "", seed = 7,
                        lr = 2f-4, batch = 32, epochs = 200, patience = 15)
    Random.seed!(seed)
    m = NoAttnGPT(vocab_size = M.nv, d_model = 64, n_layers = 3,
                  d_ff = d_ff, max_len = M.maxl, pdrop = 0.1, use_pe = use_pe)
    opt = Flux.setup(Adam(lr), m)
    loader = DataLoader((XTR_NA, YTR_NA); batchsize = batch, shuffle = true)
    best = Inf; bstate = deepcopy(Flux.state(m)); wait = 0; bep = 0
    t0 = time()
    for ep in 1:epochs
        Flux.trainmode!(m)
        for (xb, yb) in loader
            _, gs = Flux.withgradient(mm -> indus_loss(mm, xb, yb, M.nv, M.pad), m)
            Flux.update!(opt, m, gs[1])
        end
        e = indus_nll(m, XTE_NA, YTE_NA, M.nv, M.pad)
        if e < best - 1f-4
            best = e; bstate = deepcopy(Flux.state(m)); wait = 0; bep = ep
        else
            wait += 1; wait >= patience && break
        end
        ep % 25 == 0 && (@printf("    %s ep %3d  ppl %.2f  (%.0fs)\n",
                                 label, ep, exp(e), time()-t0); flush(stdout))
    end
    Flux.loadmodel!(m, bstate)
    Flux.testmode!(m)
    (model = m, ppl = exp(best), epoch = bep, params = nparams(m))
end

# ---------------------------------------------------------------------
# run
# ---------------------------------------------------------------------
@printf("full transformer (reference): ppl %.2f, %d parameters\n\n",
        M.ppl, nparams(M.model))

println("training: no attention, with positional encoding ...")
R_pe   = train_noattn(d_ff = 256, use_pe = true,  label = "noattn+pe")
@printf("  -> ppl %.2f at epoch %d, %d parameters\n\n", R_pe.ppl, R_pe.epoch, R_pe.params)

println("training: no attention, no positional encoding (pure bigram) ...")
R_nope = train_noattn(d_ff = 256, use_pe = false, label = "noattn-pe")
@printf("  -> ppl %.2f at epoch %d, %d parameters\n\n", R_nope.ppl, R_nope.epoch, R_nope.params)

# parameter-matched: widen the MLP until the totals are comparable, so a
# gap cannot be attributed to the attention-free model simply being smaller
target = nparams(M.model)
dff_wide = 256
let m0 = NoAttnGPT(vocab_size = M.nv, d_model = 64, n_layers = 3,
                   d_ff = 256, max_len = M.maxl, pdrop = 0.1)
    per_dff = (nparams(NoAttnGPT(vocab_size = M.nv, d_model = 64, n_layers = 3,
                                 d_ff = 512, max_len = M.maxl, pdrop = 0.1)) -
               nparams(m0)) / 256
    global dff_wide = 256 + round(Int, (target - nparams(m0)) / per_dff)
    dff_wide = max(256, dff_wide)
end
@printf("training: no attention, MLP widened to d_ff = %d (parameter-matched) ...\n", dff_wide)
R_wide = train_noattn(d_ff = dff_wide, use_pe = true, label = "noattn-wide")
@printf("  -> ppl %.2f at epoch %d, %d parameters\n\n", R_wide.ppl, R_wide.epoch, R_wide.params)

# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------
println("="^76)
println("IS ATTENTION NECESSARY?")
println("="^76)
@printf("  %-36s %10s %12s %10s\n", "model", "ppl", "parameters", "vs full")
@printf("  %-36s %10.2f %12d %10s\n", "full transformer (3 layers, 12 heads)",
        M.ppl, nparams(M.model), "-")
@printf("  %-36s %10.2f %12d %+10.2f\n", "no attention, with position",
        R_pe.ppl, R_pe.params, R_pe.ppl - M.ppl)
@printf("  %-36s %10.2f %12d %+10.2f\n", "no attention, no position",
        R_nope.ppl, R_nope.params, R_nope.ppl - M.ppl)
@printf("  %-36s %10.2f %12d %+10.2f\n", "no attention, parameter-matched",
        R_wide.ppl, R_wide.params, R_wide.ppl - M.ppl)
println()
println("  reference points from earlier runs:")
@printf("  %-36s %10.2f\n", "Witten-Bell bigram, same split", 25.50)
@printf("  %-36s %10.2f\n", "unigram", 69.45)
@printf("  %-36s %10.2f\n", "full model, attention ablated post hoc", 31.21)
@printf("  %-36s %10.2f\n", "trained on shuffled inscriptions", 59.30)
println("""
  Reading this. If the attention-free models land near the full
  transformer, attention is unnecessary at this corpus size and the
  model is a positional bigram. The no-position variant is the strict
  test: it cannot use context at all, only the current sign, so its
  perplexity is an empirical ceiling on a neurally parameterised bigram
  and is directly comparable to the Witten-Bell figure.

  Note the attention-free models train from scratch, so unlike the post
  hoc ablation at 31.21 there is no distribution shift to explain away
  any gap.""")

open("noattn_summary.csv","w") do io
    println(io, "model,perplexity,parameters,epoch")
    @printf(io, "full_transformer,%.4f,%d,%d\n", M.ppl, nparams(M.model), M.epoch)
    @printf(io, "noattn_with_pe,%.4f,%d,%d\n",   R_pe.ppl,   R_pe.params,   R_pe.epoch)
    @printf(io, "noattn_no_pe,%.4f,%d,%d\n",     R_nope.ppl, R_nope.params, R_nope.epoch)
    @printf(io, "noattn_param_matched,%.4f,%d,%d\n", R_wide.ppl, R_wide.params, R_wide.epoch)
end
JLD2.jldsave("indus_model_noattn.jld2";
             state = Flux.state(R_pe.model), ppl = R_pe.ppl, d_ff = 256, use_pe = true)
println("\nwrote noattn_summary.csv and indus_model_noattn.jld2")
