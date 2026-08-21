using Flux

function sinusoidal_pe(seq_len::Int, d_model::Int)
    pe = zeros(Float32, d_model, seq_len)
    for pos in 1:seq_len, i in 1:2:d_model
        denom = 10000f0 ^ ((i-1) / d_model)
        pe[i, pos] = sin((pos-1) / denom)
        i+1 <= d_model && (pe[i+1, pos] = cos((pos-1) / denom))
    end
    return pe
end

# CHANGED: optional `window`. A query at position i may attend to keys
# i-window+1 ... i. With window = seq_len (the default) the condition
# i - j >= window is never satisfied, so this is bit-identical to the
# original causal mask and every existing script keeps its behaviour.
function causal_mask(seq_len::Int, window::Int = seq_len)
    Float32[(j > i || i - j >= window) ? -Inf32 : 0f0 for i in 1:seq_len, j in 1:seq_len]
end

struct MHA
    Wq::Dense; Wk::Dense; Wv::Dense; Wo::Dense
    n_heads::Int; d_head::Int
end
Flux.@functor MHA
function MHA(d_model::Int, n_heads::Int)
    @assert d_model % n_heads == 0
    MHA(Dense(d_model,d_model), Dense(d_model,d_model), Dense(d_model,d_model),
        Dense(d_model,d_model), n_heads, div(d_model, n_heads))
end
function (m::MHA)(x, mask)
    d_model, seq_len, batch = size(x)
    H, dh = m.n_heads, m.d_head
    q,k,v = m.Wq(x), m.Wk(x), m.Wv(x)
    toh(t) = reshape(permutedims(reshape(t, dh,H,seq_len,batch),(1,3,2,4)), dh,seq_len,H*batch)
    q,k,v = toh(q), toh(k), toh(v)
    scores = Flux.batched_mul(Flux.batched_transpose(q), k) .* Float32(1/sqrt(dh))
    scores = scores .+ mask
    attn = Flux.softmax(scores; dims=2)
    out = Flux.batched_mul(v, Flux.batched_transpose(attn))
    out = reshape(permutedims(reshape(out, dh,seq_len,H,batch),(1,3,2,4)), d_model,seq_len,batch)
    return m.Wo(out)
end

struct Block
    ln1::LayerNorm; attn::MHA; ln2::LayerNorm; ff::Chain; drop::Dropout
end
Flux.@functor Block
function Block(d_model, n_heads, d_ff, pdrop)
    Block(LayerNorm(d_model), MHA(d_model,n_heads), LayerNorm(d_model),
          Chain(Dense(d_model,d_ff,gelu), Dense(d_ff,d_model)), Dropout(pdrop))
end
function (b::Block)(x, mask)
    x = x .+ b.drop(b.attn(b.ln1(x), mask))
    x = x .+ b.drop(b.ff(b.ln2(x)))
    return x
end

struct IndusGPT
    tok_emb::Flux.Embedding
    pe::Matrix{Float32}
    drop::Dropout
    blocks::Vector{Block}
    ln_f::LayerNorm
    head::Dense              # UNTIED, separate learnable output head
    max_len::Int
    window::Int              # CHANGED: attention span; max_len = unrestricted
end
Flux.@functor IndusGPT (tok_emb, drop, blocks, ln_f, head,)

function IndusGPT(; vocab_size, d_model=64, n_heads=4, n_layers=3, d_ff=256,
                    max_len=20, pdrop=0.1, window=max_len)     # CHANGED: window kwarg
    IndusGPT(Flux.Embedding(vocab_size, d_model),
             sinusoidal_pe(max_len, d_model),
             Dropout(pdrop),
             [Block(d_model,n_heads,d_ff,pdrop) for _ in 1:n_layers],
             LayerNorm(d_model),
             Dense(d_model, vocab_size),      # separate output head
             max_len,
             window)
end

function (m::IndusGPT)(x::AbstractMatrix{Int})
    seq_len, batch = size(x)
    h = m.tok_emb(x) .+ m.pe[:, 1:seq_len]
    h = m.drop(h)
    mask = Flux.Zygote.ignore(() -> causal_mask(seq_len, m.window))   # CHANGED
    for blk in m.blocks; h = blk(h, mask); end
    h = m.ln_f(h)
    return m.head(h)         # (vocab, seq, batch) — Dense broadcasts over dims 2,3
end
