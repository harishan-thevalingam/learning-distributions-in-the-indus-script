# ---- Indus script (EBUDS) data pipeline ----
# induscorpus.txt: one inscription per line, signs as whitespace-separated numbers.

const INDUS_PATH = "induscorpus.txt"

# Parse every line into a vector of sign-number strings (kept as strings so they're just tokens)
function load_inscriptions(path = INDUS_PATH)
    inscriptions = Vector{String}[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        signs = split(line)                 # whitespace-separated sign numbers
        push!(inscriptions, String.(signs))
    end
    return inscriptions
end

# ---- vocabulary with rare-sign collapsing (the <unk> design decision) ----
struct SignVocab
    stoi::Dict{String,Int}
    itos::Vector{String}
end

function build_sign_vocab(inscriptions; min_count::Int = 2)
    counts = Dict{String,Int}()
    for ins in inscriptions, s in ins
        counts[s] = get(counts, s, 0) + 1
    end
    # keep signs occurring >= min_count; collapse the rest to <unk>
    kept = sort([s for (s,c) in counts if c >= min_count])
    itos = vcat(["<pad>", "<bos>", "<eos>", "<unk>"], kept)
    stoi = Dict(s => i for (i,s) in enumerate(itos))
    return SignVocab(stoi, itos), counts
end

encode_signs(v::SignVocab, signs) = [get(v.stoi, s, v.stoi["<unk>"]) for s in signs]
decode_signs(v::SignVocab, ids)   = [v.itos[i] for i in ids]