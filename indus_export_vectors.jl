# =====================================================================
#  indus_export_vectors.jl
#
#  Exports three different vector representations of each sign, so the
#  UMAP can be run on all of them and compared. No training -- loads the
#  saved model via indus_vault.jl.
#
#  WHY THREE. The earlier cosine test compared 64-dimensional embeddings
#  against 754-dimensional bigram counts and found 0/30 vs 8/30. That
#  comparison is not clean: the bigram vector has canonical coordinates
#  (axis k means "the next sign is sign k"), whereas the 64 embedding
#  dimensions are an arbitrary internal basis the model happened to
#  settle into. Cosine is interpretable in the first space and incidental
#  in the second, so a null result for the embeddings does not show the
#  model lacks the structure.
#
#  The fix is representation (B): put the transformer in the SAME
#  coordinates as the bigram by asking what it predicts.
#
#    (A) emb64.csv     raw token embeddings, 64 dims.
#                      What the earlier test used.
#
#    (B) ctx_model.csv model's predicted next-sign distribution for each
#                      sign, averaged over every context in which that
#                      sign occurs. Same axes as the bigram, so directly
#                      comparable, and independent of the internal basis.
#
#    (C) ctx_bigram.csv empirical next-sign distribution, i.e. plain
#                      counts. The reference point.
#
#  (B) vs (C) is the honest question: does the transformer's learned
#  successor distribution group signs better than raw counts do?
#
#  USAGE (fresh session)
#      using Pkg; Pkg.activate(".")
#      include("indus_data.jl")
#      inscriptions = load_inscriptions()
#      include("indus_model_v2.jl")
#      include("indus_vault.jl")
#      M = indus_get(inscriptions)
#      include("indus_export_vectors.jl")
#
#  Writes emb64.csv, ctx_model.csv, ctx_bigram.csv, sign_meta.csv
#  Runtime: seconds.
# =====================================================================
using Flux, Statistics, Printf, DelimitedFiles

function export_sign_vectors(M, inscriptions)
    signs = M.signs
    ns    = length(signs)
    idx   = Dict(s => i for (i,s) in enumerate(signs))

    # ---------------- (A) raw embeddings ----------------
    E = M.model.tok_emb.weight                      # (d_model, vocab)
    emb = permutedims(hcat([Float64.(E[:, M.stoi[s]]) for s in signs]...))
    writedlm("emb64.csv", emb, ',')
    println("emb64.csv        $(size(emb,1)) x $(size(emb,2))")

    # ---------------- (B) model successor distributions ----------------
    # For every position holding sign s, take the model's predicted
    # distribution over the NEXT token, and average across all such
    # positions. Columns are vocabulary items, matching the bigram.
    acc = zeros(Float64, ns, M.nv)
    hits = zeros(Int, ns)
    Flux.testmode!(M.model)
    for ins in inscriptions
        ids = vcat(M.bos, M.encode(ins))
        S = length(ids)
        S < 2 && continue
        S > M.maxl && (ids = ids[1:M.maxl]; S = M.maxl)
        probs = Flux.softmax(M.model(reshape(ids, S, 1)); dims = 1)   # (vocab, S, 1)
        for i in 1:S
            tok = ids[i]
            tok in (M.bos, M.eos, M.pad) && continue
            s = M.itos[tok]
            haskey(idx, s) || continue
            r = idx[s]
            @views acc[r, :] .+= Float64.(probs[:, i, 1])
            hits[r] += 1
        end
    end
    for r in 1:ns
        hits[r] > 0 && (@views acc[r, :] ./= hits[r])
    end
    writedlm("ctx_model.csv", acc, ',')
    println("ctx_model.csv    $(size(acc,1)) x $(size(acc,2))   (canonical coords)")

    # ---------------- (C) empirical successor distributions ----------------
    succ = zeros(Float64, ns, ns)
    pred = zeros(Float64, ns, ns)
    for ins in inscriptions, (a,b) in zip(ins, ins[2:end])
        (haskey(idx,a) && haskey(idx,b)) || continue
        succ[idx[a], idx[b]] += 1
        pred[idx[b], idx[a]] += 1
    end
    for i in 1:ns
        t = sum(succ[i,:]); t > 0 && (succ[i,:] ./= t)
        t = sum(pred[i,:]); t > 0 && (pred[i,:] ./= t)
    end
    writedlm("ctx_bigram.csv", hcat(succ, pred), ',')
    println("ctx_bigram.csv   $(ns) x $(2ns)   (successor ++ predecessor)")

    # ---------------- metadata ----------------
    # end_rate / start_rate are continuous versions of the ender/beginner
    # roles, far more informative for colouring a plot than three
    # hand-picked signs.
    cnt   = zeros(Int, ns)
    fin   = zeros(Int, ns)
    init  = zeros(Int, ns)
    inpos = zeros(Int, ns)
    for ins in inscriptions
        for s in ins
            haskey(idx, s) && (cnt[idx[s]] += 1)
        end
        haskey(idx, ins[end]) && (fin[idx[ins[end]]]  += 1)
        haskey(idx, ins[1])   && (init[idx[ins[1]]]   += 1)
        for s in unique(ins)
            haskey(idx, s) && (inpos[idx[s]] += 1)
        end
    end
    open("sign_meta.csv", "w") do io
        println(io, "sign,count,texts,end_rate,start_rate")
        for i in 1:ns
            er = inpos[i] > 0 ? fin[i]/inpos[i]  : 0.0
            sr = inpos[i] > 0 ? init[i]/inpos[i] : 0.0
            @printf(io, "%s,%d,%d,%.4f,%.4f\n", signs[i], cnt[i], inpos[i], er, sr)
        end
    end
    println("sign_meta.csv    $(ns) rows  (count, texts, end_rate, start_rate)")
    println()
    println("Now run:  python indus_umap.py")
end

export_sign_vectors(M, inscriptions)
