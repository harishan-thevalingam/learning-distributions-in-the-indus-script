# =====================================================================
#  script_ramesh_check.jl
#
#  Replicates the checkable quantitative claims from
#    Rao, Yadav, Vahia, Joglekar, Adhikari & Mahadevan,
#    "A Markov Model of the Indus Script", PNAS 106(33):13685-13690 (2009)
#
#  ITEM 2  max transition probability among the 10 most frequent signs
#          paper: ~0.8, for the sign ranked 3rd -> sign ranked 2nd
#  ITEM 3  self-repetition: a specific frequent sign occurs as an
#          adjacent pair in a specific fraction of its inscriptions
#          paper: one sign occurs in 58 inscriptions, as a pair in 33
#  ITEM 4  altered (shuffled) text likelihood collapses relative to
#          the original -- paper's own example was a single adjacent
#          swap; here we use a full random shuffle (a bigger
#          perturbation), so the magnitude will not match, only the
#          direction and scale of the effect
#
#  ITEM 5 (West Asian vs Indus Valley likelihood) is DELIBERATELY
#  OMITTED. Checked against concordance-rev1.xls: every "site >= 900"
#  text is already verbatim inside induscorpus.txt, i.e. it's part of
#  EBUDS training data, not a held-out foreign set -- scoring it would
#  be circular. The real West Asian texts from PNAS Table 2 aren't
#  identifiable in the data on hand.
#
#  PURE COUNTING + one bigram model. No Flux, no training. Runs in a
#  few seconds. All local variable names below are suffixed _CHK to
#  avoid clashing with any consts already defined by indus_replicate.jl
#  or indus_model_v2.jl if you're running this in the same session.
#
#      using Pkg; Pkg.activate(".")
#      include("indus_data.jl")
#      inscriptions = load_inscriptions()
#      include("script_ramesh_check.jl")
# =====================================================================
using Random, Statistics, Printf

# ---------------------------------------------------------------------
# ITEM 2 -- max transition probability among top-10 frequent signs
# ---------------------------------------------------------------------
println("="^70)
println("ITEM 2  --  MAX TRANSITION PROBABILITY AMONG TOP-10 FREQUENT SIGNS")
println("="^70)

freq_CHK = Dict{String,Int}()
for ins in inscriptions, s in ins; freq_CHK[s] = get(freq_CHK, s, 0) + 1; end
ranked_CHK = sort(collect(freq_CHK); by = p -> -p[2])
top10_CHK = [p[1] for p in ranked_CHK[1:10]]
rankof_CHK = Dict(s => i for (i,s) in enumerate(top10_CHK))

println("top 10 by frequency:")
for s in top10_CHK
    @printf("  rank %2d : sign %-5s (count %d)\n", rankof_CHK[s], s, freq_CHK[s])
end

bi_CHK = Dict{String,Dict{String,Int}}()
for ins in inscriptions, (a,b) in zip(ins, ins[2:end])
    d = get!(bi_CHK, a, Dict{String,Int}())
    d[b] = get(d, b, 0) + 1
end

function find_max_transition_CHK(top10, bi)
    best = (0.0, "", "")
    for a in top10
        d = get(bi, a, nothing)
        d === nothing && continue
        tot = sum(values(d))
        for b in top10
            p = get(d, b, 0) / tot
            p > best[1] && (best = (p, a, b))
        end
    end
    best
end
p_CHK, a_CHK, b_CHK = find_max_transition_CHK(top10_CHK, bi_CHK)
@printf("\nmax P(sign_j | sign_i) restricted to top-10 = %.3f\n", p_CHK)
@printf("  sign %s (rank %d) -> sign %s (rank %d)\n", a_CHK, rankof_CHK[a_CHK], b_CHK, rankof_CHK[b_CHK])
println("  [paper: ~0.8, rank 3 -> rank 2]")

# ---------------------------------------------------------------------
# ITEM 3 -- self-repetition
# ---------------------------------------------------------------------
println("\n", "="^70)
println("ITEM 3  --  SELF-REPETITION (sign occurring as an adjacent pair)")
println("="^70)

contains_CHK = Dict{String,Int}()   # # inscriptions containing this sign at all
pairs_CHK    = Dict{String,Int}()   # # inscriptions where it appears as an adjacent pair
for ins in inscriptions
    for s in Set(ins)
        contains_CHK[s] = get(contains_CHK, s, 0) + 1
    end
    for i in 1:length(ins)-1
        if ins[i] == ins[i+1]
            pairs_CHK[ins[i]] = get(pairs_CHK, ins[i], 0) + 1
        end
    end
end

cand58_CHK = [(s,c,get(pairs_CHK,s,0)) for (s,c) in contains_CHK if c == 58]
println("signs occurring in exactly 58 inscriptions: ", cand58_CHK)

top_rep_CHK = sort(collect(pairs_CHK); by = p -> -p[2])
println("\ntop signs by pair-repeat count (sign : in N inscriptions, pair in M):")
for (s, m) in top_rep_CHK[1:min(8,length(top_rep_CHK))]
    @printf("  sign %-5s : in %3d inscriptions, pair in %2d\n", s, contains_CHK[s], m)
end
println("  [paper: a sign in 58 inscriptions, pair in 33]")

# ---------------------------------------------------------------------
# ITEM 4 -- altered (shuffled) text likelihood collapse
# Witten-Bell smoothed bigram, same style as the PLoS replication
# ---------------------------------------------------------------------
println("\n", "="^70)
println("ITEM 4  --  SHUFFLED-TEXT LIKELIHOOD COLLAPSE")
println("="^70)

allsigns_CHK = sort(unique(s for ins in inscriptions for s in ins))
NUMSIGNS_CHK = length(allsigns_CHK)

struct WBModelCHK
    bi::Dict{String,Dict{String,Int}}
    uni::Dict{String,Int}
    U::Int
    Tu::Int
end
function WBModelCHK(texts, nsign)
    bi = Dict{String,Dict{String,Int}}(); uni = Dict{String,Int}()
    for t in texts
        seq = vcat("#", t, "\$")
        for i in 2:length(seq)
            d = get!(bi, seq[i-1], Dict{String,Int}())
            d[seq[i]] = get(d, seq[i], 0) + 1
        end
        for w in t; uni[w] = get(uni, w, 0) + 1; end
    end
    WBModelCHK(bi, uni, sum(values(uni)), length(uni))
end
function p_uni_CHK(m::WBModelCHK, w, nsign)
    c = get(m.uni, w, 0)
    c > 0 ? c/(m.U+m.Tu) : (m.Tu/nsign)/(m.U+m.Tu)
end
function p_bi_CHK(m::WBModelCHK, w, h, nsign)
    pl = p_uni_CHK(m, w, nsign)
    d = get(m.bi, h, nothing)
    d === nothing && return pl
    n = sum(values(d)); T = length(d)
    (get(d, w, 0) + T*pl) / (n + T)
end
function loglik_CHK(m::WBModelCHK, t, nsign)
    seq = vcat("#", t, "\$")
    sum(log(p_bi_CHK(m, seq[i], seq[i-1], nsign)) for i in 2:length(seq))
end

model_CHK = WBModelCHK(inscriptions, NUMSIGNS_CHK)
rng_CHK = MersenneTwister(0)
sample_CHK = [t for t in inscriptions if length(t) >= 3]
shuffle!(rng_CHK, sample_CHK)
sample_CHK = sample_CHK[1:min(100, length(sample_CHK))]

deltas_CHK = Float64[]
for t in sample_CHK
    orig = loglik_CHK(model_CHK, t, NUMSIGNS_CHK)
    alt = shuffle(rng_CHK, copy(t))
    altered = loglik_CHK(model_CHK, alt, NUMSIGNS_CHK)
    push!(deltas_CHK, orig - altered)
end
md_CHK = mean(deltas_CHK)
@printf("mean [log-lik(original) - log-lik(shuffled)] over %d texts: %.2f nats\n", length(sample_CHK), md_CHK)
@printf("  i.e. originals are on average e^%.2f = %.1fx more likely than a full random\n", md_CHK, exp(md_CHK))
println("  shuffle of their own signs")
println("  [paper's own example: a SINGLE adjacent swap dropped likelihood ~15,000x;")
println("   a full shuffle here is a much bigger perturbation, so magnitudes aren't")
println("   comparable -- only the direction/scale of the effect is the replication]")

println("\n", "="^70)
println("DONE  (item 5, West Asian texts, intentionally omitted -- see header)")
println("="^70)
