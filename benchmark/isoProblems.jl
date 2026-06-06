using Random
using Paulimorphic

Random.seed!(1234)

function exampleGraph1(k::Int) #> Higher-degree hubs connected by low-degree vertices
    edges = NTuple{2,Int}[]
    hubs = Int[]
    nxt = 0

    for i in 0:k-1
        nxt += 1
        h = nxt
        push!(hubs, h)

        for _ in 1:((i == 0 || i == k-1) ? 4 : 3)
            nxt += 1
            push!(edges, (h, nxt))
        end
    end

    for i in 1:k-1 #> connector paths of distinct lengths
        prev = hubs[i]

        for _ in 1:i
            nxt += 1
            push!(edges, (prev, nxt))
            prev = nxt
        end

        push!(edges, (prev, hubs[i+1]))
    end
    
    SimpleGraph(nxt, edges)
end

function relabelGraph(g::SimpleGraph, perm::AbstractVector{Int})
    reorderedEdges = [(perm[i], perm[j]) for (i, j) in listEdges(g)]
    SimpleGraph(countVertices(g), reorderedEdges)
end

g1 = exampleGraph1(8)
perm = randperm(MersenneTwister(2026), countVertices(g1))
g1_prem = relabelGraph(g1, perm)
@assert isIsomorphic(g1, g1_prem)