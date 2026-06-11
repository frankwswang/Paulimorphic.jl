using Test
using Random
using Paulimorphic

#> Undirected path Pn on `n` vertices: 1-2-3-…-n.
function makePath(n::Int)
    g = SimpleGraph(n)
    for i in 1:(n-1); attachEdge!(g, (i, i+1)) end
    g
end

#> Complete graph Kn.
function makeKn(n::Int)
    g = SimpleGraph(n)
    for i in 1:n, j in (i+1):n; attachEdge!(g, (i, j)) end
    g
end

#> Cycle Cn (n ≥ 3).
function makeCn(n::Int)
    g = makePath(n)
    attachEdge!(g, (n, 1))
    g
end

#> Star K_{1,n}: centre vertex 1 connected to leaves 2:(n+1).
function makeK1n(n::Int)
    g = SimpleGraph(n + 1)
    for i in 2:(n+1); attachEdge!(g, (1, i)) end
    g
end

#> Complete bipartite K_{m,n}: left vertices 1:m, right vertices (m+1):(m+n).
function makeKmn(m::Int, n::Int)
    g = SimpleGraph(m + n)
    for i in 1:m, j in (m+1):(m+n); attachEdge!(g, (i, j)) end
    g
end

#> Two disjoint triangles on 6 vertices (2-regular, disconnected).
makeTwoTriangles() = SimpleGraph(6, [(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)])

#> Sorted degree sequence of `g`.
sortedDegrees(g::SimpleGraph) = sort(listDegrees(g))

#> Return a copy of `g` with every vertex `v` renamed to `perm[v]`. If `perm` is a
#> permutation of `1:countVertices(g)`, the result is isomorphic to `g` by construction.
function relabel(g::SimpleGraph{T}, perm::AbstractVector{<:Integer}) where {T<:Integer}
    n = countVertices(g)
    @assert length(perm) == n
    @assert sort!(collect(perm)) == 1:n
    h = SimpleGraph(countVertices(g), T)
    for (i, j) in listEdges(g); attachEdge!(h, (perm[i], perm[j])) end
    h
end

#> An Erdos-Renyi-style random simple graph on `n` vertices, each possible edge included
#> independently with probability `p`.
function randGraph(rng::AbstractRNG, n::Integer, p::Real)
    g = SimpleGraph(n, Int)
    for i in 1:n, j in (i+1):n
        rand(rng) < p && attachEdge!(g, (i, j))
    end
    g
end

#> All permutations of `1:n`, generated without external dependencies. Only for small `n`.
function allPerms(n::Integer)
    n <= 0 && return [Int[]]
    out = Vector{Vector{Int}}()
    for sub in allPerms(n - 1), pos in 1:n
        p = copy(sub)
        insert!(p, pos, n)
        push!(out, p)
    end
    out
end

#> Brute-force isomorphism oracle for small graphs: try every vertex permutation and check
#> whether any one preserves all edges. Exponential; only for tiny `n`.
function bruteIso(g1::SimpleGraph, g2::SimpleGraph)
    n = countVertices(g1)
    countVertices(g2) == n || return false
    countEdges(g1) == countEdges(g2) || return false
    e1 = listEdges(g1)
    for p in allPerms(n)
        all(((i, j),) -> containEdge(g2, (p[i], p[j])), e1) && return true
    end
    false
end

#> Independently verify that `m`, a vector of `g1Vertex => g2Vertex` pairs, is a genuine
#> isomorphism from `g1` onto `g2`: a bijection on the vertex sets that preserves
#> adjacency in both directions.
function isValidIso(g1::SimpleGraph, g2::SimpleGraph, m::AbstractVector{<:Pair})
    n = countVertices(g1)
    countVertices(g2) == n || return false
    length(m) == n || return false

    src = first.(m)
    dst = last.(m)
    #> Bijection: both sides must be exactly the vertex set 1:n (no repeats, no gaps)
    Set(src) == Set(1:n) || return false
    Set(dst) == Set(1:n) || return false

    #> Same number of edges, plus every g1-edge mapped to a g2-edge, gives a true iso
    countEdges(g1) == countEdges(g2) || return false
    d = Dict(src .=> dst)
    for (i, j) in listEdges(g1)
        containEdge(g2, (d[i], d[j])) || return false
    end
    #> Reverse direction, for good measure
    dinv = Dict(dst .=> src)
    for (i, j) in listEdges(g2)
        containEdge(g1, (dinv[i], dinv[j])) || return false
    end
    true
end

#> Check that `root` is genuinely a root graph of `g`, i.e., L(root) ≅ g. (This subsumes
#> vertex-count, edge-count, and degree-sequence agreement; `isIsomorphic` itself is
#> independently validated in its own test set below.)
isValidRoot(g::SimpleGraph, root::SimpleGraph) = isIsomorphic(genLineGraph(root), g)


@testset "Graphs.jl" begin

# ──────────────────────────────────────────────────────────────────────────────
# 1. SimpleGraph construction
# ──────────────────────────────────────────────────────────────────────────────
@testset "SimpleGraph construction" begin
    @testset "order-0 graph" begin
        g = SimpleGraph(0)
        @test countVertices(g) == 0
        @test countEdges(g) == 0
        @test isempty(listEdges(g))
    end

    @testset "edgeless graph" begin
        g = SimpleGraph(5)
        @test countVertices(g) == 5
        @test countEdges(g) == 0
        @test listDegrees(g) == zeros(Int, 5)
    end

    @testset "construction from an edge list" begin
        g = SimpleGraph(3, [(2, 3), (1, 2), (1, 3)])
        @test countVertices(g) == 3
        @test countEdges(g) == 3
        @test listEdges(g) == [(1, 2), (1, 3), (2, 3)]
    end

    @testset "invalid or duplicate edges silently ignored by default" begin
        #> Self-loop, out-of-bounds (×2), valid, duplicate of the valid edge
        g = SimpleGraph(3, [(1, 1), (0, 2), (1, 4), (1, 2), (1, 2), (2, 1)])
        @test countEdges(g) == 1
        @test listEdges(g) == [(1, 2)]
    end

    @testset "explicitError=true throws on an invalid or duplicate edge" begin
        @test_throws DomainError SimpleGraph(3, [(1, 1)], true)         #> self-loop
        @test_throws DomainError SimpleGraph(3, [(0, 1)], true)         #> out of bounds
        @test_throws DomainError SimpleGraph(3, [(1, 4)], true)         #> out of bounds
        @test_throws DomainError SimpleGraph(3, [(1, 2), (2, 1)], true) #> duplicate
    end

    @testset "negative order throws" begin
        @test_throws DomainError SimpleGraph(-1)
        @test_throws DomainError SimpleGraph(-1, Int)
    end

    @testset "vertex-label type parameter" begin
        @test SimpleGraph(3) isa SimpleGraph{Int}
        @test SimpleGraph(Int8(3)) isa SimpleGraph{Int8} #> `T` defaults to `typeof(order)`
        @test SimpleGraph(3, Int16) isa SimpleGraph{Int16}

        #> `T` is inferred from the edge-list element type
        g = SimpleGraph(3, NTuple{2, Int16}[(1, 2), (2, 3)])
        @test g isa SimpleGraph{Int16}
        @test listEdges(g) isa Vector{NTuple{2, Int16}}
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# 2. Edge operations: attachEdge! / removeEdge! / containEdge
# ──────────────────────────────────────────────────────────────────────────────
@testset "attachEdge! / removeEdge! / containEdge" begin
    g = SimpleGraph(4)

    @test  attachEdge!(g, (1, 2)) #> new edge
    @test  attachEdge!(g, (2, 3)) #> new edge
    @test !attachEdge!(g, (1, 2)) #> duplicate
    @test !attachEdge!(g, (2, 1)) #> duplicate (reversed)
    @test !attachEdge!(g, (3, 3)) #> self-loop
    @test !attachEdge!(g, (0, 2)) #> out of bounds
    @test !attachEdge!(g, (1, 5)) #> out of bounds

    @test  containEdge(g, (1, 2))
    @test  containEdge(g, (2, 1)) #> undirected
    @test !containEdge(g, (1, 3))
    @test !containEdge(g, (1, 1)) #> self-loop always absent
    @test !containEdge(g, (0, 2)) #> out of bounds always absent

    @test  removeEdge!(g, (2, 1)) #> existing edge (reversed input)
    @test !containEdge(g, (1, 2))
    @test !removeEdge!(g, (2, 1)) #> already gone
    @test !removeEdge!(g, (3, 3)) #> self-loop
    @test !removeEdge!(g, (0, 2)) #> out of bounds
    @test listEdges(g) == [(2, 3)] #> removal reflected in the edge list

    @testset "adjacency symmetry" begin
        g2 = SimpleGraph(5, [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)])
        for (a, b) in listEdges(g2) #> every edge should be visible from both ends
            @test containEdge(g2, (a, b))
            @test containEdge(g2, (b, a))
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# 3. Vertex/edge queries: countVertices / getDegree / countEdges / listEdges /
#    listDegrees
# ──────────────────────────────────────────────────────────────────────────────
@testset "countVertices / getDegree / countEdges / listEdges / listDegrees" begin
    @testset "edgeless graph" begin
        g = SimpleGraph(3)
        @test countVertices(g) == 3
        @test countEdges(g) == 0
        @test isempty(listEdges(g))
        @test listDegrees(g) == [0, 0, 0]
    end

    @testset "K3 (triangle)" begin
        g = makeKn(3)
        @test countEdges(g) == 3
        @test listEdges(g) == [(1, 2), (1, 3), (2, 3)]
        @test listDegrees(g) == [2, 2, 2]
    end

    @testset "star K_{1,4}" begin
        g = makeK1n(4)
        @test countEdges(g) == 4
        @test sortedDegrees(g) == [1, 1, 1, 1, 4]
    end

    @testset "getDegree" begin
        g = makeK1n(3)
        @test getDegree(g, 1) == 3
        @test all(getDegree(g, v) == 1 for v in 2:4)
        @test getDegree(makePath(4), 2) == 2
        @test_throws DomainError getDegree(g,  0)  #> non-positive vertex label
        @test_throws DomainError getDegree(g, -2)
        @test_throws DomainError getDegree(g,  5)
    end

    @testset "listEdges is lexicographically sorted" begin
        g = SimpleGraph(4, [(3, 4), (1, 2), (2, 4), (1, 3)])
        @test issorted(listEdges(g))
    end

    @testset "consistency between degree and edge counts" begin
        g = makeKmn(2, 3)
        @test sum(listDegrees(g)) == 2 * countEdges(g) #> Handshake lemma
        @test listDegrees(g) == [getDegree(g, v) for v in 1:countVertices(g)]
    end

    @testset "countEdges detects corrupted adjacency data" begin
        g = SimpleGraph(2)
        push!(g.adjacency[begin], 2) #> One-sided insertion breaks adjacency symmetry
        @test_throws ArgumentError countEdges(g)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# 4. genLineGraph
# ──────────────────────────────────────────────────────────────────────────────
@testset "genLineGraph" begin
    @testset "graphs without edges have empty line graphs" begin
        for g in (SimpleGraph(0), SimpleGraph(5)) #> order-0 and 5 isolated vertices
            lg = genLineGraph(g)
            @test countVertices(lg) == 0
            @test countEdges(lg) == 0
        end
    end

    @testset "L(K2) = K1" begin
        lg = genLineGraph(makeKn(2))
        @test countVertices(lg) == 1
        @test countEdges(lg) == 0
    end

    @testset "L(P3) = P2 and L(P4) = P3" begin
        lg3 = genLineGraph(makePath(3))
        @test countVertices(lg3) == 2
        @test listEdges(lg3) == [(1, 2)]

        lg4 = genLineGraph(makePath(4))
        @test countVertices(lg4) == 3
        @test listEdges(lg4) == [(1, 2), (2, 3)]
    end

    @testset "L(K3) = K3" begin
        @test isIsomorphic(genLineGraph(makeKn(3)), makeKn(3))
    end

    @testset "L(K_{1,n}) = Kn" begin
        for n in 2:5
            lg = genLineGraph(makeK1n(n))
            @test countVertices(lg) == n
            @test countEdges(lg) == n * (n - 1) ÷ 2
            @test sortedDegrees(lg) == fill(n - 1, n)
        end
    end

    @testset "L(Cn) = Cn" begin
        for n in 4:6
            @test isIsomorphic(genLineGraph(makeCn(n)), makeCn(n))
        end
    end

    @testset "L(K4) = octahedron (6 vertices, 4-regular)" begin
        lg = genLineGraph(makeKn(4))
        @test countVertices(lg) == 6
        @test countEdges(lg) == 12
        @test sortedDegrees(lg) == fill(4, 6)
    end

    @testset "two disjoint edges: line graph is two isolated vertices" begin
        lg = genLineGraph(SimpleGraph(4, [(1, 2), (3, 4)]))
        @test countVertices(lg) == 2
        @test countEdges(lg) == 0
    end

    @testset "vertex i of L(g) corresponds to listEdges(g)[i]" begin
        g = SimpleGraph(5, [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
        edges = listEdges(g)
        lg = genLineGraph(g)
        for i in eachindex(edges), j in (i+1):length(edges)
            l1, r1 = edges[i]
            l2, r2 = edges[j]
            sharing = l1 == l2 || l1 == r2 || l2 == r1 || r1 == r2
            @test containEdge(lg, (i, j)) == sharing
        end
    end

    @testset "structural invariants on pseudo-random graphs" begin
        rng = MersenneTwister(1234)
        bl1 = bl2 = bl3 = true

        for trial in 1:60
            order = rand(rng, 1:14)
            g = randGraph(rng, order, rand(rng, (0.1, 0.3, 0.5, 0.8)))
            edges = listEdges(g)
            lg = genLineGraph(g)

            #>> Vertex count of L(g) equals edge count of g
            bl1 &= countVertices(lg) == countEdges(g)
            bl1 || (println("Failed at trail $trail for bl1..."); break)

            #>> Whitney's edge-count formula: |E(L(g))| = Σᵥ C(dᵥ, 2)
            bl2 &= countEdges(lg) == sum(d * (d - 1) ÷ 2 for d in listDegrees(g); init=0)
            bl2 || (println("Failed at trail $trail for bl2..."); break)

            #>> Degree of the line-graph vertex k for edge (i, j) is dᵢ + dⱼ - 2
            bl3 &= all(enumerate(edges)) do (k, (i, j))
                getDegree(lg, k) == getDegree(g, i) + getDegree(g, j) - 2
            end
            bl3 || (println("Failed at trail $trail for bl3..."); break)
        end

        @test bl1
        @test bl2
        @test bl3
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# 5. listComponents / decompose
# ──────────────────────────────────────────────────────────────────────────────
@testset "listComponents" begin
    @testset "order-0 graph" begin
        @test isempty(listComponents(SimpleGraph(0)))
    end

    @testset "single vertex" begin
        @test listComponents(SimpleGraph(1)) == [[1]]
    end

    @testset "connected graph → one component" begin
        @test listComponents(makeKn(4)) == [[1, 2, 3, 4]]
    end

    @testset "isolated vertices → singleton components" begin
        @test listComponents(SimpleGraph(3)) == [[1], [2], [3]]
    end

    @testset "two disjoint K2s" begin
        comps = listComponents(SimpleGraph(4, [(1, 2), (3, 4)]))
        @test comps == [[1, 2], [3, 4]]
    end

    @testset "components sorted internally and by first vertex" begin
        comps = listComponents(SimpleGraph(6, [(5, 6), (1, 3)]))
        @test all(issorted, comps)
        @test issorted(first.(comps))
    end
end

@testset "decompose" begin
    @testset "order-0 graph" begin
        comps, subgraphs = decompose(SimpleGraph(0))
        @test isempty(comps)
        @test isempty(subgraphs)
    end

    @testset "agrees with listComponents; exact relabelling" begin
        g = SimpleGraph(6, [(1, 2), (2, 3), (4, 5)])
        res = decompose(g)
        @test res isa Pair
        comps, subgraphs = res
        @test comps == listComponents(g) == [[1, 2, 3], [4, 5], [6]]
        @test length(subgraphs) == 3
        @test countVertices(subgraphs[1]) == 3
        @test listEdges(subgraphs[1]) == [(1, 2), (2, 3)]
        @test countVertices(subgraphs[2]) == 2
        @test listEdges(subgraphs[2]) == [(1, 2)]
        @test countVertices(subgraphs[3]) == 1
        @test countEdges(subgraphs[3]) == 0
    end

    @testset "connected graph → trivial decomposition" begin
        comps, subgraphs = decompose(makeCn(5))
        @test length(comps) == length(subgraphs) == 1
        @test countVertices(subgraphs[1]) == 5
        @test countEdges(subgraphs[1]) == 5
    end

    @testset "two disjoint triangles → two K3 subgraphs" begin
        comps, subgraphs = decompose(makeTwoTriangles())
        @test length(comps) == 2
        for sg in subgraphs
            @test isIsomorphic(sg, makeKn(3))
        end
    end

    @testset "vertices relabelled 1:length(component)" begin
        #> A K3 on vertices {4, 5, 6} should appear as a subgraph on vertices {1, 2, 3}
        g = SimpleGraph(6, [(4, 5), (5, 6), (4, 6)])
        comps, subgraphs = decompose(g)
        triIdx = findfirst(c -> length(c) == 3, comps)
        @test comps[triIdx] == [4, 5, 6]
        @test listEdges(subgraphs[triIdx]) == [(1, 2), (1, 3), (2, 3)]
    end

    @testset "edge count preserved across decomposition" begin
        g = SimpleGraph(8, [(1, 2), (2, 3), (3, 1), (5, 6), (6, 7), (7, 8), (5, 8)])
        _, subgraphs = decompose(g)
        @test sum(countEdges, subgraphs) == countEdges(g)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# 6. breadthFirstSearch
# ──────────────────────────────────────────────────────────────────────────────
@testset "breadthFirstSearch" begin
    @testset "deterministic traversal order (convenience method)" begin
        #> Newly discovered children are sorted before being enqueued
        @test breadthFirstSearch(makePath(5)) == [1, 2, 3, 4, 5]
        @test breadthFirstSearch(makeK1n(3)) == [1, 2, 3, 4]
        @test breadthFirstSearch(makeCn(6)) == [1, 2, 6, 3, 5, 4]
        #> Starting from an interior vertex
        @test breadthFirstSearch(makePath(5), 3) == [3, 2, 4, 1, 5]
        @test breadthFirstSearch(makeCn(6), 4) == [4, 3, 5, 2, 6, 1]
    end

    @testset "disconnected graphs" begin
        g = SimpleGraph(6, [(5, 6), (1, 3)])
        #> `startingPoint=missing`: all components, rooted in increasing vertex order
        @test breadthFirstSearch(g) == [1, 3, 2, 4, 5, 6]
        @test breadthFirstSearch(g, missing) == [1, 3, 2, 4, 5, 6]
        #> Explicit starting point: only the reachable component is traversed
        @test breadthFirstSearch(g, 5) == [5, 6]
        @test breadthFirstSearch(g, 2) == [2]
    end

    @testset "order-0 graph" begin
        @test isempty(breadthFirstSearch(SimpleGraph(0)))
        v, nVisited = breadthFirstSearch(_ -> true, SimpleGraph(0), missing)
        @test v == 0
        @test nVisited == 0
    end

    @testset "predicate method: early termination" begin
        g = makePath(5)

        #> The first vertex (in BFS order) satisfying `f`, with its dequeue position
        @test breadthFirstSearch(==(3), g, 1) == (3, 3) #> BFS order from 1: 1, 2, 3, …

        #> A predicate matching the starting vertex terminates immediately
        @test breadthFirstSearch(==(2), g, 2) == (2, 1)

        #> `f` is called exactly once per dequeued vertex, in BFS order
        visited = Int[]
        v, nVisited = breadthFirstSearch(g, 1) do node
            push!(visited, node)
            node == 4
        end
        @test (v, nVisited) == (4, 4)
        @test visited == [1, 2, 3, 4]
    end

    @testset "predicate method: not found returns (0, nVisited)" begin
        g = SimpleGraph(6, [(5, 6), (1, 3)])

        #> Search restricted to one component: `nVisited` is the component size
        @test breadthFirstSearch(_ -> false, g, 5) == (0, 2)

        #> `startingPoint=missing`: every vertex is dequeued exactly once
        counter = Ref(0)
        v, nVisited = breadthFirstSearch(g, missing) do _
            counter[] += 1
            false
        end
        @test (v, nVisited) == (0, 6)
        @test counter[] == 6

        #> A vertex unreachable from the starting point is never found
        @test breadthFirstSearch(==(5), g, 1) == (0, 2) #> Only component {1, 3} searched
    end

    @testset "cache!Self buffer" begin
        g = SimpleGraph(6, [(5, 6), (1, 3)])

        #> Discovery order is recorded in the first `nVisited` slots
        buf = fill(-1, 6)
        _, nVisited = breadthFirstSearch(_ -> false, g, 5, buf)
        @test nVisited == 2
        @test buf[1:2] == [5, 6]
        @test buf[3:end] == fill(-1, 4) #> Slots beyond `nVisited` are untouched

        #> A buffer longer than the graph order is allowed
        buf2 = fill(-1, 8)
        _, nVisited2 = breadthFirstSearch(_ -> false, g, missing, buf2)
        @test nVisited2 == 6
        @test buf2[1:6] == [1, 3, 2, 4, 5, 6]
        @test buf2[7:8] == [-1, -1]

        #> A buffer shorter than the graph order throws
        @test_throws ArgumentError breadthFirstSearch(_ -> false, g, missing, fill(0, 5))
    end

    @testset "invalid starting point throws" begin
        g = makePath(3)
        @test_throws DomainError breadthFirstSearch(_ -> false, g, 0)
        @test_throws DomainError breadthFirstSearch(_ -> false, g, 4)
        @test_throws DomainError breadthFirstSearch(g, -1)
    end

    @testset "non-Int vertex-label type" begin
        g = SimpleGraph(4, NTuple{2, Int16}[(1, 2), (2, 3), (3, 4)])
        ord = breadthFirstSearch(g, Int16(2))
        @test ord isa Vector{Int16}
        @test ord == [2, 1, 3, 4]

        v, nVisited = breadthFirstSearch(_ -> false, g, Int16(2))
        @test v == 0 #> The `zero(T)` sentinel for "not found"
        @test nVisited == 4
    end

    @testset "consistency with listComponents on pseudo-random graphs" begin
        rng = MersenneTwister(1234)
        bl1 = bl2 = true
        for _ in 1:30
            g = randGraph(rng, rand(rng, 1:12), rand(rng, (0.15, 0.3, 0.6)))
            #>> The full traversal is a permutation of all vertices
            bl1 &= sort(breadthFirstSearch(g)) == 1:countVertices(g)
            #>> Vertices reachable from each component root match `listComponents`
            for comp in listComponents(g)
                bl2 &= sort(breadthFirstSearch(g, first(comp))) == comp
            end
        end
        @test bl1
        @test bl2
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# 7. isIsomorphic
# ──────────────────────────────────────────────────────────────────────────────
@testset "isIsomorphic" begin
    #> Common fixtures (read-only within this test set)
    path4  = makePath(4)        #> P4, degree sequence (1, 2, 2, 1)
    star3  = makeK1n(3)         #> K_{1,3}, degree sequence (3, 1, 1, 1)
    cycle6 = makeCn(6)          #> C6, 2-regular, connected
    twoTri = makeTwoTriangles() #> 2 × C3, 2-regular, disconnected

    @testset "size / edge-count / degree-sequence mismatches" begin
        #> Different vertex counts
        @test !isIsomorphic(SimpleGraph(3, Int), SimpleGraph(4, Int))
        #> Same order, different edge counts
        @test !isIsomorphic(path4, SimpleGraph(4, [(1, 2)]))
        @test !isIsomorphic(SimpleGraph(4, [(1, 2)]), SimpleGraph(4, [(1, 2), (3, 4)]))
        #> Same order, same edge count, different degree sequence: P4 vs star K_{1,3}
        @test countEdges(path4) == countEdges(star3)
        @test !isIsomorphic(path4, star3)
        @test !isIsomorphic(star3, path4)
    end

    @testset "edgeless and order-0 graphs" begin
        #> Two edgeless graphs of equal order are isomorphic; the identity is appended
        buf = Pair{Int, Int}[]
        @test isIsomorphic(SimpleGraph(3, Int), SimpleGraph(3, Int), buf)
        @test buf == [1 => 1, 2 => 2, 3 => 3]

        #> Edgeless graphs of different order are not isomorphic
        @test !isIsomorphic(SimpleGraph(2, Int), SimpleGraph(3, Int))

        #> Order-0 graphs: isomorphic, empty mapping
        buf0 = Pair{Int, Int}[]
        @test isIsomorphic(SimpleGraph(0, Int), SimpleGraph(0, Int), buf0)
        @test isempty(buf0)
    end

    @testset "isomorphic pairs (with mapping validation)" begin
        for g in (path4, cycle6, makeKn(4))
            n = countVertices(g)
            h = relabel(g, randperm(MersenneTwister(7n + 1), n))
            buf = Pair{Int, Int}[]
            @test isIsomorphic(g, h, buf)
            @test isValidIso(g, h, buf)
        end

        #> A graph is isomorphic to itself
        for g in (path4, cycle6, twoTri, makeKn(4))
            buf = Pair{Int, Int}[]
            @test isIsomorphic(g, g, buf)
            @test isValidIso(g, g, buf)
        end
    end

    @testset "non-isomorphic with identical degree sequence" begin
        #> C6 and 2 × C3 are both 2-regular on 6 vertices but not isomorphic
        @test sortedDegrees(cycle6) == sortedDegrees(twoTri)
        @test !isIsomorphic(cycle6, twoTri)
        @test !isIsomorphic(twoTri, cycle6)
        #> Cross-check against the brute-force oracle
        @test !bruteIso(cycle6, twoTri)
    end

    @testset "disconnected graphs" begin
        #> K2 + K3 + isolated vertex, relabeled
        g = SimpleGraph(7, [(1, 2), (3, 4), (4, 5), (3, 5)])
        h = relabel(g, [6, 7, 1, 2, 3, 4, 5])
        buf = Pair{Int, Int}[]
        @test isIsomorphic(g, h, buf)
        @test isValidIso(g, h, buf)

        #> Same order and edge count, but different degree sequences:
        #> two disjoint edges (1, 1, 1, 1) vs a 2-path plus an isolated vertex (1, 2, 1, 0)
        @test !isIsomorphic(SimpleGraph(4, [(1, 2), (3, 4)]),
                            SimpleGraph(4, [(1, 2), (2, 3)]))
    end

    @testset "non-Int vertex-label type" begin
        g = SimpleGraph(5, NTuple{2, Int16}[(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        h = relabel(g, [3, 4, 5, 1, 2])
        @test h isa SimpleGraph{Int16}
        buf = Pair{Int16, Int16}[]
        @test isIsomorphic(g, h, buf)
        @test isValidIso(g, h, buf)
    end

    @testset "match!Self: never emptied, appended on success" begin
        g = cycle6
        h = relabel(g, randperm(MersenneTwister(99), countVertices(g)))

        pre = [100 => 200, 300 => 400] #> Pre-existing data the caller wants kept
        buf = copy(pre)
        @test isIsomorphic(g, h, buf)
        #> Prefix preserved exactly
        @test buf[1:length(pre)] == pre
        #> Exactly countVertices(g) pairs appended, and they form a valid iso
        appended = buf[(length(pre)+1):end]
        @test length(appended) == countVertices(g)
        @test isValidIso(g, h, appended)
    end

    @testset "match!Self: restored to entry state on failure" begin
        pre = [11 => 22, 33 => 44]

        #> Early-exit failure (different orders): buffer untouched
        buf = copy(pre)
        @test !isIsomorphic(SimpleGraph(3, Int), SimpleGraph(4, Int), buf)
        @test buf == pre

        #> Early-exit failure (different degree sequences): buffer untouched
        buf = copy(pre)
        @test !isIsomorphic(path4, star3, buf)
        @test buf == pre

        #> Deep-search failure (C6 vs 2 × C3): backtracking must roll the buffer all the
        #> way back to its entry state
        buf = copy(pre)
        @test !isIsomorphic(cycle6, twoTri, buf)
        @test buf == pre
    end

    @testset "randomized differential test vs brute force (n ≤ 8)" begin
        rng = MersenneTwister(1234)
        for _ in 1:400
            n = rand(rng, 2:8)
            p = rand(rng, (0.3, 0.5, 0.7))
            g1 = randGraph(rng, n, p)

            #> Half the time test a guaranteed-isomorphic relabeling; half the time a
            #> fresh random graph (usually non-isomorphic, but the oracle decides)
            g2 = rand(rng, Bool) ? relabel(g1, randperm(rng, n)) : randGraph(rng, n, p)

            buf = Pair{Int, Int}[]
            got = isIsomorphic(g1, g2, buf)
            @test got == bruteIso(g1, g2)
            #> When it reports true, the recorded mapping must be a real isomorphism
            got && @test isValidIso(g1, g2, buf)
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# 8. genRootGraph
# ──────────────────────────────────────────────────────────────────────────────
@testset "genRootGraph" begin
    @testset "special cases" begin
        @testset "order 0" begin
            ok, root = genRootGraph(SimpleGraph(0))
            @test ok
            @test countVertices(root) == 0
            @test countEdges(root) == 0
        end

        @testset "order 1 (K1): root is K2" begin
            ok, root = genRootGraph(SimpleGraph(1))
            @test ok
            @test countVertices(root) == 2
            @test listEdges(root) == [(1, 2)]
        end

        @testset "disconnected input throws; decompose-then-recover works" begin
            @test_throws ArgumentError genRootGraph(SimpleGraph(4, [(1, 2), (3, 4)]))

            #> K1 ∪ K2 as a whole is the line graph L(K2 ∪ K2), but `genRootGraph`
            #> requires connected input: decompose first, then recover each component
            g = SimpleGraph(3, [(2, 3)])
            @test_throws ArgumentError genRootGraph(g)
            _, subgraphs = decompose(g)
            for (sg, rootEdges) in zip(subgraphs, ([(1, 2)], [(1, 2), (2, 3)]))
                ok, root = genRootGraph(sg)
                @test ok
                @test listEdges(root) == rootEdges
            end
        end

        @testset "checkConnectivity=false skips the check on connected input" begin
            ok, _ = genRootGraph(makeKn(3), false)
            @test ok
        end
    end

    @testset "line graphs (returns true => root)" begin
        @testset "K2 = L(P3)" begin
            g = makeKn(2)
            ok, root = genRootGraph(g)
            @test ok
            @test isValidRoot(g, root)
            @test countVertices(root) == 3
            @test countEdges(root) == 2
        end

        @testset "P3 = L(P4)" begin
            g = makePath(3)
            ok, root = genRootGraph(g)
            @test ok
            @test isValidRoot(g, root)
            @test countVertices(root) == 4
            @test countEdges(root) == 3
        end

        @testset "K3 = L(K3) or L(K_{1,3})" begin
            #> K3 is the only connected graph with two non-isomorphic roots
            g = makeKn(3)
            ok, root = genRootGraph(g)
            @test ok
            @test isValidRoot(g, root)
        end

        @testset "C4 = L(C4): root is C4" begin
            g = makeCn(4)
            ok, root = genRootGraph(g)
            @test ok
            @test isValidRoot(g, root)
            @test isIsomorphic(root, makeCn(4))
        end

        @testset "Kn = L(K_{1,n}): root is the star" begin
            for n in (4, 5)
                g = makeKn(n)
                ok, root = genRootGraph(g)
                @test ok
                @test isValidRoot(g, root)
                @test countVertices(root) == n + 1
                @test sortedDegrees(root) == [fill(1, n); n]
            end
        end

        @testset "octahedron = L(K4): root is K4" begin
            g = genLineGraph(makeKn(4))
            ok, root = genRootGraph(g)
            @test ok
            @test isValidRoot(g, root)
            @test isIsomorphic(root, makeKn(4))
        end

        @testset "L(K5) (10 vertices): root is K5" begin
            g = genLineGraph(makeKn(5))
            @test countVertices(g) == 10
            @test countEdges(g) == 30
            ok, root = genRootGraph(g)
            @test ok
            @test isValidRoot(g, root)
            @test isIsomorphic(root, makeKn(5))
        end

        @testset "diamond graph (K4 minus one edge)" begin
            g = SimpleGraph(4, [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
            ok, root = genRootGraph(g)
            @test ok
            @test isValidRoot(g, root)
            @test countVertices(root) == 4
            @test countEdges(root) == 4
        end

        @testset "round trip over assorted line graphs" begin
            #> For every line graph g, L(root(g)) must reproduce g up to isomorphism
            lineGraphs = [
                makePath(2), makePath(4), makePath(5),
                makeCn(5), makeCn(6),
                genLineGraph(makeKmn(2, 3)),
                genLineGraph(makeKmn(3, 3)),
                genLineGraph(SimpleGraph(5, [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])),
            ]
            for g in lineGraphs
                ok, root = genRootGraph(g)
                @test ok
                @test countEdges(root) == countVertices(g)
                @test isValidRoot(g, root)
            end
        end
    end

    @testset "non-line graphs (returns false => g)" begin
        wheel5 = SimpleGraph(6, [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6),    #> spokes
                                 (2, 3), (3, 4), (4, 5), (5, 6), (2, 6)])   #> rim
        nonLineGraphs = (
            makeK1n(3),                                       #> claw K_{1,3}
            makeK1n(4),                                       #> star K_{1,4}
            makeK1n(5),                                       #> star K_{1,5}
            makeKmn(2, 3),                                    #> K_{2,3}
            SimpleGraph(5, [(1, 2), (1, 4), (1, 5), (2, 3)]), #> claw w/ a subdivided leaf
            wheel5,                                           #> wheel W5
        )
        for g in nonLineGraphs
            ok, returned = genRootGraph(g)
            @test !ok
            @test returned === g #> The original graph is returned as the second element
        end
    end

    @testset "small random roots validated by bruteIso" begin
        rng = MersenneTwister(1234)
        for _ in 1:100
            root0 = randGraph(rng, rand(rng, 2:6), rand(rng, (0.2, 0.4, 0.6)))
            countEdges(root0) == 0 && continue
            countEdges(root0) > 8 && continue

            g = genLineGraph(root0)

            # genRootGraph expects connected input
            length(listComponents(g)) == 1 || continue

            ok, root = genRootGraph(g)
            @test ok
            @test bruteIso(genLineGraph(root), g)
        end
    end
end

end
