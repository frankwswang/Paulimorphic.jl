export SimpleGraph, countVertices, attachEdge!, removeEdge!, containEdge, countEdges, 
       listEdges, listDegrees, genLineGraph, listComponents, decompose, genRootGraph


"""
    SimpleGraph{T<:Integer}

A simple graph represented by a graph order (`.order`) and adjacency sets (`.adjacency`).
Vertices are labeled by positive integers of type `T`.

≡≡≡ Initialization Method(s) ≡≡≡

    SimpleGraph(order::T) where {T<:Integer} -> SimpleGraph{T}

Construct a simple graph of `order` with no edges.

    SimpleGraph(order::Integer, edges::AbstractVector{<:NTuple{2, Integer}}, 
                explicitError::Bool=false) -> 
    SimpleGraph

Construct a simple graph of `order` with valid (undirected) edge elements from `edges`. All 
invalid (e.g., self-loop, out-of-bound edges) or duplicate elements in `edges` are silently 
ignored unless `explicitError=true`, in which case a `DomainError` is thrown.
"""
struct SimpleGraph{T<:Integer}
    order::T
    adjacency::Memory{Set{T}}

    function SimpleGraph(order::T) where {T<:Integer}
        order < 0 && throw(DomainError(order, "`order` of the graph must be non-negative."))
        adj = Memory{Set{T}}(undef, order)
        for i in eachindex(adj)
            adj[i] = Set{T}()
        end
        new{T}(order, adj)
    end
end

function SimpleGraph(order::T, edges::AbstractVector{<:NTuple{2, Integer}}, 
                     explicitError::Bool=false) where {T<:Integer}
    typeO = typemax(T) > typemax(Int) ? T : Int
    g = SimpleGraph(order|>typeO)
    for edge in edges
        success = attachEdge!(g, edge)
        if explicitError && !success
            throw(DomainError(edge, "This is an invalid or duplicate edge."))
        end
    end
    g
end


"""
    countVertices(g::SimpleGraph{T}) where {T<:Integer} -> T

Return the number of vertices (i.e., the order) of `g`.
"""
countVertices(g::SimpleGraph) = g.order


function modEdge!(g::SimpleGraph, edge::NTuple{2, Integer}, connect::Bool)
    i, j = minmax(edge...)

    if i == j
        false
    elseif 1 <= i && j <= countVertices(g)
        if connect
            success = !in!(j, g.adjacency[begin+i-1])
            success && push!(g.adjacency[begin+j-1], i)
        else
            res = pop!(g.adjacency[begin+i-1], j, nothing)
            success = (res === nothing) ? false : (pop!(g.adjacency[begin+j-1], i); true)
        end

        success
    else
        false
    end
end

"""
    attachEdge!(g::SimpleGraph, edge::NTuple{2, Integer}) -> Bool

Attach the undirected `edge=(i, j)` to `g`. Return `true` if `edge` was newly attached, 
and `false` if `edge` is a self-loop (i.e., `i == j`), out of bounds, or already present.
"""
attachEdge!(g::SimpleGraph, edge::NTuple{2, Integer}) = modEdge!(g, edge, true)

"""
    removeEdge!(g::SimpleGraph, edge::NTuple{2, Integer}) -> Bool

Remove the undirected `edge=(i, j)` from `g`. Return `true` if `edge`, as an existed edge, 
was successfully removed, and `false` if `edge` was absent or invalid.
"""
removeEdge!(g::SimpleGraph, edge::NTuple{2, Integer}) = modEdge!(g, edge, false)


"""
    containEdge(g::SimpleGraph, edge::NTuple{2, Integer}) -> Bool

Return whether `edge` is a valid non-loop edge present in `g`.
"""
function containEdge(g::SimpleGraph, (m, n)::NTuple{2, Integer})
    i, j = minmax(m, n)

    if i == j
        false
    elseif 1 <= i && j <= countVertices(g)
        in(j, g.adjacency[begin+i-1])
    else
        false
    end
end


"""
    countEdges(g::SimpleGraph) -> Integer

Return the number of undirected edges in `g`. Throws an `AssertionError` if the
adjacency representation is internally inconsistent.
"""
function countEdges(g::SimpleGraph{T}) where {T<:Integer}
    typeC = typemax(T) > typemax(Int) ? T : Int
    count = zero(typeC)
    for list in g.adjacency; (count += (typeC∘length)(list)) end
    isodd(count) && throw(AssertionError("The adjacency lists of `g` have been corrupted."))
    count ÷ typeC(2)
end


"""
    listEdges(g::SimpleGraph{T}) -> Vector{NTuple{2, T}}

Return all undirected edges of `g` as sorted endpoint pairs `(i, j)` in a lexicographically 
ordered `Vector`.
"""
function listEdges(g::SimpleGraph{T}) where {T<:Integer}
    edges = NTuple{2, T}[]

    for i in 1:countVertices(g)
        for j in g.adjacency[begin+i-1]
            if j > i
                push!(edges, (i, j))
            end
        end
    end

    sort!(edges)
end


"""
    listDegrees(g::SimpleGraph) -> Vector{Int}

Return a `Vector` whose `i`-th entry is the degree of vertex `i`.
"""
function listDegrees(g::SimpleGraph)
    length.(g.adjacency)
end


function shareEndPoint(nodeEdge1::NTuple{2, Integer}, nodeEdge2::NTuple{2, Integer})
    l1, r1 = nodeEdge1
    l2, r2 = nodeEdge2
    (l1 == l2 != 0) || (l1 == r2 != 0) || (l2 == r1 != 0) || (r1 == r2 != 0)
end


"""
    genLineGraph(g::SimpleGraph) -> SimpleGraph

Return the line graph of `g`: each edge of `g` becomes a vertex in the returned graph, and 
every two such vertices are adjacent iff the corresponding edges of `g` share an endpoint. 
As a result, Vertex `i` of the constructed graph corresponds to `listEdges(g)[i]`.
"""
function genLineGraph(g::SimpleGraph{T}) where {T<:Integer}
    typeO = typemax(T) > typemax(Int) ? T : Int
    edges = listEdges(g)
    m = (typeO∘length)(edges)
    lg = SimpleGraph(m)

    for i in 1:m, j in (i+1):m
        #> Two edges are adjacent iff they share an endpoint
        if shareEndPoint(edges[i], edges[j])
            attachEdge!(lg, (i, j))
        end
    end

    lg
end


"""
    listComponents(g::SimpleGraph{T}) where {T} -> Vector{Vector{T}}

Return the connected components of `g` as a `Vector` of sorted vertices. Components
are listed in increasing order based on their first listed vertex.
"""
function listComponents(g::SimpleGraph{T}) where {T<:Integer}
    n = countVertices(g)
    seen = falses(n)
    components = Vector{T}[]

    for start in one(T):n
        seen[begin+start-1] && continue

        component = T[]
        stack = T[start]
        seen[begin+start-1] = true

        while !isempty(stack)
            v = pop!(stack)
            push!(component, v)

            for w in g.adjacency[begin+v-1]
                if !seen[begin+w-1]
                    seen[begin+w-1] = true
                    push!(stack, w)
                end
            end
        end

        push!(components, sort!(component))
    end

    components
end


"""
    decompose(g::SimpleGraph{T}) where {T<:Integer} -> 
    Pair{Vector{Vector{T}}, Vector{SimpleGraph{T}}}

Return `components => subgraphs`, where `components` is the output of `listComponents(g)` 
and `subgraphs[k]` is the induced subgraph on `components[k]`, with its vertices relabelled 
by `1:k` respectively.
"""
function decompose(g::SimpleGraph{T}) where {T<:Integer}
    components = listComponents(g)
    subgraphs = SimpleGraph{T}[]
    newLabels = Memory{T}(undef, countVertices(g))
    newLabels .= zero(T)

    for nodes in components
        for (i, node) in enumerate(nodes)
            newLabels[begin+node-1] = i
        end

        sgOrder = (T∘length)(nodes)
        sg = SimpleGraph(sgOrder)

        for node in nodes, adj in g.adjacency[begin+node-1]
            if adj > node
                edge = minmax(newLabels[begin+node-1], newLabels[begin+adj-1])
                if !(0 < first(edge) < last(edge) <= sgOrder)
                    throw(AssertionError("The component decomposition is inconsistent."))
                end
                attachEdge!(sg, edge)
            end
        end

        push!(subgraphs, sg)
    end

    components => subgraphs
end