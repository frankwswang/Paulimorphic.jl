export SimpleGraph, countVertices, attachEdge!, removeEdge!, containEdge, getDegree, 
       countEdges, listEdges, listDegrees, genLineGraph, listComponents, decompose, 
       isIsomorphic, genRootGraph


"""
    SimpleGraph{T<:Integer}

A simple graph represented by a graph order (`.order::Int <= typemax(Int)`) and adjacency 
sets (`.adjacency`). Vertices are labeled by positive integers of type `T`.

≡≡≡ Initialization Method(s) ≡≡≡

    SimpleGraph(order::Integer, ::Type{T}=typeof(order)) where {T<:Integer} -> 
    SimpleGraph{T}

Construct a simple graph of `order` with no edges.

    SimpleGraph(order::Integer, edges::AbstractVector{NTuple{2, T}}, 
                explicitError::Bool=false) where {T<:Integer} -> 
    SimpleGraph

Construct a simple graph of `order` with valid (undirected) edge elements from `edges`. All 
invalid (e.g., self-loop, out-of-bound edges) or duplicate elements in `edges` are silently 
ignored unless `explicitError=true`, in which case a `DomainError` is thrown.
"""
struct SimpleGraph{T<:Integer}
    order::Int
    adjacency::Memory{Set{T}}

    function SimpleGraph(order::Integer, ::Type{T}=typeof(order)) where {T<:Integer}
        order < 0 && throw(DomainError(order, "`order` of the graph must be non-negative."))
        order = Int(order)
        adj = Memory{Set{T}}(undef, order)
        for i in eachindex(adj)
            adj[i] = Set{T}()
        end
        new{T}(order, adj)
    end
end

function SimpleGraph(order::Integer, edges::AbstractVector{NTuple{2, T}}, 
                     explicitError::Bool=false) where {T<:Integer}
    g = SimpleGraph(order, T)
    for edge in edges
        success = attachEdge!(g, edge)
        if explicitError && !success
            throw(DomainError(edge, "This is an invalid or duplicate edge."))
        end
    end
    g
end


"""
    countVertices(g::SimpleGraph) -> Int

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

    getDegree(g::SimpleGraph, vertex::Integer) -> Int

Return the degree (i.e., number of neighbors) of the input `vertex` in `g`.
"""
function getDegree(g::SimpleGraph, vertex::Integer)
    vertex < 1 && throw(DomainError(vertex, "`vertex` must be positive integer`"))
    length(g.adjacency[begin+vertex-1])
end


"""
    countEdges(g::SimpleGraph) -> Int

Return the number of undirected edges in `g`. Throws an `ArgumentError` if the adjacency 
representation is internally inconsistent.
"""
function countEdges(g::SimpleGraph)
    count = 0
    for node in 1:countVertices(g); count += getDegree(g, node) end
    isodd(count) && throw(ArgumentError("The adjacency lists of `g` have been corrupted."))
    count ÷ 2
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

    for start in 1:n
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
by `1:length(components[k])` respectively.
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


function isConnected(g::SimpleGraph{T}) where {T<:Integer}
    countVertices(g) == 0 && (return true)

    seen = falses(countVertices(g))
    stack = [one(T)]
    seen[begin] = true

    while !isempty(stack)
        v = pop!(stack)
        for w in g.adjacency[begin+v-1]
            if !seen[begin+w-1]
                seen[begin+w-1] = true
                push!(stack, w)
            end
        end
    end

    all(seen)
end


#> Function and types for line graph analysis
mutable struct NodeEdgeInfo{T<:Integer}
    const edgeLabel::Memory{NTuple{2, T}} #> The (root-graph) edge labels for nodes
    const backtrack::Memory{T}            #> Adjacent (witness) nodes in discovered cliques
    rootOrder::T                          #> Current maximal index for potential root nodes
    halfMarker::Set{T}

    function NodeEdgeInfo(order::T) where {T<:Integer}
        edgeLabel = Memory{NTuple{2, T}}(undef, order)
        backtrack = Memory{T}(undef, order)
        edgeLabel .= Ref(( T(0), T(0) ))
        backtrack .= T(0)

        new{T}(edgeLabel, backtrack, T(0), Set{T}())
    end
end


#> Reference(s): 
## [DOI] 10.1145/321850.321853
"""
    genRootGraph(g::SimpleGraph, checkConnectivity::Bool=true) -> Pair{Bool, SimpleGraph}

Attempt to recognize connected `g` as a line graph using Lehot-style edge-label
algorithm to reconstruct potentially corresponding root graph `r`. Return `true => r` if 
`g` is indeed a (connected) line graph; return `false => g` otherwise. This function is 
only well behaved when the input `g` is a connected `SimpleGraph`. Hence, in default, 
`checkConnectivity=true` such that any input that is a disconnected graph throws an 
`ArgumentError`. For disconnected graphs, one can first apply [`decompose`](@ref) to 
obtain connected subgraphs, and then apply `genRootGraph` to each subgraph.
"""
function genRootGraph(g::SimpleGraph{T}, checkConnectivity::Bool=true) where {T<:Integer}
    if checkConnectivity && !isConnected(g)
        throw(ArgumentError("The input graph `g` must be connected."))
    end

    order = countVertices(g)
    adjList = g.adjacency

    if order == 0
        return (true => SimpleGraph(0))
    elseif order == 1
        rootGraph = SimpleGraph(2)
        attachEdge!(rootGraph, (1, 2))
        return (true => rootGraph)
    end

    #> Declare two valid vertices to be the basic nodes
    node12 = 0
    node12NeighborNum = 0
    for (i, list) in enumerate(adjList)
        neighborNum = length(list)

        if i == 1 || node12NeighborNum > neighborNum
            node12 = i
            node12NeighborNum = neighborNum
        end
    end
    node23 = first(adjList[begin+node12-1])

    info = NodeEdgeInfo(order)
    edgeLabels = info.edgeLabel
    edgeLabels[begin+node12-1] = (1, 2)
    edgeLabels[begin+node23-1] = (2, 3)

    #> Construct the shared adjacency list w.r.t. basic nodes (1, 2) and (2, 3)
    sharedAdjs = [n for n in adjList[begin+node12-1] if n in adjList[begin+node23-1]]
    nSharedAdj = length(sharedAdjs)

    #> Mark the nodes (2, ...) in clique-2 and potentially a cross node (1, 3)
    if nSharedAdj == 1
        x = first(sharedAdjs) #> `x` is either (1, 3) or (2, 4)
        xEdge = isOddTriangle((node12, node23, x), g) ? (2, 4) : (1, 3)
        edgeLabels[begin+x-1] = xEdge
        info.rootOrder = last(xEdge)
    elseif nSharedAdj == 2
        x = first(sharedAdjs)
        y =  last(sharedAdjs)

        xEdge, yEdge, newOrder = if y in adjList[begin+x-1]  #> `x` and `y` are adjacent
            (2, 4), (2, 5), 5
        else
            edgePair = if isOddTriangle((node12, node23, x), g)
                (2, 4), (1, 3)
            else #> Including when both (1, 2)-(2, 3)-`x` and (1, 2)-(2, 3)-`y` are even
                (1, 3), (2, 4)
            end

            first(edgePair), last(edgePair), 4
        end

        edgeLabels[begin+x-1] = xEdge
        edgeLabels[begin+y-1] = yEdge
        info.rootOrder = newOrder
    elseif nSharedAdj >= 3
        b = last(sharedAdjs) #> The (one-based indexed) location of `b` is at least 3
        bAdjs = adjList[begin+b-1]
        crossNodeNum = 0
        maxRootOrder = 3
        bAsCrossNode = false

        for (i, node) in enumerate(sharedAdjs)
            asCrossNode = if node in bAdjs #> Assuming `b` is not in `bAdjs`
                false #> If landed on i=1, `b` is assumed to not be a cross node either
            elseif i == 1 #> When `a` is the first to be investigated in the neighbors
                tieBreaker = sharedAdjs[begin+1]
                bAsCrossNode = (node in adjList[begin+tieBreaker-1])
                !bAsCrossNode
            else
                node == b ? bAsCrossNode : !bAsCrossNode
            end

            asCrossNode && (crossNodeNum += 1)
            crossNodeNum > 1 && (return (false => g)) #> At most one cross node allowed
            edgeLabels[begin+node-1] = asCrossNode ? (1, 3) : (2, (maxRootOrder += 1))
        end

        info.rootOrder = maxRootOrder
    else #> No shared adjacent nodes so the only indexed nodes are 1, 2, 3
        info.rootOrder = 3
    end

    discoveredNodes = push!(sharedAdjs, node12, node23) #> Built upon `sharedAdjs`
    halfName!(info, adjList, discoveredNodes, 2) #> Half name all nodes adjacent to clique-2
    fullyName!(info, adjList)

    rootGraph = SimpleGraph(info.rootOrder)

    for edge in edgeLabels
        (isFullyNamed(edge) && attachEdge!(rootGraph, edge)) || (return (false => g))
    end

    gEdges = listEdges(g)

    for (nodeL, nodeR) in gEdges
        edgeL = edgeLabels[begin+nodeL-1]
        edgeR = edgeLabels[begin+nodeR-1]
        shareEndPoint(edgeL, edgeR) || (return (false => g))
    end

    #> In case the line graph is a supergraph of `g`
    lRootEdgeNum = 0
    for adjs in rootGraph.adjacency
        d = length(adjs)
        lRootEdgeNum += d * (d - 1) ÷ 2
    end
    length(gEdges) == lRootEdgeNum ? (true => rootGraph) : (false => g)
end


function isOddTriangle(triangle::NTuple{3, Integer}, g::SimpleGraph)
    order = countVertices(g)
    for i in triangle
        (i < 1 || i > order) && throw(DomainError(i, "The vertex label is out of bounds."))
    end
    adjList = g.adjacency

    for v in 1:countVertices(g)
        v in triangle && continue
        nAdj = count(t -> t in adjList[begin+v-1], triangle)
        isodd(nAdj) && return true
    end

    false
end


function isFullyNamed(nodeEdge::NTuple{2, Integer})
    first(nodeEdge) != 0 && last(nodeEdge) != 0
end

function isHalfNamed(nodeEdge::NTuple{2, Integer})
    first(nodeEdge) != 0 && last(nodeEdge) == 0
end

function isUnNamed(nodeEdge::NTuple{2, Integer})
    nodeEdge == (0, 0)
end


function halfName!(info::NodeEdgeInfo, adjList, discoveredNodes, cliqueLabel)
    edgeLabels = info.edgeLabel
    backtracks = info.backtrack
    halfNamedNodes = info.halfMarker

    for node in discoveredNodes
        nodeEdge = edgeLabels[begin+node-1]
        if !isFullyNamed(nodeEdge)
            throw(ArgumentError("Node $node in `discoveredNodes` is not fully named."))
        end

        if cliqueLabel in nodeEdge #> Filter out cross nodes and only keep clique nodes
            rootGraphNodeL, rootGraphNodeR = nodeEdge
            otherEnd = (cliqueLabel == rootGraphNodeL) ? rootGraphNodeR : rootGraphNodeL
            for adjNode in adjList[begin+node-1]
                adjNodeEdge = edgeLabels[begin+adjNode-1]
                if isUnNamed(adjNodeEdge) #> Initialize a half-named node
                    edgeLabels[begin+adjNode-1] = (otherEnd, 0)
                    backtracks[begin+adjNode-1] = node
                    push!(halfNamedNodes, adjNode)
                elseif isHalfNamed(adjNodeEdge) #> Fully name an already half-named node
                    namedEnd = first(adjNodeEdge)
                    if namedEnd != otherEnd
                        fullEdgeName = minmax(otherEnd, namedEnd)
                        edgeLabels[begin+adjNode-1] = fullEdgeName
                        pop!(info.halfMarker, adjNode)
                    end #> It's necessary to silently ignore `namedEnd == otherEnd` case
                end
            end
        end
    end
end


discoverNode!(info, cliqueLabel) = minmax(cliqueLabel, (info.rootOrder += 1))

function fullyName!(info::NodeEdgeInfo{T}, adjList) where {T<:Integer}
    edgeLabels = info.edgeLabel
    backtracks = info.backtrack
    halfNamedNodes = info.halfMarker

    while !isempty(halfNamedNodes)
        node = pop!(halfNamedNodes)
        edge = edgeLabels[begin+node-1]
        bktk = backtracks[begin+node-1]

        #> `node` and `bktk` form the couple of basic nodes for a new clique
        if bktk == 0 || !shareEndPoint(edge, edgeLabels[begin+bktk-1])
            throw(AssertionError("A half-named node should have a valid adjacent node " * 
                                 "that already belongs to a fully discovered clique."))
        end

        cliqueLabel = first(edge)
        edgeLabels[begin+node-1] = discoverNode!(info, cliqueLabel)
        sharedAdjs = [n for n in adjList[begin+node-1] if n in adjList[begin+bktk-1]]

        for cliqueNode in sharedAdjs
            cliqueNodeEdge = edgeLabels[begin+cliqueNode-1]
            if isUnNamed(cliqueNodeEdge)
                edgeLabels[begin+cliqueNode-1] = discoverNode!(info, cliqueLabel)
            elseif isHalfNamed(cliqueNodeEdge)
                namedEnd = first(cliqueNodeEdge)
                pop!(info.halfMarker, cliqueNode)

                edgeLabels[begin+cliqueNode-1] = if namedEnd == cliqueLabel
                    discoverNode!(info, cliqueLabel)
                else
                    minmax(cliqueLabel, namedEnd)
                end
            end
        end

        discoveredNodes = push!(sharedAdjs, bktk, node)
        halfName!(info, adjList, discoveredNodes, cliqueLabel)
    end
end