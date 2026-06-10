export SimpleGraph, countVertices, attachEdge!, removeEdge!, containEdge, getDegree, 
       countEdges, listEdges, listDegrees, genLineGraph, listComponents, decompose, 
       isIsomorphic, genRootGraph


"""
    SimpleGraph{T<:Integer}

A simple graph represented by a graph order (`.order::Int <= typemax(Int)`) and adjacency 
sets (`.adjacency`). Vertices are labeled by positive integers (i.e., from `1` to `.order`) 
of type `T`.

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
    c = 0
    for node in 1:countVertices(g); c += getDegree(g, node) end
    isodd(c) && throw(ArgumentError("The adjacency lists of `g` have been corrupted."))
    c ÷ 2
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
        nAdj = count(in(adjList[begin+v-1]), triangle)
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


const MissingOr{T} = Union{Missing, T}


"""

    breadthFirstSearch(f, graph::SimpleGraph{T}, startingPoint::MissingOr{T}, 
                       cache!Self::AbstractVector{T}=zeros(T, graph.order)
                       ) where {T<:Integer} -> 
    Tuple{T, Int}

Perform a breadth-first search (BFS) on `graph`, starting from `startingPoint`. 

If `startingPoint === missing`, the search visits all connected components, iterating 
through vertices in `1:graph.order` as deterministic component roots. If `startingPoint` is
a vertex label, only the connected component reachable from `startingPoint` is
searched.

The predicate `f` is called once for each dequeued vertex as `f(v)::Bool`. If `f(v)` 
returns `true`, the search terminates immediately and returns `(v, nVisited)`, where `v::T` 
is the first vertex satisfying `f`, and `nVisited` is the number of vertices dequeued, 
equivalently the number of calls to `f`. If no vertex satisfies `f`, the function returns 
`(zero(T), nVisited)` is used as the sentinel value for "not found".

The optional `cache!Self` is used as a buffer to store the deterministic BFS queue. It must 
have length at least `graph.order`. The elements of `cache!Self` up to the returned 
`nVisited` position contain the BFS discovery order.
"""
function breadthFirstSearch(f::F, graph::SimpleGraph{T}, startingPoint::MissingOr{T}, 
                            cache!Self::AbstractVector{T}=zeros(T, graph.order)
                            ) where {T<:Integer, F}
    order = graph.order
    register = Memory{Bool}(undef, order)
    register .= false
    nodeRange = if ismissing(startingPoint)
        (1 : order)
    else
        (startingPoint < 1 || startingPoint > order) && 
        throw(DomainError(startingPoint, "`startingPoint` should be between 1 and $order."))
        (startingPoint,)
    end

    if length(cache!Self) < order
        throw(ArgumentError("The length of `cache!Self` must be at least `$order`."))
    end
    tail = 0
    offset = firstindex(cache!Self) - 1

    for localStart in nodeRange #> In case input graph is not connected
        if !register[begin+localStart-1]
            register[begin+localStart-1] = true
            head = (tail += 1)
            cache!Self[offset+tail] = T(localStart)

            while head <= tail
                lastNode = Int(cache!Self[offset+head])

                if f(lastNode)::Bool
                    return (lastNode, head) #> Matched node and call count of `f`
                else
                    tailLast = tail
                    for child in graph.adjacency[begin+lastNode-1]
                        if !register[begin+child-1]
                            register[begin+child-1] = true
                            tail += 1
                            cache!Self[offset+tail] = child
                        end
                    end
                    iStart = offset + tailLast + 1
                    iFinal = offset + tail
                    iStart < iFinal && sort!(@view cache!Self[iStart:iFinal])
                end

                head += 1
            end
        end
    end

    #> Sanity check for node traversal
    if ismissing(startingPoint) && tail != order
        throw(AssertionError("The number of visited vertices is not correct."))
    end

    #> First index fallbacks to zero if no `node` is found such that `f(node)`
    (zero(T), tail)
end

"""

    breadthFirstSearch(graph::SimpleGraph{T}, startingPoint::MissingOr{T}=missing
                       ) where {T<:Integer} -> 
    Vector{T}

Return the deterministic breadth-first traversal order of `graph`. Newly discovered 
children of each visited vertex are sorted before being enqueued into the returned value.

If `startingPoint === missing`, the returned value contains the BFS traversal order over 
all connected components. Components are started in increasing vertex order from 
`1:graph.order`. If `startingPoint` is a vertex label, the returned value contains only the 
BFS order of the connected component reachable from `startingPoint`.
"""
function breadthFirstSearch(graph::SimpleGraph{T}, startingPoint::MissingOr{T}=missing
                            ) where {T<:Integer}
    storage = zeros(T, graph.order)
    _, nVisited = breadthFirstSearch(_->false, graph, startingPoint, storage)
    ismissing(startingPoint) ? storage : storage[begin:begin+nVisited-1]
end


const SameTypePair{T} = Pair{T, T}

mutable struct GraphMapInfo{T<:Integer}
    const graph::SameTypePair{SimpleGraph{T}} #> Compared graph: g1 -> g2
    const track::Memory{Pair{T, T}}           #> Element: prev-matched node => candidate
    const register::Memory{Bool}              #> `.register[begin+g2Cand-1] == isUsed`
    const frontier::SameTypePair{Memory{T}}   #> T1 => T2
    indexer::T                                #> The latest matched node in g1

    function GraphMapInfo(g1::SimpleGraph{T}, g2::SimpleGraph{T}) where {T<:Integer}
        g1Order, g2Order = g1.order, g2.order
        track = Memory{Pair{T, T}}(undef, g1Order); track .= (zero(T) => zero(T))
        g1Front = Memory{T}(undef, g1Order); g1Front .= zero(T)
        g2Front = Memory{T}(undef, g2Order); g2Front .= zero(T)
        register = Memory{Bool}(undef, g2Order); register .= false
        new{T}(g1=>g2, track, register, g1Front=>g2Front, 0)
    end
end


function popMatchPair!(info::GraphMapInfo{T}) where {T<:Integer}
    g1Node = info.indexer
    matchTrack = info.track

    if !iszero(g1Node)
        #> Remove the record of the pair
        lastNode, g2Cand = matchTrack[begin+g1Node-1]
        matchTrack[begin+g1Node-1] = (zero(T) => zero(T))
        info.indexer = lastNode
        candReg = info.register

        #> Revoke registration of matched node-candidate pair
        if candReg[begin+g2Cand-1]
            candReg[begin+g2Cand-1] = false
        else
            throw(AssertionError("The candidate `$(g2Cand)` should have been registered."))
        end

        #> Remove the frontier nodes matched `g1Node` by resetting their stamps to zero
        graph1, graph2 = info.graph
        front1, front2 = info.frontier
        for u in graph1.adjacency[begin+g1Node-1]
            front1[begin+u-1] == g1Node && (front1[begin+u-1] = 0)
        end
        for v in graph2.adjacency[begin+g2Cand-1]
            front2[begin+v-1] == g1Node && (front2[begin+v-1] = 0)
        end
    end

    g1Node #> Fallback to `zero(T)` for an empty track
end


function addMatchPair!(info::GraphMapInfo{T}, pair::SameTypePair{T}) where {T<:Integer}
    g1Node, g2Cand = pair
    candReg = info.register

    iszero(g1Node) && throw(ArgumentError("`pair.first` cannot equal zero."))
    if !iszero(candReg[begin+g2Cand-1])
        throw(ArgumentError("`pair.second` should not haven been registered."))
    end

    g1, g2 = info.graph
    adjs1 = g1.adjacency[begin+g1Node-1]
    adjs2 = g2.adjacency[begin+g2Cand-1]
    front1, front2 = info.frontier

    if length(adjs1) != length(adjs2) #> Premise of using one-direction consistency rules
        throw(ArgumentError("The degree of the candidate should equal that of the source."))
    end

    #> Node-pair feasibility check
    #>> Cutting rules (Need to be run before the consistency rules)
    leftoverCount1 = frontierCount1 = leftoverCount2 = frontierCount2 = 0
    matchTrack = info.track
    for u in adjs1
        v = matchTrack[begin+u-1].second
        if !iszero(v)
            if !candReg[begin+v-1]
                throw(AssertionError("Matched candidate `$v` should have been registered."))
            end
            continue #> Skip already matched node
        end
        if iszero(front1[begin+u-1]) #> `u` is in the leftover set of g1 nodes (T1^)
            leftoverCount1 += 1
        else #> `u` is the frontier of g1 nodes (T1)
            frontierCount1 += 1
        end
    end
    for v in adjs2
        candReg[begin+v-1] && continue #> Skip already matched candidate
        if iszero(front2[begin+v-1]) #> `v` is in the leftover set of g2 nodes (T2^)
            leftoverCount2 += 1
        else #> `v` is the frontier of g2 nodes (T2)
            frontierCount2 += 1
        end
    end
    (leftoverCount1 == leftoverCount2 && frontierCount1 == frontierCount2) || (return false)

    #>> The two nodes must originate from the same set (either frontier or leftover)
    g1InFrontier = !iszero(front1[begin+g1Node-1]) #> `false` -> `g1Node` is from T1^
    g2InFrontier = !iszero(front2[begin+g2Cand-1]) #> `false` -> `g2Cand` is from T2^
    g1InFrontier == g2InFrontier || (return false)

    #> Under the condition: `g1InFrontier == g2InFrontier`
    #>> Consistency rules (which also serve as a candidate selection scheme)
    if g1InFrontier #>> One-direction check is safe as |match-n ∩ adjs1|==|match-c ∩ adjs2|
        for u in adjs1
            v = matchTrack[begin+u-1].second
            iszero(v) || in(v, adjs2) || (return false)
        end
    end

    #> `g1Node` as the stamp for promoted frontier nodes
    for u in adjs1
        if iszero(matchTrack[begin+u-1].second) && iszero(front1[begin+u-1])
            front1[begin+u-1] = g1Node
        end
    end
    for v in adjs2
        if !candReg[begin+v-1] && iszero(front2[begin+v-1])
            front2[begin+v-1] = g1Node
        end
    end

    #> Success registration
    matchTrack[begin+g1Node-1] = (info.indexer => g2Cand)
    candReg[begin+g2Cand-1] = true
    info.indexer = g1Node

    true
end


function connectivityOrder(g::SimpleGraph{T}, 
                           rootOrder::AbstractVector{Int}) where {T<:Integer}
    nv = countVertices(g)
    if length(rootOrder) != nv
        throw(ArgumentError("The length of `rootOrder` does not match the order of `g`."))
    end

    order = Memory{T}(undef, nv)
    buffer = Memory{T}(undef, nv)
    ordered = Memory{Bool}(undef, nv)
    ordered .= false
    slot = 1

    for root in rootOrder
        ordered[begin+root-1] && continue
        _, nVisited = breadthFirstSearch(_->false, g, T(root), buffer)

        for idx in 1:nVisited
            v = buffer[begin+idx-1]
            order[begin+slot-1] = v
            ordered[begin+v-1] = true
            slot += 1
        end
    end

    order
end


"""
    isIsomorphic(g1::SimpleGraph{T}, g2::SimpleGraph{T},
                 match!Self::MissingOr{AbstractVector{SameTypePair{T}}}=missing) where
    {T<:Integer} ->
    Bool

Return whether `g1` is isomorphic to `g2`, i.e., whether there exists a bijection
between their vertices that preserves vertex adjacency.

The underlying algorithm of this function is a non-recursive variant of the VF2++ 
algorithm. Instead of applying the exact vertex ordering subroutine from VF2++, the 
vertices of `g1` are directly processed through BFS where the roots for connected 
components are selected following a non-increasing degree order. Additionally, for a given 
`g1` vertex, only `g2` vertices of the same degree are promoted as matching candidates.

When the optional argument `match!Self` is set to `missing` (the default), no 
vertex-to-vertex mapping is recorded. Otherwise, `match!Self` is treated as a mutable 
buffer of `(g1Vertex => g2Vertex)::Pair{T, T}` pairs. By design it is **never emptied on 
entry**, so that any data already in it is preserved after the function call. In other 
words, on success, the discovered mappings are directly *appended* to the buffer. On 
failure, the buffer is restored to exactly its initial state.
"""
function isIsomorphic(g1::SimpleGraph{T}, g2::SimpleGraph{T}, 
                      match!Self::MissingOr{AbstractVector{ SameTypePair{T} }}=missing
                      ) where {T<:Integer}
    storeMatch = ismissing(match!Self) ? false : true

    nv = countVertices(g1)
    nv == countVertices(g2) || (return false)

    ne = countEdges(g1)
    if ne == countEdges(g2)
        if ne == 0
            storeMatch && (for n in 1:nv; push!(match!Self, T(n)=>T(n)) end)
            return true
        end
    else
        (return false)
    end

    g1Degrees = listDegrees(g1)
    g2Degrees = listDegrees(g2)

    #> Organize the nodes by their degrees in non-increasing order
    g1NodesByDegree = sortperm(g1Degrees; rev=true)
    g2NodesByDegree = sortperm(g2Degrees; rev=true)

    #> Ensure nodes are labeled by one-based integers
    g1NodesByDegree .+= 1 - firstindex(g1Degrees)
    g2NodesByDegree .+= 1 - firstindex(g2Degrees)

    #> Verify the degree matching and record the info of each degree block
    blocks = Dict{Int, Pair{Int, Int}}() #>> degree => (blockStartIdx => blockSpace)
    for i in 1:nv
        d1 = g1Degrees[begin+g1NodesByDegree[begin+i-1]-1]
        d2 = g2Degrees[begin+g2NodesByDegree[begin+i-1]-1]
        d1 == d2 || (return false) #> Mismatched degree sequence
        haskey(blocks, d2) || (blocks[d2] = i => count(==(d2), g2Degrees))
    end

    #> Process g1 by connectivity (degrees interleaved); map each depth to its degree block
    g1NodeOrder = connectivityOrder(g1, g1NodesByDegree)
    globalCandScopes = [blocks[g1Degrees[begin+g1NodeOrder[begin+d-1]-1]] for d in 1:nv]

    depth = 1
    info = GraphMapInfo(g1, g2)
    searchOffsets = Memory{Int}(undef, nv)
    searchOffsets .= 0
    foundMatching = false
    localCandScopes = Dict{Pair{T, Int}, Vector{T}}()

    #> `nodeDepths[d]`: Depth (one-based position in `g1NodeOrder`) of node labeled by `d`
    nodeDepths = Memory{Int}(undef, nv)
    for d in 1:nv; nodeDepths[begin+g1NodeOrder[begin+d-1]-1] = d end
    #> `prevNodes[d]`: An already matched node neighboring the `d`-th node in `g1NodeOrder`
    prevNodes = Memory{Int}(undef, nv)
    for (d, n) in enumerate(g1NodeOrder)
        nodeStat = (typemax(Int), typemax(Int)) #> (degree, depth) of the potential neighbor
        bestNode = zero(T) #> Fall back to zero if no best neighbor found
        for node in g1.adjacency[begin+n-1]
            nodePos = nodeDepths[begin+node-1]
            nodePos < d || continue #> Restrict to already-matched (shallower) neighbors
            stat = (g1Degrees[begin+node-1], nodePos)
            if stat < nodeStat
                nodeStat = stat
                bestNode = node
            end
        end
        prevNodes[begin+d-1] = bestNode
    end

    while depth >= 1
        if depth > nv
            foundMatching = true
            break
        end

        node = g1NodeOrder[begin+depth-1]
        offset = searchOffsets[begin+depth-1]
        prevNode = prevNodes[begin+depth-1]
        descend = false

        candList = if iszero(prevNode) #> `node` has no optimal previous neighbor
            iStart, space = globalCandScopes[begin+depth-1]
            @view g2NodesByDegree[(begin+iStart-1):(begin+iStart+space-2)]
        else
            deg = g1Degrees[begin+node-1]
            img = info.track[begin+prevNode-1].second #> Previously matched candidate
            iszero(img) && throw(AssertionError("The matched candidate must not be zero."))
            get!(localCandScopes, img=>deg) do
                sort!([v for v in g2.adjacency[begin+img-1] if g2Degrees[begin+v-1]==deg])
            end
        end

        while offset < length(candList)
            cand = candList[begin+offset]
            offset += 1

            if !info.register[begin+cand-1] #> Ensure the candidate has not been used
                newPair = T(node) => T(cand)
                if addMatchPair!(info, newPair) #> Feasibility check by `addMatchPair!`
                    storeMatch && push!(match!Self, newPair)
                    descend = true
                    #> Memoize the resumed starting point if later ascend to this branch
                    searchOffsets[begin+depth-1] = offset
                    break
                end
            end
        end

        if descend
            depth += 1
        else
            evictedNode = popMatchPair!(info)
            storeMatch && !iszero(evictedNode) && pop!(match!Self)
            searchOffsets[begin+depth-1] = 0 #> Reset branch starting offset to be 0
            depth -= 1
        end
    end

    foundMatching
end