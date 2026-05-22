export SimpleGraph, countVertices, attachEdge!, removeEdge!, containEdge, countEdges, 
       listEdges, listDegrees, genLineGraph, listComponents, decompose, genRootGraph

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
                     throwError::Bool=false) where {T<:Integer}
    typeO = typemax(T) > typemax(Int) ? T : Int
    g = SimpleGraph(order|>typeO)
    for edge in edges
        success = attachEdge!(g, edge)
        if throwError && !success
            throw(DomainError(edge, "This is an invalid or repeated edge."))
        end
    end
    g
end


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

attachEdge!(g::SimpleGraph, edge::NTuple{2, Integer}) = modEdge!(g, edge, true)

removeEdge!(g::SimpleGraph, edge::NTuple{2, Integer}) = modEdge!(g, edge, false)


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


function countEdges(g::SimpleGraph{T}) where {T<:Integer}
    typeC = typemax(T) > typemax(Int) ? T : Int
    count = zero(typeC)
    for list in g.adjacency; (count += (typeC∘length)(list)) end
    isodd(count) && throw(AssertionError("The adjacency lists of `g` have been corrupted."))
    count ÷ typeC(2)
end


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


function listDegrees(g::SimpleGraph)
    length.(g.adjacency)
end


function shareEndPoint(nodeEdge1::NTuple{2, Integer}, nodeEdge2::NTuple{2, Integer})
    l1, r1 = nodeEdge1
    l2, r2 = nodeEdge2
    (l1 == l2 != 0) || (l1 == r2 != 0) || (l2 == r1 != 0) || (r1 == r2 != 0)
end


function genLineGraph(g::SimpleGraph)
    edges = listEdges(g)
    m = length(edges)
    lg = SimpleGraph(m)

    for i in 1:m, j in (i+1):m
        #> Two edges are adjacent iff they share an endpoint
        if shareEndPoint(edges[i], edges[j])
            attachEdge!(lg, (i, j))
        end
    end

    lg
end


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