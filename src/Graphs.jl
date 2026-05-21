export DenseGraph, UnweightedGraph, SimpleGraph


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

function SimpleGraph(order::Integer, edges::AbstractVector{NTuple{2, T}}, 
                     throwError::Bool=false) where {T<:Integer}
    g = SimpleGraph(order)
    for edge in edges
        success = addEdge!(g, edge)
        if throwError && !success
            throw(DomainError(edge, "This is an invalid or repeated edge."))
        end
    end
    g
end


function modEdge!(g::SimpleGraph, edge::NTuple{2, Integer}, connect::Bool)
    i, j = minmax(edge...)

    if i == j
        false
    elseif 1 <= i && j <= g.order
        if connect
            bl = !in!(j, g.adjacency[begin+i-1])
            bl && push!(g.adjacency[begin+j-1], i)
        else
            bl = in(j, g.adjacency[begin+i-1])
            if bl
                pop!(g.adjacency[begin+i-1], j)
                pop!(g.adjacency[begin+j-1], i)
            end
        end

        bl
    else
        false
    end
end

addEdge!(g::SimpleGraph, edge::NTuple{2, Integer}) = modEdge!(g, edge, true)

rmvEdge!(g::SimpleGraph, edge::NTuple{2, Integer}) = modEdge!(g, edge, false)


function hasEdge(g::SimpleGraph, edge::NTuple{2, Integer})
    i, j = sort(edge)

    if i == j
        false
    elseif 1 <= i && j <= g.order
        in(j, g.adjacency[begin+i-1])
    else
        false
    end
end


function listEdge(g::SimpleGraph{T}) where {T<:Integer}
    edges = NTuple{2, T}[]

    for i in 1:g.order
        for j in g.adjacency[begin+i-1]
            if j > i
                push!(edges, (i, j))
            end
        end
    end

    sort!(edges)
end


function countEdge(g::SimpleGraph{T}) where {T<:Integer}
    typeC = typemax(T) > typemax(Int) ? T : Int
    count = zero(typeC)
    for list in g.adjacency; (count += (typeC∘length)(list)) end
    isodd(count) && throw(AssertionError("The adjacency lists of `g` have been corrupted."))
    count ÷ typeC(2)
end


function listDegree(g::SimpleGraph)
    length.(g.adjacency)
end


function shareEndPoint(nodeEdge1::NTuple{2, Integer}, nodeEdge2::NTuple{2, Integer})
    l1, r1 = nodeEdge1
    l2, r2 = nodeEdge2
    (l1 == l2 != 0) || (l1 == r2 != 0) || (l2 == r1 != 0) || (r1 == r2 != 0)
end


function getLineGraph(g::SimpleGraph)
    edges = listEdge(g)
    m = length(edges)
    lg = SimpleGraph(m)

    for i in 1:m, j in (i+1):m
        #> Two edges are adjacent iff they share an endpoint
        if shareEndPoint(edges[i], edges[j])
            addEdge!(lg, (i, j))
        end
    end

    lg
end