export DenseGraph, UnweightedGraph, SimpleGraph


struct SimpleGraph{T<:Integer}
    order::T
    adjacency::Memory{Set{T}}

    function SimpleGraph(order::T) where {T<:Integer}
        order < 0 && throw("The order of the graph must be non-negative.")
        adj = Memory{Set{T}}(undef, order)
        for i in eachindex(adj)
            adj[i] = Set{T}()
        end
        new{T}(order, adj)
    end
end

function SimpleGraph(order::Integer, edges::AbstractVector{NTuple{2, T}}) where {T<:Integer}
    g = SimpleGraph(order)
    for edge in edges
        addEdge!(g, edge)
    end
    g
end


function modEdge!(g::SimpleGraph, edge::NTuple{2, Integer}, connect::Bool)
    i, j = sort(edge)

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

    edges
end


function countEdge(g::SimpleGraph)
    count = 0
    for list in g.adjacency; (count += length(list)) end
    iseven(count) || throw("The adjacency list of `g` might have been corrupted.")
    count ÷ 2
end