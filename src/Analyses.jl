export getFrustrationInfo

"""
    getFrustrationInfo(ham::PauliSum; 
                       edgeThreshold::Real=0, nodeThreshold::Real=edgeThreshold) -> Pair

Compute the (anticommutation) frustration graph of the Pauli strings in `ham`: each 
retained term forms a node, and two nodes are joined by an edge whenever their 
corresponding Pauli strings anticommute (as determined by [`checkAntiCom`](@ref)). Such a 
graph records which terms of a Hamiltonian fail to commute — e.g. as a basis for 
partitioning `ham` into mutually commuting groups.

The retained nodes are the terms in `ham` whose coefficient magnitude exceeds 
`nodeThreshold`, taken in by the order in which `ham` stores its terms 
(see [`PauliSum`](@ref)). A pair of retained nodes `(i, j)` with `i < j` is joined by an 
edge only when the two strings anticommute *and* the magnitude of the product of their 
coefficients exceeds `edgeThreshold`; the coefficient weighting lets negligible terms and 
negligible couplings be pruned in a single pass. Both thresholds are compared strictly 
(`>`), and `nodeThreshold` defaults to `edgeThreshold`, so passing only `edgeThreshold` 
applies the same cutoff to both nodes and edges.

The graph is returned not as a [`SimpleGraph`](@ref) but as its structural information: 

    (nodes => edges)::Pair

where `nodes::Vector{PauliStr}` are the retained node strings and 
`edges::Vector{NTuple{2, Int}}` lists the anticommuting pairs as one-based index pairs
`(i, j)`, `i < j`, indexing into `nodes` (instead of the original terms of `ham`). If no 
node survives `nodeThreshold`, both `nodes` and `edges` are empty.

# Keyword Arguments
- `edgeThreshold::Real=0`: an edge `(i, j)` is kept only if
  `abs(coeff_i * coeff_j) > edgeThreshold`.
- `nodeThreshold::Real=edgeThreshold`: a term is kept as a node only if
  `abs(coeff) > nodeThreshold`.
"""
function getFrustrationInfo(ham::PauliSum; 
                            edgeThreshold::Real=0, nodeThreshold::Real=edgeThreshold)
    strs = ham.str
    coeffs = ham.coeff

    validNodes = PauliStr[]
    nodeCoeffs = eltype(coeffs)[]
    for (coeff, str) in zip(coeffs, strs)
        if abs(coeff) > nodeThreshold
            push!(validNodes, str)
            push!(nodeCoeffs, coeff)
        end
    end

    nodeNum = length(validNodes)
    validEdges = NTuple{2, Int}[]

    for i in 1:(nodeNum-1), j in (i+1):nodeNum
        weight = abs(nodeCoeffs[begin+i-1] * nodeCoeffs[begin+j-1])
        weight > edgeThreshold && getFrustrationInfoCore!(validEdges, validNodes, (i, j))
    end

    validNodes => validEdges
end

"""
    getFrustrationInfo(strings::AbstractVector{PauliStr}) -> Pair

Compute the (anticommutation) frustration graph of `strings`, treating every element as a
node with no coefficient weighting or thresholding. Two nodes are joined by an edge 
whenever their Pauli strings anticommute (via [`checkAntiCom`](@ref)).

Same as `getFrustrationInfo(::PauliSum)`, this method returns as the underlying information 
of the frustration graph as a `Pair` 
    
    `strings => edges`

where `edges::Vector{NTuple{2, Int}}` lists the anticommuting pairs as one-based index 
pairs `(i, j)`, `i < j`, indexing into `strings` in the given order.
"""
function getFrustrationInfo(strings::AbstractVector{PauliStr})
    nodeNum = length(strings)

    validEdges = NTuple{2, Int}[]
    for i in 1:(nodeNum-1), j in i+1:nodeNum
        getFrustrationInfoCore!(validEdges, strings, (i, j))
    end

    strings => validEdges
end

"""
    getFrustrationInfoCore!(validEdges, strings, edge::NTuple{2, Int}) -> Pair

Core function for [`getFrustrationInfo`](@ref). It tests the single candidate 
`edge = (i, j)` (one-based indices into `strings`) and pushes `(i, j)` onto `validEdges` if 
`strings[i]` and `strings[j]` anticommute. It returns `strings => validEdges`.
"""
function getFrustrationInfoCore!(validEdges::AbstractVector{NTuple{2, Int}}, 
                                 strings::AbstractVector{PauliStr}, 
                                 edge::NTuple{2, Int}) #> One-based index
    i, j = edge
    checkAntiCom(strings[begin+i-1], strings[begin+j-1]) && push!(validEdges, (i, j))
    strings => validEdges
end