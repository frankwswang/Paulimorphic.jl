export getFrustrationInfo

"""
    getFrustrationInfo(ham::PauliSum, edgeThreshold::Real=0; inclusive::Bool=false) -> 
    Pair{Vector{PauliStr}, Vector{ NTuple{2, Int} }}

Compute the (anticommutation) frustration graph of the Pauli strings in `ham`: each 
term (a weighted [`PauliStr`](@ref)) forms a vertex, and whether two vertices are joined by 
an edge is determined by the anticommutator of their corresponding Pauli strings (evaluated 
by [`checkAntiCom`](@ref)). The graph is returned not as a [`SimpleGraph`](@ref) but as its 
structural information: 

    vertices => edges

where `vertices::Vector{PauliStr}` are the Pauli strings stored inside `ham` and 
`edges::Vector{NTuple{2, Int}}` lists the anticommuting pairs as one-based index pairs
`(i, j)`, `i < j`, indexing into `vertices`. Such a graph records which terms in a 
target Hamiltonian fail to commute — e.g. as a basis for partitioning it into mutually 
commuting groups.

# Approximate frustration information
This method also supports returning an approximate frustration graph through the optional 
argument `edgeThreshold::Real`. Specifically, an edge `(i, j)` is included in `edges` only 
if the `i`-th and `j`-th `PauliStr` in `vertices` anticommute *and* the magnitude of the 
their coefficient product `c_ij` clears `edgeThreshold`:

    abs(c_ij) >  edgeThreshold    (inclusive = false)
    abs(c_ij) >= edgeThreshold    (inclusive = true )

Additionally, keyword `inclusive` selects whether the `c_ij` that equals `edgeThreshold` 
exactly is kept (`>=`) or dropped (`>`). When `edgeThreshold = 0`, `inclusive = true` 
will also admit anticommuting term even when their coefficient product is exactly zero, 
whereas `inclusive = false` excludes them. A negative `edgeThreshold` throws an 
`ArgumentError`. To prune *vertices* (terms) by their own magnitude beforehand, apply 
[`curtail`](@ref) to `ham` first.
"""
function getFrustrationInfo(ham::PauliSum, edgeThreshold::Real=0; inclusive::Bool=false)
    edgeThreshold < 0 && throw(ArgumentError("`edgeThreshold` must be non-negative."))

    magnitudes = map(abs, ham.coeff)
    validNodes = copy(ham.str)
    nodeNum = length(validNodes)
    validEdges = NTuple{2, Int}[]

    for i in 1:(nodeNum-1), j in (i+1):nodeNum
        weight = magnitudes[begin+i-1] * magnitudes[begin+j-1]
        bl = inclusive ? weight >= edgeThreshold : weight > edgeThreshold
        bl && getFrustrationInfoCore!(validEdges, validNodes, (i, j))
    end

    validNodes => validEdges
end

"""
    getFrustrationInfo(strings::AbstractVector{PauliStr}) -> 
    Pair{Vector{PauliStr}, Vector{ NTuple{2, Int} }}

Compute the (anticommutation) frustration graph of `strings`, treating every its element as 
a vertex. Two vertices are joined by an edge whenever their Pauli strings anticommute 
(via [`checkAntiCom`](@ref)). Same as `getFrustrationInfo(::PauliSum)`, this method returns 
the frustration graph by its underlying information of as a `Pair`:

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