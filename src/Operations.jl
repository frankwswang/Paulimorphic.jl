export checkCommute, checkAntiCom, evalCommute, evalAntiCom, getFrustrationInfo



"""
    mul(s1::PauliStr, s2::PauliStr) -> PauliStr

Multiply two `PauliStr`, returning the product `s3::PauliStr` with the its associated phase 
folded into `.phase`.
"""
function mul(s1::PauliStr, s2::PauliStr)
    bl1 = iszero(s1.n)
    if bl1 || iszero(s2.n) #> Trivial case: s * I == s, phases combined
        controlStr = bl1 ? s2 : s1
        x3 = controlStr.x
        z3 = controlStr.z
        n3 = controlStr.n 
        phase = PhaseFactor((UInt8(s1.phase) + UInt8(s2.phase)) & 0x3)
    else
        x1, x2 = s1.x, s2.x
        z1, z2 = s1.z, s2.z
        n1, n2 = s1.n, s2.n
        len1, len2 = length(z1), length(z2)
        nWord, n3 = n1 <= n2 ? (len2, n2) : (len1, n1)
        z3 = Memory{UInt}(undef, nWord)
        x3 = Memory{UInt}(undef, nWord)
        nY1 = nY2 = nY3 = 0
        sgnParity = 0

        @inbounds for w in 1:nWord
            #> Pad shorter string with identities
            z1w = w <= len1 ? z1[begin+w-1] : zero(UInt)
            x1w = w <= len1 ? x1[begin+w-1] : zero(UInt)
            z2w = w <= len2 ? z2[begin+w-1] : zero(UInt)
            x2w = w <= len2 ? x2[begin+w-1] : zero(UInt)

            x3w = x1w ⊻ x2w
            z3w = z1w ⊻ z2w
            x3[begin+w-1] = x3w
            z3[begin+w-1] = z3w
            nY1 += count_ones(z1w & x1w)
            nY2 += count_ones(z2w & x2w)
            nY3 += count_ones(z3w & x3w)
            sgnParity += count_ones(x1w & z2w)
        end

        phase = (( Int(UInt8(s1.phase)) + Int(UInt8(s2.phase))
                   + 3*(nY1 + nY2) + nY3 + 2*(sgnParity & 1) ) & 3) |> UInt |> PhaseFactor
    end

    PauliStr(x3, z3, phase, n3)
end


function mul!(s::PauliStr, p::PhaseFactor)
    newPhase = PhaseFactor((UInt8(s.phase) + UInt8(p)) & 3)
    s.phase = newPhase
    s
end


"""
    scale!(s::PauliStr, p::PhaseFactor) -> PauliStr

Multiply `s` in place by a phase `p` (absorbed into `s.phase`), returning the mutated `s`.
"""
function scale!(s::PauliStr, p::PhaseFactor)
    newPhase = PhaseFactor((UInt8(s.phase) + UInt8(p)) & 3)
    s.phase = newPhase
    s
end

"""
    scale!(h::PauliSum, p::PhaseFactor) -> PauliSum

Multiply `h` in place by a phase `p`, returning the mutated `h`. `p` is first converted to 
[`evalPhase`](@ref)`(p)`, then absorbed into `h.coeff`. This is to match the canonical form 
of `PauliSum` in which its phases are carried by coefficients rather than the stored 
`PauliStr`.
"""
scale!(h::PauliSum, p::PhaseFactor) = UInt8(p) > 0 ? scale!(h, evalPhase(p)) : h

"""
    scale!(h::PauliSum, c::Union{Real, Complex}) -> PauliSum

Multiply `h` in place by a coefficient `c`, returning the mutated `h`. `h.coeff` is scaled 
in place, thus each product `h.coeff[i] * c` must be able to be converted to 
`eltype(h.coeff)`. No re-canonicalization is performed.
"""
scale!(h::PauliSum, c::RealOrComplex) = (h.coeff .*= c; h)


"""
    checkCommute(str1::PauliStr, str2::PauliStr) -> Bool

Return `true` if the Pauli strings `str1` and `str2` commute and `false` if they anticommute
(any two Pauli strings do one or the other). When the two strings span different numbers of 
sites, only the overlapping words are examined — the extra sites from the longer string act 
against implicit identities and does not affect commutation.
"""
function checkCommute(str1::PauliStr, str2::PauliStr)::Bool
    z1, x1 = str1.z, str1.x
    z2, x2 = str2.z, str2.x
    nWord = min(length(z1), length(z2))   #> Shorter string is complemented with identities
    parity = 0
    @inbounds for w in 1:nWord
        parity += count_ones(z1[w] & x2[w]) + count_ones(x1[w] & z2[w])
    end
    iseven(parity)
end


"""
    checkAntiCom(str1::PauliStr, str2::PauliStr) -> Bool

Return `true` if `str1` and `str2` anticommute and `false` if they commute. It is the 
logical negation of [`checkCommute`](@ref).
"""
function checkAntiCom(str1::PauliStr, str2::PauliStr)::Bool
    !checkCommute(str1, str2)
end


"""
    evalCommute(str1::PauliStr, str2::PauliStr) -> PauliSum{Int}

Return the commutator [`str1`, `str2`] = `str1`*`str2` - `str2`*`str1` as a `PauliSum`. 
Specifically, when the commutator is zero (when `str1` and `str2` commute), this function 
returns an empty `PauliSum` as the zero operator.
"""
function evalCommute(str1::PauliStr, str2::PauliStr)
    prod1 = mul(str1, str2)
    prod2 = mul(str2, str1)
    prod1 == prod2 ? PauliSum() : PauliSum([prod1, scale!(prod2, PhaseFactor(2))])
end


"""
    evalAntiCom(str1::PauliStr, str2::PauliStr) -> PauliSum{Int}

Return the anticommutator {`str1`, `str2`} = `str1`*`str2` + `str2`*`str1` as a `PauliSum`. 
Specifically, when the anticommutator is zero (when `str1` and `str2` anitcommute), this 
function returns an empty `PauliSum` as the zero operator.
"""
function evalAntiCom(str1::PauliStr, str2::PauliStr)
    prod1 = mul(str1, str2)
    prod2 = mul(str2, str1)
    prod1 == prod2 ? PauliSum([prod1, prod2]) : PauliSum()
end


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