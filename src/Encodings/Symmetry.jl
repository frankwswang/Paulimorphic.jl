using LinearAlgebra: Symmetric, Hermitian

"""
    checkInteHermiticity(inte::AbstractArray{C, D}, explicitError::Bool=false) where 
                        {T<:Real, C<:Union{Complex{T}, T}, D} -> 
    Bool

Return `true` if `inte`, an `N`-particle integral tensor (`N=D÷2`) whose axes follow the 
index-pair layout (axes `(2n-1, 2n)` host the index pairs of particle `n`), is Hermitian 
under simultaneously transposing every index pair `(c_n, a_n)`: 

    inte[c_1, a_1, ..., c_N, a_N] == inte[a_1, c_1, ..., a_N, c_N]'

If `inte` does not hold this symmetry, `checkInteHermiticity` returns `false`, or throws 
an error when `explicitError` is set to `true`. The tensor rank of `inte`, `D`, must be 
even; for every index pair, their corresponding axes must have equal extents.
"""
function checkInteHermiticity(inte::AbstractArray{<:RealOrComplex, D}, 
                              explicitError::Bool=false)::Bool where {D}
    iseven(D) || throw(ArgumentError("`ndims(inte)` must be even."))
    axisPerm = ntuple(d->(isodd(d) ? d+1 : d-1), Val(D))
    idx = findAxisPermViolation(inte, axisPerm, true)

    if idx !== nothing
        if explicitError
            throw(ArgumentError("`inte` should be Hermitian under simultaneously "*
                                "transposing every index pair: `inte[c_1, a_1, ..., "*
                                "c_N, a_N] == inte[a_1, c_1, ..., a_N, c_N]'` for all "*
                                "possible index values; but `inte[$idx]` violates it."))
        else
            return false
        end
    end

    true
end


"""
    checkInteParticleExchange(inte::AbstractArray{C, D}, posPair::NTuple{2, Integer}, 
                              explicitError::Bool=false) where 
                             {T<:Real, C<:Union{Complex{T}, T}, D} -> 
    Bool

Return `true` if `inte`, an `N`-particle integral tensor (`N=D÷2`) whose axes follow the 
index-pair layout (axes `(2n-1, 2n)` host the index pairs of particle `n`), is invariant 
under exchanging the index pairs of the two particles `p` and `q` specified by `posPair`: 

    inte[..., c_p, a_p, ..., c_q, a_q, ...] == inte[..., c_q, a_q, ..., c_p, a_p, ...]

An integral tensor may hold this symmetry when the underlying core operator 
`O(x_1, ..., x_N)` is invariant under exchanging the respective coordinates of particles 
`p` and `q`, `x_p` and `x_q`. In other words, these `N` particles are considered 
**identical** with respect to operations (e.g., interactions) characterized by `O`.

If `inte` does not hold this symmetry, `checkInteParticleExchange` returns `false`, or 
throws an error when `explicitError` is set to `true`. The tensor rank of `inte`, `D`, must 
be even; for index pairs `p` and `q`, their corresponding axes must have equal extents. In 
the case of `p == q`, i.e., no distinct particles are exchanged, 
`checkInteParticleExchange` always returns `true`.
"""
function checkInteParticleExchange(inte::AbstractArray{<:RealOrComplex, D}, 
                                   posPair::NTuple{2, Integer}, 
                                   explicitError::Bool=false)::Bool where {D}
    iseven(D) || throw(ArgumentError("`ndims(inte)` must be even."))
    nParticle = D ÷ 2
    p, q = minmax(Int(posPair[begin]), Int(posPair[end]))
    if !(1 <= p && q <= nParticle)
        throw(DomainError(posPair, "Both elements in `posPair` must be within "*
                                   "`1:$nParticle`."))
    end
    p == q && (return true)

    posShift = 2 * (q - p)
    axisPerm = ntuple(Val(D)) do d
        pos = (d + 1) ÷ 2
        pos == p ? d + posShift : (pos == q ? d - posShift : d)
    end
    idx = findAxisPermViolation(inte, axisPerm, false)

    if idx !== nothing
        if explicitError
            throw(ArgumentError("`inte` should be invariant under exchanging the index "*
                                "pairs of particle `$p` and particle `$q`; but "*
                                "`inte[$idx]` violates it."))
        else
            return false
        end
    end

    true
end


"""
    checkIntePairTransposition(inte::AbstractArray{C, D}, pos::Integer, 
                               explicitError::Bool=false) where 
                              {T<:Real, C<:Union{Complex{T}, T}, D} -> 
    Bool

Return `true` if `inte`, an `N`-particle integral tensor (`N=D÷2`) whose axes follow the 
index-pair layout (axes `(2n-1, 2n)` host the index pairs of particle `n`), is invariant 
under transposing the index pair `(a_p, c_p)` of particle `p = pos` (`1 <= p <= N`): 

    inte[..., c_p, a_p, ...] == inte[..., a_p, c_p, ...]

An integral tensor

    inte[c_1, a_1, ..., c_N, a_N] = 
    \\int_{dx_1 ... dx_N} O[f_{c_1}', f_{a_1}, ..., f_{c_N}', f_{a_N}](x_1, ..., x_N)

may hold such a symmetry when the action of the underlying core operator `O` onto all the 
basis wavefunctions `f` indexed by `(c_1, a_1, ..., c_N, a_N)` is invariant under the 
transposition of `(a_p, c_p)` inside the integration expression. For example, when `O` is 
the Coulomb operator (`1/r_{12}`), its corresponding two-particle integral tensor holds a 
four-fold symmetry of this type as long as all the basis wavefunctions used for the 
integration are real.

If `inte` does not hold this symmetry, `checkIntePairTransposition` returns `false`, or 
throws an error when `explicitError` is set to `true`. The tensor rank of `inte`, `D`, must 
be even; the axes corresponding to `(a_p, c_p)` must have equal extents.
"""
function checkIntePairTransposition(inte::AbstractArray{<:RealOrComplex, D}, pos::Integer, 
                                    explicitError::Bool=false)::Bool where {D}
    iseven(D) || throw(ArgumentError("`ndims(inte)` must be even."))
    nParticle = D ÷ 2
    if !(1 <= pos <= nParticle)
        throw(DomainError(pos, "`pos` must be within `1:$nParticle`."))
    end
    p = Int(pos)

    axisPerm = ntuple(d->(d == 2p-1 ? 2p : (d == 2p ? 2p-1 : d)), Val(D))
    idx = findAxisPermViolation(inte, axisPerm, false)

    if idx !== nothing
        if explicitError
            throw(ArgumentError("`inte` should be invariant under transposing the index "*
                                "pair of particle `$p`; but `inte[$idx]` violates it."))
        else
            return false
        end
    end

    true
end


"""
    checkNBodyInteTensor(inte::AbstractArray{C, D}, nOrb::NTuple{N, Integer}, 
                         orbSecLabel::NTuple{N, Integer}, 
                         idxPairSymm::NTuple{N, Bool}, 
                         explicitError::Bool=false; 
                         particleExch::Bool=true, 
                         hermiticity::Bool=true) where 
                        {T<:Real, C<:Union{Complex{T}, T}, D, N} -> 
    Bool

Return `true` if `inte` is an `N`-particle integral tensor (`N == D÷2 >= 1`), given a tuple 
of orbital-sector extents `nOrb` (i.e., `nOrb[begin+p-1]` specifies the mode count for 
particle `p`), and the index-pair transposition symmetry flags. More specifically, 
`idxPairSymm[begin+p-1]` indicates whether `inte[..., c_p, a_p, ...]` is invariant under 
the transposition of the index pair `(c_p, a_p)` for particle `p` (i.e., under `c_p, a_p = 
a_p, c_p`). For `inte` to be a valid integral tensor, its axes must follow the index-pair 
layout: axes `(2n-1, 2n)` host all the index pairs of particle `n` (`1<=n<=N`). Therefore, 
the size of `inte` must satisfy

    size(inte) == (nOrb[begin], nOrb[begin], ..., nOrb[end], nOrb[end])

and the axes sharing the same orbital-sector label must also have the same extents.

In addition to the shape of `inte`, the following symmetries are (at least partially) 
checked based on the values of the configurational arguments: 
- Hermiticity (when `hermiticity=true`): `inte` must be Hermitian under simultaneously 
  transposing every index pair (via [`checkInteHermiticity`](@ref)). 
- Particle-exchange invariance (when `particleExch=true`): `inte` must be invariant under 
  exchanging the index pairs of every two particles within the same orbital sector 
  specified by `orbSecLabel` (via [`checkInteParticleExchange`](@ref)). This invariance 
  holds under the assumption that the underlying operator for the integration expression of 
  `inte` has particle (coordinate) exchange symmetry.
- Index-pair transposition invariance: `inte` must be invariant under transposing the index 
  pair of every particle whose orbital sector is flagged as `true` by `idxPairSymm` 
  (via [`checkIntePairTransposition`](@ref)).

If any above symmetry check fails, or the size of `inte` does not match `nOrb`, 
`checkNBodyInteTensor` returns `false`, or throws an error when `explicitError` is set to 
`true`. Furthermore, the function throws an error regardless of the value of 
`explicitError` if any configurational arguments (including `nOrb`) are not in valid forms 
(e.g., `nOrb` containing any negative elements).

!!! info
    when `particleExch=true`, for `idxPairSymm` to be in a valid form (aside from being of 
    the allowed type), `idxPairSymm` must have the same equality pattern as `orbSecLabel`. 
    In other words, for any valid index pair `i` and `j` (`1<=i<=j<=N`), 
    `orbSecLabel[begin+i-1] == orbSecLabel[begin+j-1]` must imply 
    `idxPairSymm[begin+i-1] == idxPairSymm[begin+j-1]`.
"""
function checkNBodyInteTensor(inte::AbstractArray{<:RealOrComplex, D}, 
                              nOrb::NonEmptyTuple{Integer, M}, 
                              orbSecLabel::NonEmptyTuple{Integer, M}, 
                              idxPairSymm::NonEmptyTuple{Bool, M}, 
                              explicitError::Bool=false; 
                              particleExch::Bool=true, 
                              hermiticity::Bool=true)::Bool where {D, M}
    N = M + 1
    D != 2N && throw(ArgumentError("`2length(nOrb)` must equal `D = $D`."))

    for i in eachindex(nOrb)
        nOrbSec = nOrb[i]
        nOrbSec < 0 && throw(DomainError(nOrbSec, "`nOrb[$i]` must be non-negative."))
    end

    if !isIndexLabel(orbSecLabel)
        throw(ArgumentError("`orbSecLabel` is not in a valid form."))
    end

    for q in 2:N, p in 1:(q-1)
        if orbSecLabel[begin+p-1] == orbSecLabel[begin+q-1]
            if particleExch && idxPairSymm[begin+p-1] != idxPairSymm[begin+q-1]
                throw(ArgumentError("Particles `$p` and `$q` share the same sector label, "*
                                    "so `idxPairSymm[begin+$(p-1)]` and "*
                                    "`idxPairSymm[begin+$(q-1)]` must be equal."))
            end

            if nOrb[begin+p-1] != nOrb[begin+q-1]
                throw(ArgumentError("Particles `$p` and `$q` share the same sector label, "*
                                    "so `nOrb[begin+$(p-1)]` and `nOrb[begin+$(q-1)]` "*
                                     "must be equal."))
            end
        end
    end

    #> Tensor axes follow the index-pair layout: axes `(2n-1, 2n)` belong to particle `n`
    shape = ntuple(d->nOrb[begin + (d-1)÷2], Val(D))
    if size(inte) != shape
        if explicitError
            throw(ArgumentError("`inte` should be of size `$shape`."))
        else
            return false
        end
    end

    #> Type-aware fast path for one-body integral tensors
    if iszero(M)
        if eltype(inte) <: Real
            inte isa Union{Symmetric, Hermitian}
        else
            (inte isa Hermitian) && !idxPairSymm[begin]
        end && (return true)
    end

    if hermiticity
        checkInteHermiticity(inte, explicitError) || (return false)
    end

    #> Exchanging with the nearest same-label particle generates the full exchange group
    if particleExch
        for q in 2:N
            p = q - 1
            while p > 0 && orbSecLabel[begin+p-1] != orbSecLabel[begin+q-1]
                p -= 1
            end
            if p > 0
                checkInteParticleExchange(inte, (p, q), explicitError) || (return false)
            end
        end
    end

    for p in 1:N
        if idxPairSymm[begin+p-1]
            checkIntePairTransposition(inte, p, explicitError) || (return false)
        end
    end

    true
end
