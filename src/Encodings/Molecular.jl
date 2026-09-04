"""
    checkSpinSectoredEnc(enc::NTuple{2, PairwiseSumEnc}, explicitError::Bool=false) -> 
    Bool

Return `true` if the two Dirac fermionic encodings in `enc` can jointly serve as the two 
spin sectors of one composite fermionic encoding. Specifically, these two sector encodings 
must obey all the following conditions:
- They must cover disjoint sets of fermionic modes
- Each sector must pair equal numbers of annihilation and creation operators
- The concatenation of the two sectors must form a valid Dirac fermionic encoding (verified 
  via [`checkDiracEnc`](@ref)).

If any condition fails, `checkSpinSectoredEnc` returns `false`, or throws an error when 
`explicitError` is set to `true`.
"""
function checkSpinSectoredEnc((sec1Enc, sec2Enc)::NTuple{2, PairwiseSumEnc}, 
                              explicitError::Bool=false)::Bool
    if sec1Enc === sec2Enc || sec1Enc == sec2Enc
        if explicitError
            throw(ArgumentError("`sec1Enc` and `sec2Enc` should not be equal, as they "*
                                "correspond to separate spin sectors."))
        else
            false
        end
    else
        for (encLocal, title) in ((sec1Enc, "sec1Enc"), (sec2Enc, "sec2Enc"))
            if length(encLocal.first) != length(encLocal.second)
                if explicitError
                    throw(ArgumentError("`$title.first` should have the same length as "*
                                        "`$title.second`."))
                else
                    return false
                end
            end
        end
        compEnc = vcat(sec1Enc.first, sec2Enc.first) => vcat(sec1Enc.second, sec2Enc.second)
        checkDiracEnc(compEnc, explicitError)
    end
end


"""
    formatSpinSectoredEnc(enc::PairwiseSumEnc, spinOrbNumPair::NTuple{2, Integer}) -> 
    NTuple{2, PairwiseSumEnc}

Split a (spin-1/2) Dirac fermionic encoding `enc` into two spin-sector encodings, assigning 
its first `spinOrbNumPair[begin]` modes to the first (spin) sector and the following 
`spinOrbNumPair[end]` modes to the second (spin) sector; any remaining modes of `enc` 
belong to neither sector. `enc` is validated via [`checkDiracEnc`](@ref); both elements 
of `spinOrbNumPair` must be non-negative, and `enc` must contain at least 
`sum(spinOrbNumPair)` modes. The returned sector encodings are **views** into `enc`, so 
they share its underlying encoding operators.
"""
function formatSpinSectoredEnc(enc::PairwiseSumEnc, spinOrbNumPair::NTuple{2, Integer})
    checkDiracEnc(enc, true)
    nOp = length(enc.first)
    if any(x->x<0, spinOrbNumPair)
        throw(DomainError(spinOrbNumPair, "Both elements in `spinOrbNumPair` should be "*
                                          "non-negative."))
    end
    nSpinOrb = sum(spinOrbNumPair)
    nOp < nSpinOrb && throw(ArgumentError("`enc` must support at least $nSpinOrb modes."))

    annOps, creOps = enc
    s1 = first(spinOrbNumPair)
    s2 = nOp - nSpinOrb
    spin1Enc = (@view annOps[ begin:(begin+s1-1)]) => (@view creOps[ begin:(begin+s1-1)])
    spin2Enc = (@view annOps[(begin+s1):(end-s2)]) => (@view creOps[(begin+s1):(end-s2)])
    (spin1Enc, spin2Enc)
end

"""
    formatSpinSectoredEnc(enc::NTuple{2, PairwiseSumEnc}, 
                          spinOrbNumPair::NTuple{2, Integer}) -> 
    NTuple{2, PairwiseSumEnc}

Validate a given pair of spin-sector encodings via [`checkSpinSectoredEnc`](@ref) and 
return a pair of spin-sector encodings where the `i`-th sector is a **view** into exactly 
first `i_k = spinOrbNumPair[begin+i-1]` modes from the corresponding sector of `enc`. 
Therefore, the returned pair shares underlying encoding operators with `enc`, and the 
`i`-th sector of `enc` must contain at least that `i_k` modes.
"""
function formatSpinSectoredEnc(enc::NTuple{2, PairwiseSumEnc}, 
                               spinOrbNumPair::NTuple{2, Integer})
    checkSpinSectoredEnc(enc, true)

    map(spinOrbNumPair, enc, ("begin", "end")) do nOrb, sec, iName
        annOps, creOps = sec
        nOp = length(annOps)
        if nOp < nOrb
            throw(ArgumentError("`length(enc[$iName].first)=$nOp` should be no less than "*
                                "$nOrb."))
        end
        (@view annOps[ begin:(begin+nOrb-1)]) => (@view creOps[ begin:(begin+nOrb-1)])
    end
end


"""
    PairedOrder <: $StructuredType

The singleton format tag for the operator string ordering in which each particle's 
creation-annihilation operator pair stays adjacent: 

    c_1 * a_1 * c_2 * a_2 * ... * c_N * a_N

where `c_n == (a_n)'` (`1 <= n <= N`).
"""
struct PairedOrder <: StructuredType end #> (1', 1, 2', 2, ..., N', N)

"""
    NormalOrder <: $StructuredType

The singleton format tag for the normal ordering of a given operator string, which places 
every creation operator to the left of every annihilation operator: 

    c_1 * c_2 * ... * c_N * a_N * ... * a_2 * a_1

where `c_n == (a_n)'` (`1 <= n <= N`).
"""
struct NormalOrder <: StructuredType end #> (1', 2', ..., N', N, ..., 2, 1)

const NBodyOpFormat = Union{PairedOrder, NormalOrder}


"""
    genNBodyOperator(format::$NBodyOpFormat, enc::NTuple{2, PairwiseSumEnc}, 
                     spinSecConfig::NTuple{N, Bool}, 
                     modeIdxConfig::NTuple{N, NTuple{2, Integer}}, 
                     checkEncoding::Bool=true) -> 
    PauliSum

Return a spin-1/2 `N`-particle operator monomial as the product of `PauliSum` provided by 
the input encoding `enc`. For particle `p` (`1 <= p <= N`), `spinSecConfig[begin+p-1]` 
specifies its spin `sector` (`false` for the first element of `enc`, `true` for the 
second), and `(iCre, iAnn) = modeIdxConfig[begin+p-1]` specifies the modes associated with 
its creation and annihilation operators within that `sector`: 

    c_p' = sector.second[begin+iCre-1];     a_p = sector.first[begin+iAnn-1]

Additionally, `format` specifies the operator ordering of the monomial: 
[`PairedOrder`](@ref) produced pairwise ordering (similar to the Chemist notation for the 
indexing of two-body molecular integrals), and [`NormalOrder`](@ref) produces the normal 
ordering: `c_1' c_2' ... c_N' a_N ... a_2 a_1`. When `checkEncoding` is set to `true`, 
`enc` is validated via [`checkSpinSectoredEnc`](@ref) before the construction.
"""
function genNBodyOperator(format::NBodyOpFormat, enc::NTuple{2, PairwiseSumEnc}, 
                          spinSecConfig::NonEmptyTuple{Bool, M}, 
                          modeIdxConfig::NonEmptyTuple{NTuple{2, Integer}, M}, 
                          checkEncoding::Bool=true)::PauliSum where {M}
    checkEncoding && checkSpinSectoredEnc(enc, true)

    if format isa PairedOrder
        mapreduce(*, spinSecConfig, modeIdxConfig) do isSec2, idxPair
            iCre, iAnn = idxPair #> `iCre` and `iAnn` belong to the same particle
            spinSec = enc[begin+isSec2]
            spinSec.second[begin+iCre-1] * spinSec.first[begin+iAnn-1]
        end
    else #> `format isa NormalOrder`
        mapreduce(*, (true, false), (0:(+1):M, M:(-1):0)) do isCreOp, particlePtr
            mapfoldl(*, particlePtr) do offset
                isSec2 = spinSecConfig[begin+offset]
                iOpPair = modeIdxConfig[begin+offset]
                ops = enc[begin+isSec2][begin+isCreOp]
                ops[begin+iOpPair[end-isCreOp]-1]
            end
        end
    end
end


function getOrbSecLabel(spinIdxFirstKeys::NonEmptyTuple{Tuple{Bool, Integer}, M}) where {M}
    buffer = ntuple(_->0, Val(M+1))
    header = 0

    for offset in 0:M
        keyTarget = spinIdxFirstKeys[begin+offset]
        matchedIdx = 0

        for idx in 1:offset
            keyLabeled = spinIdxFirstKeys[begin+idx-1]

            if keyTarget == keyLabeled
                matchedIdx = idx
                break
            end
        end

        label = iszero(matchedIdx) ? (header += 1) : buffer[begin+matchedIdx-1]
        buffer = buffer .+ ntuple(i->ifelse(i==offset+1, label, 0), Val(M+1))
    end

    buffer
end


"""
    genNBodyOperatorSum(format::$NBodyOpFormat, enc::NTuple{2, PairwiseSumEnc}, 
                        orbInte::AbstractArray{C, D}, spinSecConfig::NTuple{N, Bool}, 
                        iModeStart::NTuple{N, Integer}=ntuple(_->1, Val(N)); 
                        particleExch::Bool=true, 
                        hermiticity::Bool=true, 
                        idxPairSymm::NTuple{N, Bool}=ntuple(_->false, Val(N)), 
                        checkInput::Bool=true) where 
                       {T<:Real, C<:Union{Complex{T}, T}, D, N} -> 
    PauliSum

Return a spin-1/2 `N`-particle operator summation as the linear combination of products of 
`PauliSum` provided by the input encoding `enc`. The linear coefficients of the `PauliSum` 
products are specified by the `N`-particle integral tensor `orbInte` (`D == 2N`), whose 
axes follow the index-pair layout: axes `(2p-1, 2p)` host the index pair 
`(c_p, a_p) == ((a_p)', a_p)` of particle `p` (`1 <= p <= N`). Specifically, index `i_k` 
along axis `k` of `orbInte` maps to the mode `iModeStart[begin+p-1] + i_k - 1` of particle 
`p = (k+1) ÷ 2` in the one of the two spin sectors specified by `spinSecConfig[begin+p-1]`. 
Subsequently, the returned summation is evaluated as 

    prefactor * sum(orbInte[idx] * monomial(idx) for idx in CartesianIndices(orbInte))

where each `monomial(idx)` is generated by [`genNBodyOperator`](@ref) in the operator 
ordering determined by `format`; the value of `prefactor` is set based on `particleExch` 
following two validation protocols:

- `particleExch=true`: the core operator underlying the integration expression for 
  `orbInte` is asserted to be invariant under exchanging the particle coordinates of the 
  many-particle wavefunctions it acts on. Consequently, particles sharing both the same 
  spin sector and the same mode window (range of modes indexing a subset of mode operators 
  in `enc`) form interchangeable classes. Accordingly, `prefactor` is set to the inverse of 
  the product of the class-size factorials, compensating the over-counting of unordered 
  particle selections by ordered indices in the summation. Simultaneously, the axes of 
  `orbInte` that are associated with the same spin sectors but different mode windows must 
  occupy disjoint windows (i.e., no partially shared mode indices are allowed). 
  When `checkInput=true`, the consistency between the index-pair exchange symmetry of 
  `orbInte` and the (index pairs for the) interchangeable classes determined by 
  `spinSecConfig` and `iModeStart`, is checked.

- `particleExch=false`: every entry of `orbInte` is the literal weight of its monomial, 
  meaning `prefactor` is fixed at `1`, and no exchange symmetry is validated for `orbInte` 
  and assumed for its underlying operator. As a result, tensor elements with distinct 
  indices may be added together to be associated with the same operator monomial (after 
  reforming them in canonical form, e.g., 
  `canonicalize!(c_i * c_j * a_j * a_i) == canonicalize!(c_j * c_i * a_i * a_j)` with 
  `c_i == (a_i)' && c_j == (a_j)'`).

Aside from `particleExch`, there are two more optional keyword arguments that are used to 
validate the structure of `orbInte` when `checkInput=true`: 

- `hermiticity=true`: `orbInte` must have pairwise Hermiticity (checked via 
  [`checkInteHermiticity`](@ref)). Under `particleExch=true`, this Hermiticity 
  guarantees the Hermiticity of the returned operator summation; under 
  `particleExch=false`, it only guarantees the same operator-summation Hermiticity when 
  `format = NormalOrder()`.

- `idxPairSymm=true`: `orbInte` must have index-pair transposition symmetry (checked via 
  [`checkIntePairTransposition`](@ref)).

!!! info
    When `checkInput` is set to `true`, `enc` is also checked via 
    [`checkSpinSectoredEnc`](@ref), and each mode window is checked against the capacity of 
    its sector in `enc`.
"""
function genNBodyOperatorSum(format::NBodyOpFormat, enc::NTuple{2, PairwiseSumEnc}, 
                             orbInte::AbstractArray{T, D}, 
                             spinSecConfig::NonEmptyTuple{Bool, M}, 
                             iModeStart::NonEmptyTuple{Integer, M}=ntuple(_->1, Val(M+1)); 
                             particleExch::Bool=true, 
                             hermiticity::Bool=true, 
                             idxPairSymm::NonEmptyTuple{Bool}=ntuple(_->false, Val(M+1)), 
                             checkInput::Bool=true) where {T<:RealOrComplex, D, M}
    N = M + 1
    D == 2N || throw(ArgumentError("`ndims(orbInte)` must equal `2M+2==$(2N)`."))
    if any(i <= 0 for i in iModeStart)
        throw(DomainError(iModeStart, "All elements of `iModeStart` must be positive."))
    end

    orbSecLabel = getOrbSecLabel(map(tuple, spinSecConfig, iModeStart))

    if particleExch
        for q in 2:N, p in 1:(q-1)
            if spinSecConfig[begin+p-1] == spinSecConfig[begin+q-1] &&
                 orbSecLabel[begin+p-1] !=   orbSecLabel[begin+q-1]
                pHead, qHead = iModeStart[begin+p-1], iModeStart[begin+q-1]
                pTail = pHead + size(orbInte, 2p-1) - 1
                qTail = qHead + size(orbInte, 2q-1) - 1
                if max(pHead, qHead) <= min(pTail, qTail)
                    throw(ArgumentError("Particles `$p` and `$q` have different mode "*
                                        "(index) windows, $(pHead:pTail) and "*
                                        "$(qHead:qTail), under the same spin sector. "*
                                        "When `particleExch=true`, these two windows "*
                                        "should have no index overlap."))
                end
            end
        end
    end

    if checkInput
        checkSpinSectoredEnc(enc, true)
        inteShape = size(orbInte)
        orbSecExtents = ntuple(i->inteShape[begin+2i-2], Val(N))

        checkNBodyInteTensor(orbInte, orbSecExtents, orbSecLabel, idxPairSymm, true; 
                             particleExch, hermiticity)

        #> After the shape of `orbInte` is verified to be consistent with `orbSecExtents`
        for (n, isSec2, idx) in zip(1:N, spinSecConfig, eachindex(iModeStart))
            iStart = iModeStart[idx]
            windowSize = length(enc[begin+isSec2].first) - iStart + 1
            nOrb = orbSecExtents[begin+n-1]
            if nOrb > windowSize
                throw(ArgumentError("The window size (bounded by `iModeStart[$idx]`) for "*
                                    "`enc` is $windowSize. It is not large enough to be "*
                                    "associated with the $n-th axis of `orbInte`, which "*
                                    "has an extent of $nOrb."))
            end
        end
    end

    realT = (typeof∘inv∘one∘real)(T)
    coreT = Complex{realT}
    cache = Dict{PauliStr, coreT}()

    prefactor = one(realT)
    if particleExch
        for c in 1:maximum(orbSecLabel)
            prefactor /= (realT∘factorial∘count)(isequal(c), orbSecLabel)
        end
    end
    iFirstAxial = map(first, axes(orbInte))

    for carteIdx in CartesianIndices(orbInte)
        inteCoeff = orbInte[carteIdx]
        iszero(inteCoeff) && continue

        idxTuple = Tuple(carteIdx)
        iPairs = map(ntuple(identity, Val(N)), iModeStart) do i, iStart
            m, n = (2i - 1), 2i
            offset = (iStart, iStart) .- (iFirstAxial[begin+m-1], iFirstAxial[begin+n-1])
            offset .+ (idxTuple[begin+m-1], idxTuple[begin+n-1])
        end

        op = genNBodyOperator(format, enc, spinSecConfig, iPairs, false)

        for (str, encCoeff) in zip(op.str, op.coeff)
            opCoeff = prefactor * encCoeff * inteCoeff
            cache[str] = get(cache, str, zero(coreT)) + coreT(opCoeff)
        end
    end

    PauliSum((collect∘keys)(cache), (collect∘values)(cache))
end

"""
    gen1BodyOperatorSum(oneSecEnc::PairwiseSumEnc, orbInte::AbstractArray{C, D}, 
                        iModeStart::Integer=1; hermiticity::Bool=true, 
                        idxPairSymm::Bool=false, checkInput::Bool=true) where 
                       {T<:Real, C<:Union{Complex{T}, T}, D} -> 
    PauliSum

Return a one-particle operator summation as the linear combination of products of 
`PauliSum` provided by the input encoding `oneSecEnc`. The linear coefficients of the 
`PauliSum` products are specified by the one-particle integral matrix `orbInte`, for which 
the elements of each (one-based) index pair `(i, j)` also map to the creation operator 
`oneSecEnc.second[begin+iModeStart+i-2]` and the annihilation operator 
`oneSecEnc.first[begin+iModeStart+j-2]`, respectively.

This function is a specialization of [`genNBodyOperatorSum`](@ref) in the case of `N=1`, 
for which the two operator orderings, [`PairedOrder`](@ref) and [`NormalOrder`](@ref), 
coincide. When `checkInput` is set to `true`, `oneSecEnc` is validated via 
[`checkDiracEnc`](@ref), the mode window (see docstring for `genNBodyOperatorSum` for more 
details) is checked against the capacity of `oneSecEnc`, and the symmetries of `orbInte` 
(as well as the resulting operator summation) asserted by `hermiticity` and `idxPairSymm` 
are verified via [`checkNBodyInteTensor`](@ref).
"""
function gen1BodyOperatorSum(oneSecEnc::PairwiseSumEnc, orbInte::AbstractArray{T, D}, 
                             iModeStart::Integer=1; hermiticity::Bool=true, 
                             idxPairSymm::Bool=false, checkInput::Bool=true) where 
                            {T<:RealOrComplex, D}
    idxPairSymm = (idxPairSymm,)

    if checkInput
        checkDiracEnc(oneSecEnc, true)

        nOrb = size(orbInte, 1)
        checkNBodyInteTensor(orbInte, (nOrb,), (1,), idxPairSymm, true; hermiticity)

        windowSize = length(oneSecEnc.first) - iModeStart + 1
        if nOrb > windowSize
            throw(ArgumentError("The window size (bounded by `iModeStart`) for "*
                                "`oneSecEnc` is $windowSize. It is not large enough to be "*
                                "associated with the first axis of `orbInte`, which has "*
                                "an extent of $nOrb."))
        end
    end

    genNBodyOperatorSum(PairedOrder(), (oneSecEnc, oneSecEnc), orbInte, (false,), 
                        (iModeStart,); hermiticity, idxPairSymm, checkInput=false)
end


const MolInteTensor1B2B{T<:RealOrComplex, T1<:AbstractMatrix{T}, T2<:AbstractArray{T, 4}} = 
      Tuple{T1, T2}

const OptSpinSectoredEnc = Union{NTuple{2, PairwiseSumEnc}, PairwiseSumEnc}

"""
    encodeElecHam(format::$NBodyOpFormat, 
                  enc::Union{NTuple{2, PairwiseSumEnc}, PairwiseSumEnc}, 
                  inte1B2BSpin1::Tuple{AbstractMatrix{C}, AbstractArray{C, 4}}, 
                  inte1B2BSpin2::Tuple{AbstractMatrix{C}, AbstractArray{C, 4}}, 
                  inte2BCross::AbstractArray{C, 4}; 
                  hermiticity::Bool=true, 
                  idxPairSymm::NTuple{2, Bool}=ntuple(_->(hermiticity && C<:Real), 2)
                  ) where {T<:Real, C<:Union{Complex{T}, T}} -> 
    PauliSum

Return `PauliSum`-based encoding (representation) of the molecular electronic Hamiltonian 

    H_{elec} = ∑_s H_s + ∑_{i,j,m,n} inte2BCross[i,j,m,n] c_{i,1} c_{m,2} a_{n,2} a_{j,1}

where `c_{i,s} = (a_{i,s})'` is the Dirac creation operator for mode `i` in spin sector 
`s`, represented by products of single-mode fermionic operator from the input encoding 
`enc` reordered by [`formatSpinSectoredEnc`](@ref)`(enc)`; `inte2BCross` is the 
cross-spin-sector two-body spatial molecular integral tensor and `H_s` is a per-spin-sector 
Hamiltonian fragment 

    H_s = ∑_{i,j} h1_s[i,j] c_{i,s} a_{j,s} + 
          (1/2) ∑_{i,j,m,n} h2_s[i,j,m,n] c_{i,s} c_{m,s} a_{n,s} a_{j,s}

with `(h1_s, h2_s)` respectively being one-body and two-body tensors of per-spin-sector 
molecular integrals, specified by either `inte1B2BSpin1` or `inte1B2BSpin2`. Both 
per-spin-sector two-body tensors, as well as `inte2BCross`, follow the index-pair layout 
(i.e., the Chemist notation for two-body molecular integrals); the numbers of 
spatial-orbital modes per spin sector `(n1, n2)`, are inferred from `size(inte2BCross)`. 
Hence, the per-spin-sector tensors must match these counts, whereas the total number of 
spin-orbital modes covered by `enc` can exceed `n1 + n2`. 

## Deciding the ordering of two-body terms

The form of the two-body terms in `H_{elec}` (as well as in `H_s`) described above 

        c_{i,s1} c_{m,s2} a_{n,s2} a_{j,s1}

follows the normal ordering convention, which can be enforced upon the returned encoding by 
setting `format = NormalOrder()`. Alternatively, `format = PairedOrder()` encodes the 
two-body term in the form (associated with the same two-body tensor)

        c_{i,s1} a_{j,s1} c_{m,s2} a_{n,s2}

Consequently, for the resulting encoding to represent the same electronic Hamiltonian, the 
coefficient matrix for the one-body terms (`c_{i,s} a_{j,s}`) can no longer directly be the 
one-body tensor (e.g., first(inte1B2BSpin1)), but is obtained by subtracting a `residue`:

        residue[i, j] = (1/2) * ∑_k h2_s[i, k, k, j]

This compensation is carried out automatically (via [`formatMolecularInteData`](@ref)) 
when `format = PairedOrder()`, hence `encodeElecHam` always returns an equivalent encoding 
for the same input molecular integral tensors (i.e., preserving the eigenspectrum of the 
underlying electronic Hamiltonian) regardless of the value of `format`.

## Simplified method

    encodeElecHam(format::$NBodyOpFormat, enc::OptSpinSectoredEnc, 
                  orbInte1B2B::Tuple{AbstractMatrix{C}, AbstractArray{C, 4}}; 
                  hermiticity::Bool=true, 
                  idxPairSymm::NTuple{2, Bool}=ntuple(_->(hermiticity && C<:Real), 2)) -> 
    PauliSum

The single-set form encodes both spin sectors and their coupling from one shared 
spatial-orbital tensor pair `orbInte1B2B`.

## Symmetry assertion
In both method signatures of `encodeElecHam`, keyword argument `hermiticity` (in default 
`true`) asserts the pairwise Hermiticity of all the input integral tensors (from 
`inte1B2BSpin1`, from `inte1B2BSpin2`, and `inte2BCross`), and each element of 
`idxPairSymm` asserts the index-pair transposition symmetry in each spin sector for these 
tensors, which is imposed by the default value of `idxPairSymm` exactly when the tensors 
are real and pairwise Hermitian, matching the symmetry of real spatial orbitals under the 
Coulomb interaction. As a result, `encodeElecHam` (with default argument values) assumes 
that the input per-spin-sector two-body tensors hold an eight-fold symmetry (two-fold 
particle-exchange times four-fold index-pair transposition) as long as there elements are 
all real.
"""
function encodeElecHam(format::NBodyOpFormat, enc::OptSpinSectoredEnc, 
                       inte1B2BSpin1::MolInteTensor1B2B{T}, 
                       inte1B2BSpin2::MolInteTensor1B2B{T}, 
                       inte2BCross::AbstractArray{T, 4}; 
                       hermiticity::Bool=true, 
                       idxPairSymm::NTuple{2, Bool}=ntuple(_->(hermiticity && T<:Real), 2)
                       ) where {T<:RealOrComplex}
    nOrbSpin1, nOrbSpin2 = size.(Ref(inte2BCross), (1, 3))
    crossERIShape = (nOrbSpin1, nOrbSpin1, nOrbSpin2, nOrbSpin2)
    if size(inte2BCross) != crossERIShape
        throw(ArgumentError("The size of `inte2BCross` should be $crossERIShape."))
    end

    checkInput = false
    secEnc = formatSpinSectoredEnc(enc, (nOrbSpin1, nOrbSpin2))
    inte1BSpin1, inte2BSpin1 = formatMolecularInteData(
        format, inte1B2BSpin1, nOrbSpin1; idxPairSymm=first(idxPairSymm), hermiticity)
    inte1BSpin2, inte2BSpin2 = formatMolecularInteData(
        format, inte1B2BSpin2, nOrbSpin2; idxPairSymm=last(idxPairSymm),  hermiticity)

    #> Check cross-section two-body integrals
    if inte2BCross != inte2BSpin1 && inte2BCross != inte2BSpin2
        checkNBodyInteTensor(inte2BCross, (nOrbSpin1, nOrbSpin2), (1, 2), idxPairSymm, 
                             true; hermiticity)
    end

    sec1Symm, sec2Symm = idxPairSymm

    h1Spin1 = gen1BodyOperatorSum(first(secEnc), inte1BSpin1; 
                                  hermiticity, idxPairSymm=sec1Symm, checkInput)
    h1Spin2 = gen1BodyOperatorSum( last(secEnc), inte1BSpin2; 
                                  hermiticity, idxPairSymm=sec2Symm, checkInput)

    h2Spin1 = genNBodyOperatorSum(format, secEnc, inte2BSpin1, (false, false); 
                                  hermiticity, idxPairSymm=(sec1Symm, sec1Symm), checkInput)
    h2Spin2 = genNBodyOperatorSum(format, secEnc, inte2BSpin2, (true , true ); 
                                  hermiticity, idxPairSymm=(sec2Symm, sec2Symm), checkInput)

    h2Cross = genNBodyOperatorSum(format, secEnc, inte2BCross, (false, true ); 
                                  hermiticity, idxPairSymm, checkInput)

    h1Spin1 + h1Spin2 + h2Spin1 + h2Spin2 + h2Cross
end

encodeElecHam(format::NBodyOpFormat, enc::OptSpinSectoredEnc, 
              orbInte1B2B::MolInteTensor1B2B{T}; hermiticity::Bool=true, 
              idxPairSymm::NTuple{2, Bool}=ntuple(_->(hermiticity && T<:Real), 2)) where 
             {T<:RealOrComplex} = 
encodeElecHam(format, enc, orbInte1B2B, orbInte1B2B, last(orbInte1B2B); 
              hermiticity, idxPairSymm)


"""
    formatMolecularInteData(::NormalOrder, 
                            inteData::Tuple{AbstractMatrix{C}, AbstractArray{C, 4}}, 
                            nOrbital::Integer=size(first(inteData), 1); 
                            hermiticity::Bool=true, 
                            idxPairSymm::Bool=(hermiticity && C<:Real)) where 
                           {T<:Real, C<:Union{Complex{T}, T}} -> 
    Tuple{AbstractMatrix{C}, AbstractArray{C, 4}}

Validate the format of molecular integrals (`first(inteData)` as the one-body matrix and 
`last(inteData)` as the two-body tensor) of a single spin sector return `inteData` 
unchanged to comply with the coefficient requirement for a second-quantized molecular 
electronic Hamiltonian under the `NormalOrder` format.

`nOrbital` sets the number of (spatial) orbitals in the sector to constrain the sizes of 
tensors in `inteData` via [`checkNBodyInteTensor`](@ref), which also takes in `hermiticity` 
and `idxPairSymm` as the values for the symmetry-assertion arguments with the same names, 
respectively. Since one spin sector draws on a single spatial-orbital set, `idxPairSymm` 
applies to the index pairs of both tensors; its default value asserts the index-pair 
transposition symmetry exactly when the tensors are real and pairwise Hermitian, matching 
the symmetry of real spatial orbitals under the Coulomb interaction.
"""
function formatMolecularInteData(::NormalOrder, inteData::MolInteTensor1B2B{T}, 
                                 nOrbital::Integer=size(first(inteData), 1); 
                                 hermiticity::Bool=true, 
                                 idxPairSymm::Bool=(hermiticity && T<:Real)) where 
                                {T<:RealOrComplex}
    inte1B, inte2B = inteData
    nOrbPair = (nOrbital, nOrbital)
    oneBodyPairSymm = (idxPairSymm,)
    twoBodyPairSymm = (idxPairSymm, idxPairSymm)

    checkNBodyInteTensor(inte1B, (nOrbital,), (1,  ), oneBodyPairSymm, true; hermiticity)
    checkNBodyInteTensor(inte2B,  nOrbPair,   (1, 1), twoBodyPairSymm, true; hermiticity)

    inteData
end

"""
    formatMolecularInteData(::PairedOrder, 
                            inteData::Tuple{AbstractMatrix{C}, AbstractArray{C, 4}}, 
                            nOrbital::Integer=size(first(inteData), 1); 
                            hermiticity::Bool=true, 
                            idxPairSymm::Bool=(hermiticity && C<:Real)) where 
                            {T<:Real, C<:Union{Complex{T}, T}} -> 
    Tuple{AbstractMatrix, AbstractArray{C, 4}}

Validate the format of molecular integrals (`first(inteData)` as the one-body matrix and 
`last(inteData)` as the two-body tensor) of a single spin sector and return integral data 
`(newInte1B, last(inteData))` that comply with the coefficient requirement for an 
second-quantized molecular electronic Hamiltonian under the `PairedOrder` format.

`nOrbital` sets the number of (spatial) orbitals in the sector to constrain the sizes of 
tensors in `inteData` via [`checkNBodyInteTensor`](@ref), which also takes in `hermiticity` 
and `idxPairSymm` as the values for the symmetry-assertion arguments with the same names, 
respectively. Since one spin sector draws on a single spatial-orbital set, `idxPairSymm` 
applies to the index pairs of both tensors; its default value asserts the index-pair 
transposition symmetry exactly when the tensors are real and pairwise Hermitian, matching 
the symmetry of real spatial orbitals under the Coulomb interaction.

## Reformatted one-body integrals
The first element of the returned integral data, `newInte1B`, is the result of subtracting 
the coefficient matrix for the residue of the two-body term in the Hamiltonian under the 
`PairedOrder` format from `first(inteData)`:

    newInte1B[i, j] == inte1B[i, j] - (1/2) * sum(inte2B[i, k, k, j] for k in 1:nOrbital)

This correction on the one-body matrix follows from the two-body operator monomial identity 

    c_i a_j c_k a_l == c_i c_k a_l a_j + (j == k) * c_i a_l

so that assembling the `PairedOrder` two-body sum with the corrected one-body matrix 
produces a `PairedOrder` encoding equivalent to the `NormalOrder` encoding of the same 
molecular electronic Hamiltonian.

!!! info
    The returned one-body matrix is freshly allocated with a potentially promoted element 
    type (e.g., `Int` inputs promote to `Float64`, while float, rational, and complex 
    element types are preserved), whereas the returned two-body tensor is exactly 
    `last(inteData)`, not a copy.
"""
function formatMolecularInteData(::PairedOrder, inteData::MolInteTensor1B2B{T}, 
                                 nOrbital::Integer=size(first(inteData), 1); 
                                 hermiticity::Bool=true, 
                                 idxPairSymm::Bool=(hermiticity && T<:Real)) where 
                                {T<:RealOrComplex}
    inte1B, inte2B = formatMolecularInteData(NormalOrder(), inteData, nOrbital; 
                                              idxPairSymm, hermiticity)
    eleT = (typeof∘inv∘one)(eltype(inte1B))
    newInte1B = similar(inte1B, eleT, size(inte1B))
    offset = nOrbital - 1
    prefactor = inv(2|>eleT)

    for j in 0:offset, i in 0:offset
        residue = zero(eleT)
        for k in 0:offset #> No residue contribution from cross-spin two-body integrals
            residue += inte2B[begin+i, begin+k, begin+k, begin+j]
        end
        newInte1B[begin+i, begin+j] = inte1B[begin+i, begin+j] - prefactor * residue
    end

    (newInte1B, inte2B)
end
