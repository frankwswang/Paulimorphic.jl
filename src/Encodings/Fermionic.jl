export checkMajoranaEnc, genJordanWignerEnc, genParityEnc, genBravyiKitaevEnc, toDiracEnc, 
       checkDiracEnc

const PairwiseStrEnc{T1<:AbstractVector{PauliStr}, T2<:AbstractVector{PauliStr}} = 
      Pair{T1, T2}

const PairwiseSumEnc{T1<:AbstractVector{<:PauliSum}, T2<:AbstractVector{<:PauliSum}} = 
      Pair{T1, T2}


"""
    checkMajoranaEnc(enc::Pair{<:AbstractVector{PauliStr}, <:AbstractVector{PauliStr}}, 
                     explicitError::Bool=false) -> 
    Bool

Return `true` if `enc` forms a valid Majorana-operator encoding. It must contain a `Pair` 
of `AbstractVector` holding in total (positive) `2p` (Hermitian) Majorana operators 
`γ_1, ..., γ_{2p}` that satisfy the following anticommutation relation: 

    γ_i γ_j + γ_j γ_i == 2δ_{ij} I

In other words, they all carry real phases and mutually anticommute (verifiable by 
[`checkAntiCom`](@ref)).

Additionally, `enc` must be formatted such that
- All contained `PauliStr` explicitly act on the same number of sites 
  (i.e., [`countSites`](@ref) returns the same value)
- `enc.first`  contains `p` elements that represent ( odd-indexed) γ_1, ..., γ_{2p-1}
- `enc.second` contains `p` elements that represent (even-indexed) γ_2, ..., γ_{2p}

When `enc` fails to meet any necessary condition to form such an encoding, the failure is 
reported by returning `false` unless `explicitError=true`, in which case an `ArgumentError` 
identifying the violated condition (and the immediate offending operators) is thrown 
instead.
"""
function checkMajoranaEnc(enc::PairwiseStrEnc, explicitError::Bool=false)::Bool
    nMode = length(enc.first)

    if iszero(nMode)
        if explicitError
            throw(ArgumentError("`length(enc.first)` must be a positive integer."))
        else
            return false
        end
    end

    if nMode != length(enc.second)
        if explicitError
            throw(ArgumentError("`length(enc.first)` must equal length(enc.second)."))
        else
            return false
        end
    end

    nSite = (countSites∘first∘first)(enc)

    for (sec, ops) in enumerate(enc), (i, op) in enumerate(ops)

        if op.n != nSite
            if explicitError
                throw(ArgumentError("All operators held by `enc` should explicitly act on "*
                                    "the same number of sites. Compared to the site count "*
                                    "of the first operator (`first(enc)[begin]`), "*
                                    "operator $i in `enc[$sec]` disagrees."))
            else
                return false
            end
        end

        if op.phase == posImg || op.phase == negImg
            if explicitError
                throw(ArgumentError("Every operator held by `enc` should be Hermitian ("*
                                    "and an involution, i.e., squared to the identity). "*
                                    "Operator $i in `enc[$sec]` violates this condition "*
                                    "due to having an imaginary phase."))
            else
                return false
            end
        end
    end

    @inbounds for secPair in ((1, 1), (2, 2), (1, 2))
        sec1, sec2 = secPair
        sameSec = (sec1 == sec2)
        ops1 = enc[begin+sec1-1]
        ops2 = enc[begin+sec2-1]

        for i in 1:(nMode - sameSec), j in (sameSec ? i + 1 : 1):nMode
            if !checkAntiCom(ops1[begin+i-1], ops2[begin+j-1])
                if explicitError
                    throw(ArgumentError("Every pair of distinct operators in `enc` should "*
                                        "anticommute. Operator $i in `enc[$sec1]` and "*
                                        "operator $j in `enc[$sec2]` do not."))
                else
                    return false
                end
            end
        end
    end

    true
end


function checkEncodingCounts(nMode::Integer, nSite::Integer)
    nModeInt = Int(nMode)
    nSiteInt = Int(nSite)

    if nModeInt < 1
        throw(DomainError(nModeInt, "`Int(nMode)` must be positive."))
    end

    if nSiteInt < nModeInt
        throw(DomainError(nSiteInt, "`Int(nSite)` must be no less than `Int(nMode)`."))
    end

    (nModeInt, nSiteInt)
end


#>-- Reference(s) --<#
#> [DOI] 10.1007/BF01331938
"""
    genJordanWignerEnc(nMode::Integer, nSite::Integer=nMode) -> 
    Pair{Vector{PauliStr}, Vector{PauliStr}}

Generate the [Jordan–Wigner encoding](https://doi.org/10.1007/BF01331938) of `nMode` 
fermionic modes as a Majorana encoding (see [`checkMajoranaEnc`](@ref)): a collection of 
`2nMode` Hermitian, mutually anticommuting Pauli strings, each explicitly acting on `nSite` 
sites. Mode `p` is associated with the Majorana-operator pair, 

    γ_{2p-1} = Z_1 ⋯ Z_{p-1} X_p,    γ_{2p} = Z_1 ⋯ Z_{p-1} Y_p

As a result, the expression of each occupation-number vector state remains unchanged in the 
Pauli-Z basis: 

    |n_1, n_2, ..., n_p, ...>  ->  |n_1, n_2, ..., n_p, ...>

The maximum Pauli weight (i.e., the number of non-identity single-qubit Pauli operators) of 
`PauliStr` generated by this encoding grows linearly with `nMode`. `nMode` must be positive 
and `nSite` must be no less than `nMode`. When `nSite > nMode`, every site above `nMode` 
carries the identity.

# Returned format
The encoding is returned as a `res::Pair` such that the Majorana operators with odd indices 
(`γ_{2p-1}`) are stored in `res.first` and Majorana operators with even indices (`γ_{2p}`) 
are stored in `res.second`.
"""
function genJordanWignerEnc(nMode::Integer, nSite::Integer=nMode)::PairwiseStrEnc
    nModeInt, nSiteInt = checkEncodingCounts(nMode, nSite)
    oddMajs = Vector{PauliStr}(undef, nModeInt)
    evnMajs = Vector{PauliStr}(undef, nModeInt)
    symBuffer = Memory{PauliSym}(undef, nSiteInt)
    symBuffer .= symI #> Reused across modes; each `PauliStr` copies its content

    for p in 1:nModeInt
        symBuffer[begin+p-1] = symX
          oddMajs[begin+p-1] = PauliStr(symBuffer)
        symBuffer[begin+p-1] = symY
          evnMajs[begin+p-1] = PauliStr(symBuffer)
        symBuffer[begin+p-1] = symZ #> Reset to `symZ` for registering the next mode
    end

    oddMajs => evnMajs
end


#>-- Reference(s) --<#
#> [DOI] 10.1063/1.4768229
"""
    genParityEnc(nMode::Integer, nSite::Integer=nMode) -> 
    Pair{Vector{PauliStr}, Vector{PauliStr}}

Generate the [parity encoding](https://doi.org/10.1063/1.4768229) of `nMode` fermionic 
modes as a Majorana encoding (see [`checkMajoranaEnc`](@ref)): a collection of `2nMode` 
Hermitian, mutually anticommuting Pauli strings, each explicitly acting on `nSite` sites. 
Mode `p` is associated with the Majorana-operator pair, 

    γ_{2p-1} = Z_{p-1} X_p X_{p+1} ⋯ X_{nMode},    γ_{2p} = Y_p X_{p+1} ⋯ X_{nMode}

where `Z_{p-1}` is absent for `p == 1`.

As a result, site `j` stores the parity of the first `j` mode occupancies, making the 
encoding the prefix-sum dual of the Jordan–Wigner encoding. each occupation-number vector 
state is represented by its prefix-parity counterpart in the Pauli-Z basis:

    |n_1, n_2, ..., n_p, ...>  ->  |n_1, (n_1 ⊻ n_2), ..., (⊻_{1<=q<=p} n_q), ...>

The maximum Pauli weight (i.e., the number of non-identity single-qubit Pauli operators) of 
`PauliStr` generated by this encoding grows linearly with `nMode`. `nMode` must be positive 
and `nSite` must be no less than `nMode`. When `nSite > nMode`, every site above `nMode` 
carries the identity.

# Returned format
The encoding is returned as a `res::Pair` such that the Majorana operators with odd indices 
(`γ_{2p-1}`) are stored in `res.first` and Majorana operators with even indices (`γ_{2p}`) 
are stored in `res.second`.
"""
function genParityEnc(nMode::Integer, nSite::Integer=nMode)::PairwiseStrEnc
    nModeInt, nSiteInt = checkEncodingCounts(nMode, nSite)
    oddMajs = Vector{PauliStr}(undef, nModeInt)
    evnMajs = Vector{PauliStr}(undef, nModeInt)
    symBuffer = Memory{PauliSym}(undef, nSiteInt)
    symBuffer .= symI #> Reused across modes; each `PauliStr` copies its content
    symBuffer[begin:begin+nModeInt-1] .= symX #> Flipping mode 1 updates all bits in 1:nMode

    for p in 1:nModeInt
        #> Bit `p - 1` stores the parity of modes `1:(p-1)` (no such bit when `p == 1`)
        p > 1 && (symBuffer[begin+p-2] = symZ)
          oddMajs[begin+p-1] = PauliStr(symBuffer)
        p > 1 && (symBuffer[begin+p-2] = symI)
        symBuffer[begin+p-1] = symY                #> Only reset to `symZ` in next iteration
          evnMajs[begin+p-1] = PauliStr(symBuffer)
    end

    oddMajs => evnMajs
end


#>-- Reference(s) --<#
#> [DOI] 10.1006/aphy.2002.6254
#> [DOI] 10.1002/qua.24969
"""
    genBravyiKitaevEnc(nMode::Integer, nSite::Integer=nMode) -> 
    Pair{Vector{PauliStr}, Vector{PauliStr}}

Generate the [Bravyi–Kitaev encoding](https://doi.org/10.1006/aphy.2002.6254) of `nMode` 
fermionic modes as a Majorana encoding (see [`checkMajoranaEnc`](@ref)): a collection of 
`2nMode` Hermitian, mutually anticommuting Pauli strings, each explicitly acting on `nSite` 
sites. Mode `p` is associated with the Majorana-operator pair, 

    γ_{2p-1} = X_{U(p)} X_p Z_{P(p)},    γ_{2p} = X_{U(p)} Y_p Z_{P(p)\\F(p)},

where the update set `U(p)`, parity set `P(p)`, and flip set `F(p)` are the index sets (of 
mode `p`) determined by the Fenwick-tree (i.e., binary-indexed-tree) structure over an 
arbitrary (positive) number `nMode` of modes, following the formulation of the encoding for 
electronic-structure problems in this [review](https://doi.org/10.1002/qua.24969). The 
maximum Pauli weight (i.e., the number of non-identity single-qubit Pauli operators) of 
`PauliStr` generated by this encoding grows as `O(log(nMode))`. `nMode` must be positive 
and `nSite` must be no less than `nMode`. When `nSite > nMode`, every site above `nMode` 
carries the identity.

# Returned format
The encoding is returned as a `res::Pair` such that the Majorana operators with odd indices 
(`γ_{2p-1}`) are stored in `res.first` and Majorana operators with even indices (`γ_{2p}`) 
are stored in `res.second`.
"""
function genBravyiKitaevEnc(nMode::Integer, nSite::Integer=nMode)::PairwiseStrEnc
    nModeInt, nSiteInt = checkEncodingCounts(nMode, nSite)
    oddMajs = Vector{PauliStr}(undef, nModeInt)
    evnMajs = Vector{PauliStr}(undef, nModeInt)
    symBuffer = Memory{PauliSym}(undef, nSiteInt)

    #> Mode occupancies (items) are stored as Fenwick-tree partial sums mod 2 (bits)
    #>> orbital-mode index (p) -> spin-site index (i, j, k)
    for p in 1:nModeInt
        fill!(symBuffer, symI)

        #> Parity set P(p): bits needed to compute the parity prefix below the p-th mode
        #>> Correspondence: obtain the sum of (Boolean) items whose labels are in 1:(p-1)
        i = p - 1
        while i > 0
            symBuffer[begin+i-1] = symZ #> Write the operators mirroring the bits in P(p)
            i -= (i & (-i)) #> index bits change: 01010001 -> 01010000 -> 01000000 -> ...
        end

        #> Update set U(p): bits to be updated to reflect the update of the p-th mode
        #>> Correspondence: update of the Fenwick-tree buffer to modify the p-th item
        j = p + (p & (-p))
        while j <= nModeInt
            symBuffer[begin+j-1] = symX
            j += (j & (-j)) #> index bits change: 00000101 -> 00000110 -> 00001000 -> ...
        end

        #> Finish up the string representation of the odd-indexed Majorana operator
        symBuffer[begin+p-1] = symX
        oddMajs[begin+p-1] = PauliStr(symBuffer)

        #>   Flip set F(p): bits needed to compute the (occupancy) difference between the 
        #>                  p-th mode and the p-th bit (in the Fenwick-tree buffer)
        #>> Correspondence: obtain the sum (mod 2) of items in indices ((p & (p-1))+1):(p-1)
        k = p - 1
        while k > (p & (p - 1)) #> `(p & (p - 1)) == p - (p & (-p))`
            symBuffer[begin+k-1] = symI
            k -= (k & (-k))
        end #>> Remainder set: R(p) = P(p) ∖ F(p); F(p) == ∅ when `isodd(p)`

        #> Finish up the string representation of the even-indexed Majorana operator
        symBuffer[begin+p-1] = symY
        evnMajs[begin+p-1] = PauliStr(symBuffer)
    end

    oddMajs => evnMajs
end


"""
    toDiracEnc(::Type{T}, enc::Pair{<:AbstractVector{PauliStr}, <:AbstractVector{PauliStr}} 
               ) where {T<:Real} -> 
    Pair{Vector{ PauliSum{T} }, Vector{ PauliSum{T} }}

    toDiracEnc(enc::Pair{<:AbstractVector{PauliStr}, <:AbstractVector{PauliStr}}) -> 
    Pair{Vector{ PauliSum{Rational{Int}} }, Vector{ PauliSum{Rational{Int}} }}

Convert a valid Majorana encoding `enc` (verified via [`checkMajoranaEnc`](@ref) with 
argument `explicitError=true`) of `length(enc.first)` fermionic modes to a Dirac-operator 
encoding returned as `res`, whose stored operators obey the following anticommutation 
relations: 

    a_p (a_q)' + (a_q)' a_p == δ_{pq} I,    a_p a_q + a_q a_p == 0, 

where `a_p` is the annihilation operator stored at `res.first[p]` and `(a_p)'` is the 
creation operator stored at `res.second[p]`. Furthermore, the relations between the Dirac 
operators and the Majorana operators are as follows: 

    a_p = (γ_{2p-1} + im * γ_{2p}) / 2,    (a_p)' = (γ_{2p-1} - im * γ_{2p}) / 2, 

where `γ_{2p-1} = enc.first[begin+p-1]` and `γ_{2p} = enc.second[begin+p-1]`. Each Dirac 
operator is a two-term canonical-form [`PauliSum`](@ref). In the first method, `T` must be 
able to exactly represent `1//2` (e.g., a concrete `AbstractFloat` or `Rational` subtype); 
the second method falls back to the first method with `T=Rational{Int}`. The result in both 
methods does not reference any data in `enc`.
"""
function toDiracEnc(::Type{T}, enc::PairwiseStrEnc)::PairwiseSumEnc where {T<:Real}
    oneHalf = one(T) / 2
    if !(oneHalf isa T) || oneHalf + oneHalf != one(T)
        throw(ArgumentError("`T` must be able to exactly represent one half (`1//2`)."))
    end
    halfRe = Complex{T}(oneHalf, 0)
    halfIm = Complex{T}(0, oneHalf)
    checkMajoranaEnc(enc, true)
    nMode = length(enc.first)

    annOps = Vector{PauliSum{T}}(undef, nMode)
    creOps = Vector{PauliSum{T}}(undef, nMode)
    aCoeff = [halfRe,  halfIm]
    cCoeff = [halfRe, -halfIm]

    for p in 1:nMode
        majOdd  = enc.first[ begin+p-1]
        majEven = enc.second[begin+p-1]
        opPair  = [majOdd, majEven]
        annOps[p] = PauliSum(opPair, aCoeff)
        creOps[p] = PauliSum(opPair, cCoeff)
    end

    annOps => creOps
end

toDiracEnc(enc::PairwiseStrEnc) = toDiracEnc(Rational{Int}, enc)


"""
    checkDiracEnc(enc::Pair{<:AbstractVector{<:PauliSum}, <:AbstractVector{<:PauliSum}}, 
                  explicitError::Bool=false) -> 
    Bool

Return `true` if `enc` forms a valid Dirac-operator encoding. It must contain a `Pair` of 
`AbstractVector`, holding in total (positive) `2p` operators represented by `PauliSum`. 
Specifically, `enc.first` and `enc.second` must respectively store `p` annihilation 
operators (`a_i = enc.first[i]`) and creation operators (`c_j = enc.second[j]`), which 
satisfy the following adjoint condition and anticommutation relations: 

    c_i == (a_i)',    a_i c_j + c_j a_i == δ_{ij} I,    a_i a_j + a_j a_i == 0

for every mode pair `(i, j)`. The adjoint condition is checked structurally (via 
[`toAdjoint`](@ref)). The verified anticommutation relations are invariant under swapping 
`enc.first` and `enc.second`, so the annihilation-versus-creation role assignment is a 
positional convention carried by the `Pair` structure rather than a checkable property.

Additionally, `enc` must be formatted such that
- All contained `PauliSum` are in the canonical form (see [`PauliSum`](@ref) and 
  [`canonicalize!`](@ref) for more details)
- All contained `PauliSum` explicitly act on the same number of sites (i.e., 
  [`countSites`](@ref) returns the same value)

All operator comparisons are exact, so an encoding whose coefficients are subject to 
floating-point errors (e.g., mode-mixing amplitudes that are not exactly representable) 
can fail the verification despite being valid in exact arithmetic; exact coefficient types 
(e.g., `Rational` subtypes) do not have this limitation.

When `enc` fails to meet any necessary condition to form such an encoding, the failure is 
reported by returning `false` unless `explicitError=true`, in which case an `ArgumentError` 
identifying the violated condition (and the immediate offending operators) is thrown 
instead.
"""
function checkDiracEnc(enc::PairwiseSumEnc, explicitError::Bool=false)::Bool
    nMode = length(enc.first)

    if iszero(nMode)
        if explicitError
            throw(ArgumentError("`length(enc.first)` must be a positive integer."))
        else
            return false
        end
    end

    if nMode != length(enc.second)
        if explicitError
            throw(ArgumentError("`length(enc.first)` must equal length(enc.second)."))
        else
            return false
        end
    end

    annOps = enc.first
    creOps = enc.second
    nSite = (countSites∘first)(annOps)

    for (sec, ops) in enumerate(enc), (i, op) in enumerate(ops)
        if countSites(op) != nSite
            if explicitError
                throw(ArgumentError("All operators held by `enc` should explicitly act on "*
                                    "the same number of sites. Compared to the site count "*
                                    "of the first operator (`first(enc)[begin]`), "*
                                    "operator $i in `enc[$sec]` disagrees."))
            else
                return false
            end
        end
    end

    for p in 1:nMode
        if creOps[begin+p-1] != adjoint(annOps[begin+p-1])
            if explicitError
                throw(ArgumentError("Each creation operator should be the adjoint of the "*
                                    "annihilation operator of the same mode (i.e., "*
                                    "located at the same index). Operator $p in "*
                                    "`enc.second` violates this condition."))
            else
                return false
            end
        end
    end

    for p in 1:nMode, q in p:nMode
        if !isempty(evalAntiCom(annOps[begin+p-1], annOps[begin+q-1]).str)
            if explicitError
                throw(ArgumentError("Every pair of annihilation operators (including an "*
                                    "operator paired with itself) should anticommute to "*
                                    "zero. Operators $p and $q in `enc.first` do not."))
            else
                return false
            end
        end
    end

    idSum = PauliSum(Int, [PauliStr(nSite)])

    for p in 1:nMode, q in p:nMode
        antiCom = evalAntiCom(annOps[begin+p-1], creOps[begin+q-1])
        valid = (p == q) ? (antiCom == idSum) : isempty(antiCom.str)
        if !valid
            if explicitError
                target = (p == q) ? "the identity operator" : "zero"
                throw(ArgumentError("Operator $p in `enc.first` and operator $q in "*
                                    "`enc.second` should anticommute to $target. They "*
                                    "do not."))
            else
                return false
            end
        end
    end

    true
end