export genJordanWignerEnc, genParityEnc, genBravyiKitaevEnc, checkMajoranaEnc, toDiracEnc, 
       checkDiracEnc, toMajoranaEnc, toMajoranaPair, buildMajoranaFrame

const PairwiseStrEnc{T1<:AbstractVector{PauliStr}, T2<:AbstractVector{PauliStr}} = 
      Pair{T1, T2}

const PairwiseSumEnc{T1<:AbstractVector{<:PauliSum}, T2<:AbstractVector{<:PauliSum}} = 
      Pair{T1, T2}


"""
    checkMajoranaEnc(enc::Pair{<:AbstractVector{PauliStr}, <:AbstractVector{PauliStr}}, 
                     explicitError::Bool=false) -> 
    Bool

    checkMajoranaEnc(enc::Pair{<:AbstractVector{<:PauliSum}, <:AbstractVector{<:PauliSum}}, 
                     explicitError::Bool=false) -> 
    Bool

Return `true` if `enc` forms a valid `PauliStr`-based or `PauliSum`-based Majorana encoding 
of `length(enc.first)` fermionic modes, where `enc.first` and `enc.second` respectively 
store the odd-indexed (`γ_{2p-1}`) and even-indexed (`γ_{2p}`) Majorana operators. Every 
stored operator `γ_i` must be Hermitian and the operators must also satisfy the following 
anticommutation relation: 

    γ_i γ_j + γ_j γ_i == 2δ_{ij} I. 

Additionally, `enc` must be formatted such that all contained Majorana operators explicitly 
act on the same number of sites (i.e., [`countSites`](@ref) returns the same value). All 
operator comparisons are exact, so an encoding whose coefficients are subject to 
floating-point rounding can fail the verification despite being valid in 
exact arithmetic; exact coefficient types (e.g., `Rational` subtypes) do not have this 
limitation. Note that this validity class is disjoint from the one verified by 
[`checkDiracEnc`](@ref).

When `enc` fails to meet any necessary condition to form such an encoding, the failure is 
reported by returning `false` unless `explicitError=true`, in which case an error 
identifying the violated condition (and the immediate offending operators) is thrown 
instead.
"""
function checkMajoranaEnc(enc::Union{PairwiseStrEnc, PairwiseSumEnc}, 
                          explicitError::Bool=false)::Bool
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

    firstOp = (first∘first)(enc)
    nSite = countSites(firstOp)
    checkSum = (firstOp isa PauliSum)
    offset = firstindex(enc) - 1

    for (sec, ops) in enumerate(enc), (i, op) in enumerate(ops)

        if countSites(op) != nSite
            if explicitError
                throw(ArgumentError("All operators held by `enc` should explicitly act on "*
                                    "the same number of sites. Compared to the site count "*
                                    "of the first operator (`first(enc)[begin]`), "*
                                    "operator $i in `enc[$(sec+offset)]` disagrees."))
            else
                return false
            end
        end

        if !isHermitian(op) #> Hermitian `PauliStr` is automatically an involution
            if explicitError
                throw(ArgumentError("Every operator held by `enc` should be Hermitian. "*
                                    "Operator $i in `enc[$(sec+offset)]` violates this "*
                                    "condition."))
            else
                return false
            end
        end

        if checkSum
            if !(isIdentity∘mul)(op, op)
                if explicitError
                    throw(ArgumentError("Every operator held by `enc` should be an "*
                                        "involution (i.e., squared to the identity "*
                                        "operator). Operator $i in `enc[$(sec+offset)]` "*
                                        "violates this condition."))
                else
                    return false
                end
            end
        end
    end

    @inbounds for secPair in ((1, 1), (2, 2), (1, 2))
        sec1, sec2 = secPair
        sameSec = (sec1 == sec2)
        idx1 = offset + sec1
        idx2 = offset + sec2
        ops1 = enc[idx1]
        ops2 = enc[idx2]

        for i in 1:(nMode - sameSec), j in (sameSec ? i + 1 : 1):nMode
            if !checkAntiCom(ops1[begin+i-1], ops2[begin+j-1])
                if explicitError
                    throw(ArgumentError("Every pair of distinct operators in `enc` should "*
                                        "anticommute. Operator $i in `enc[$idx1]` and "*
                                        "operator $j in `enc[$idx2]` do not."))
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

where `a_p` is the annihilation operator stored at `res.first[begin+p-1]` and `(a_p)'` is 
the creation operator stored at `res.second[begin+p-1]`. Furthermore, the relations between 
the Dirac operators and the Majorana operators are as follows: 

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
        annOps[begin+p-1] = PauliSum(opPair, aCoeff)
        creOps[begin+p-1] = PauliSum(opPair, cCoeff)
    end

    annOps => creOps
end

toDiracEnc(enc::PairwiseStrEnc) = toDiracEnc(Rational{Int}, enc)


"""
    toMajoranaPair(opPair::Pair{<:PauliSum, <:PauliSum}, checkForDiracPair::Bool=true) -> 
    Pair{<:PauliSum, <:PauliSum}

Convert a single-mode Dirac-operator pair `opPair`, with `opPair.first` and 
`opPair.second` respectively storing the annihilation operator `a` and the creation 
operator `c` of the same fermionic mode, into the corresponding pair of Majorana operators 
returned as `res`: 

    res.first = a + c,    res.second = im * (c - a). 

When `checkForDiracPair=true`, `opPair` is first verified to form a valid single-mode 
Dirac pair; any violation is reported by a thrown error, and both operators must explicitly 
act on the same number of sites for the verification to pass. When 
`checkForDiracPair=false`, the verification is skipped, in which case the output is 
physically meaningful only when the input is a valid Dirac-operator pair. The two operators 
held by `res` are in the canonical form and do not reference any data in `opPair`.

# Example
```jldoctest
julia> dEnc = toDiracEnc(genJordanWignerEnc(1));

julia> mPair = toMajoranaPair(dEnc.first[1] => dEnc.second[1]);

julia> mPair == (PauliSum(Int, [pauli"X"]) => PauliSum(Int, [pauli"Y"]))
true
```
"""
function toMajoranaPair(opPair::Pair{<:PauliSum, <:PauliSum}, checkForDiracPair::Bool=true)
    annOp, creOp = opPair

    if checkForDiracPair
        if PauliSum(creOp) != toAdjoint(annOp)
            throw(ArgumentError("`opPair.first` must be the Hermitian adjoint of "*
                                "`opPair.second`."))
        end

        if !(isIdentity∘evalAntiCom)(annOp, creOp)
            throw(ArgumentError("The anticommutator between `opPair.first` and "*
                                "`opPair.second` must equal an identity."))
        end

        for op in opPair
            if !checkAntiCom(op, op)
                throw(ArgumentError("Each operator in `opPair` must anticommute with "*
                                    "itself."))
            end
        end
    end

    gOdd = add(annOp, creOp)
    gEve = mul(add(creOp, mul(annOp, -1)), im)
    gOdd => gEve
end


#>-- Reference(s) --<#
#> [DOI] 10.21468/SciPostPhysLectNotes.54
"""
    checkDiracEnc(enc::Pair{<:AbstractVector{<:PauliSum}, <:AbstractVector{<:PauliSum}}, 
                  explicitError::Bool=false; strRestricted::Bool=false) -> 
    Bool

Return `true` if `enc` forms a valid Dirac-operator encoding. It must contain a `Pair` of 
`AbstractVector`, holding in total (positive) `2p` operators represented by `PauliSum`. 
Specifically, `enc.first` and `enc.second` must respectively store `p` annihilation 
operators (`a_i = enc.first[begin+i-1]`) and creation operators 
(`c_j = enc.second[begin+j-1]`), which satisfy the following adjoint condition and 
anticommutation relations: 

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

When `strRestricted=true`, `enc` must additionally be convertible to a Pauli-string-based 
Majorana encoding: for each mode `i`, `a_i + c_i` and `im * (c_i - a_i)` must each be 
representable as a `PauliStr` with a phase of exactly `1` or `-1`. Under the default 
`strRestricted=false`, encodings whose modes are represented as `PauliSum` beyond 
containing a single `PauliStr` are accepted as long as they satisfy the Hermitian adjoint 
condition and the anticommutation relations.

All operator comparisons are exact, so an encoding whose coefficients are subject to 
floating-point errors (e.g., mode-mixing amplitudes that are not exactly representable) 
can fail the verification despite being valid in exact arithmetic; exact coefficient types 
(e.g., `Rational` subtypes) do not have this limitation.

When `enc` fails to meet any necessary condition to form such an encoding, the failure is 
reported by returning `false` unless `explicitError=true`, in which case an `ArgumentError` 
identifying the violated condition (and the immediate offending operators) is thrown 
instead.
"""
function checkDiracEnc(enc::PairwiseSumEnc, explicitError::Bool=false; 
                       strRestricted::Bool=false)::Bool
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
    offset = firstindex(enc) - 1

    for (sec, ops) in enumerate(enc), (i, op) in enumerate(ops)
        if countSites(op) != nSite
            if explicitError
                throw(ArgumentError("All operators held by `enc` should explicitly act on "*
                                    "the same number of sites. Compared to the site count "*
                                    "of the first operator (`first(enc)[begin]`), "*
                                    "operator $i in `enc[$(sec+offset)]` disagrees."))
            else
                return false
            end
        end
    end

    for p in 1:nMode
        if adjoint(annOps[begin+p-1]) != PauliSum(creOps[begin+p-1])
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

    if strRestricted
        for p in 1:nMode
            for op in toMajoranaPair(enc.first[begin+p-1]=>enc.second[begin+p-1], false)
                isValidStr = if (isone∘countTerms)(op)
                    coeff = first(op.coeff)
                    coeff == 1 || coeff == -1
                else
                    false
                end #> `op` must equal a single Pauli string up to a factor of `1` or `-1`

                if !isValidStr
                    if explicitError
                        throw(ArgumentError("The two Majorana operators (converted via "*
                                            "`toMajoranaPair`) corresponding to mode $p "*
                                            "should both be single Pauli strings."))
                    else
                        return false
                    end
                end
            end
        end
    end

    true
end


"""
    toMajoranaEnc(::Type{PauliStr}, 
                  enc::Pair{<:AbstractVector{<:PauliSum}, <:AbstractVector{<:PauliSum}}) -> 
    Pair{Vector{PauliStr}, Vector{PauliStr}}

    toMajoranaEnc(::Type{PauliSum}, 
                  enc::Pair{<:AbstractVector{<:PauliSum}, <:AbstractVector{<:PauliSum}}) -> 
    Pair{<:AbstractVector{<:PauliSum}, <:AbstractVector{<:PauliSum}}

    toMajoranaEnc(enc::Pair{<:AbstractVector{<:PauliSum}, <:AbstractVector{<:PauliSum}}) -> 
    Pair{Vector{PauliStr}, Vector{PauliStr}}

Convert a valid Dirac-operator encoding `enc` (verified via [`checkDiracEnc`](@ref)) of 
`length(enc.first)` fermionic modes back to a Majorana encoding returned as `res`, acting 
as the inverse transformation of [`toDiracEnc`](@ref): 

    γ_{2p-1} = (a_p)' + a_p,    γ_{2p} = im * ((a_p)' - a_p), 

where `a_p = enc.first[begin+p-1]`, `γ_{2p-1} = res.first[begin+p-1]`, and `γ_{2p} = 
res.second[begin+p-1]`. In the first two method signatures, the first argument selects the 
output representation:

- `::Type{PauliStr}`: each recovered Majorana operator must be representable as a single 
  `PauliStr` with a coefficient of exactly `1` or `-1`; equivalently, `enc` must satisfy 
  `checkDiracEnc(enc; strRestricted=true)`.
- `::Type{PauliSum}`: applicable to every valid Dirac encoding. Each recovered Majorana 
  operator is stored as a (Hermitian) `PauliSum`, and the result satisfies the 
  `PauliSum`-based method of [`checkMajoranaEnc`](@ref).

The third method signature falls back to the first method signature.

In both representations, the result does not reference any data in `enc`. The 
`PauliStr`-based conversion and `toDiracEnc` are mutually inverse: 

```jldoctest
julia> mEnc = genBravyiKitaevEnc(3);

julia> dEnc = toDiracEnc(mEnc);

julia> toMajoranaEnc(dEnc) == mEnc
true

julia> toDiracEnc(toMajoranaEnc(dEnc)) == dEnc
true
```
"""
function toMajoranaEnc(::Type{PauliStr}, enc::PairwiseSumEnc)::PairwiseStrEnc
    checkDiracEnc(enc, true, strRestricted=true)
    nMode = length(enc.first)

    oddOps = Vector{PauliStr}(undef, nMode)
    eveOps = Vector{PauliStr}(undef, nMode)

    for p in 1:nMode
        gOdd, gEve = toMajoranaPair(enc.first[begin+p-1]=>enc.second[begin+p-1], false)
        oddOps[begin+p-1] = toPauliStr(gOdd)
        eveOps[begin+p-1] = toPauliStr(gEve)
    end

    oddOps => eveOps
end

function toMajoranaEnc(::Type{PauliSum}, enc::PairwiseSumEnc)::PairwiseSumEnc
    checkDiracEnc(enc, true, strRestricted=false)

    pairs = map(enc.first, enc.second) do annOp, creOp
        toMajoranaPair(annOp=>creOp, false)
    end

    map(first, pairs) => map(last, pairs)
end

toMajoranaEnc(enc::PairwiseSumEnc) = toMajoranaEnc(PauliStr, enc)


"""
    buildMajoranaFrame(enc::Pair{<:AbstractVector{<:PauliSum}, 
                                 <:AbstractVector{<:PauliSum}}) -> 
    Tuple{Pair{Matrix{T}, Matrix{T}}, Vector{PauliStr}} where {T<:Real}

Given a valid `PauliSum`-based Majorana encoding `enc` (verified via the matching method 
of [`checkMajoranaEnc`](@ref)) of `nMode = length(enc.first)` fermionic modes, attempt to 
rebuild its underlying `PauliStr`-based Majorana frame: `2nMode` single Pauli strings 
`frame` (returned in ascending order, all carrying the positive-real phase) together with 
a pair of (real) transformation matrices `matAnn => matCre`, each of size `2nMode` by 
`nMode`, such that 

    PauliSum(enc.first[ begin+i-1]) == PauliSum(frame, matAnn[:, i]), 
    PauliSum(enc.second[begin+i-1]) == PauliSum(frame, matCre[:, i]).

In other words, the return value is `(matAnn=>matCre, frame)`. Furthermore, `frame` exists 
if and only if the number of distinct Pauli strings appearing across all operators in `enc` 
equals `2nMode`, in which case `hcat(matAnn, matCre)` is exactly orthogonal and its columns 
list the odd-indexed Majorana operators followed by the even-indexed ones (a sorted-block 
ordering rather than the interleaved index ordering). When no such frame exists, the 
returned matrices are 0-by-0 and `frame` is empty. The returned `frame` carries no mode 
pairing or role assignment beyond its ascending string order. The result does not reference 
any data in `enc`.
"""
function buildMajoranaFrame(enc::PairwiseSumEnc)
    checkMajoranaEnc(enc, true)
    nMode = length(enc.first)
    nMajs = 2 * nMode
    tMatT = mapreduce(op->real(eltype(op.coeff)), promote_type, Iterators.flatten(enc))
    frameSet = Set{PauliStr}()
    annOpVec = map(PauliSum, enc.first)  #> Canonicalize encoding operators to make sure 
    creOpVec = map(PauliSum, enc.second) #> each string carries a `posRea` phase
    for ops in (annOpVec, creOpVec), op in ops, str in op.str
        push!(frameSet, str)
    end

    #> The frame is well defined iff exactly `nMajs` distinct strings appear: the trace-
    #> orthonormality of a valid encoding forces at least `nMajs`, and with exactly `nMajs` 
    #> the (square) coefficient matrix is orthogonal.
    if length(frameSet) != nMajs
        return (Matrix{tMatT}(undef, 0, 0)=>Matrix{tMatT}(undef, 0, 0), PauliStr[])
    end

    frame = collect(frameSet)

    #> Defensive check: distinct strings inside the frame of a validated encoding must 
    #> anticommute; reachable solely through broken upstream invariants
    for i in 1:nMajs, j in (i+1):nMajs
        if !checkAntiCom(frame[begin+i-1], frame[begin+j-1])
            return (Matrix{tMatT}(undef, 0, 0)=>Matrix{tMatT}(undef, 0, 0), PauliStr[])
        end
    end

    sort!(frame)
    matAnn = zeros(tMatT, nMajs, nMode)
    matCre = zeros(tMatT, nMajs, nMode)
    for (ops, mat) in ((annOpVec, matAnn), (creOpVec, matCre)), (iCol, op) in enumerate(ops)
        for (str, coeff) in zip(op.str, op.coeff)
            iRow = searchsortedfirst(frame, str)
            mat[iRow, begin+iCol-1] = real(coeff) #> The imaginary part should be zero
        end
    end

    (matAnn=>matCre, frame)
end
