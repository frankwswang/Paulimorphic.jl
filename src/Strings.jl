export PauliStr, @pauli_str, PauliSum, canonicalize!

using LinearAlgebra: dot

"""
    PauliStr <: DiscreteOperator

An element of the `n`-site Pauli group, stored in the symplectic `z`-`x`
representation: two bit-packed `Memory{UInt}` buffers plus an overall phase.

Specifically, each element is represented by 

    phase * (P_1 ⊗ P_2 ⊗ … ⊗ P_n),

where `P_k` is a single-site Pauli encoded as a pair of bits specified by the `k`-th 
bits of the `z` and `x` (each as one contiguous bitstring), 

    (z_k, x_k):  (0, 0) => I,  (1, 0) => Z,  (0, 1) => X,  (1, 1) => Y;

and `phase::`[`PhaseFactor`](@ref) takes one of the values in `(+1, +im, -1, -im)`. 
Carrying the phase coefficient keeps `PauliStr` a member of the Pauli group under 
multiplication. The phase generated when two `PauliStr`s are multiplied is absorbed into 
the result's `.phase`.

# Bit layout
With `8 * sizeof(UInt)` bits per word (entry of bit-packed `Memory{UInt}`), the 
`k`-th (1-based) site of every `PauliStr` occupies the `(k - 1) % W`-th bit of word 
`cld(k, W)` in each buffer, least-significant bit first (i.e., right-to-left in each word, 
from `i`-th word to `i+1`-th word). Bits past site `n` in the final word are zero by 
construction.

# Fields
- `.x::Memory{UInt}`: X-component — bit `k` is set iff `P_k ∈ {X, Y}`.
- `.z::Memory{UInt}`: Z-component — bit `k` is set iff `P_k ∈ {Z, Y}`.
- `.phase::PhaseFactor`: phase information such that `im^Int(.phase)` returns one of the 
  four overall phase coefficients.
- `n::Int`: number of sites such that `length(x) == length(z) == cld(n, 8sizeof(UInt))`.

The fields `.x` and `.z` are owned by `PauliStr` instances and never reassigned, though 
their entries along with `.phase` may be mutated in place. NOTE: Every bit above site `n` 
in the final word should always be set to zero.

≡≡≡ Initialization Method(s) ≡≡≡

    PauliStr(xWords::AbstractVector{UInt}, zWords::AbstractVector{UInt}, 
             phase::PhaseFactor, nSite=8*sizeof(UInt)*length(xWords))

Construct a `PauliStr` on `nSite` sites from the bit-packed components, `xWords` and 
`zWords`. The layouts of `xWords` and `zWords` should follow the layout for fields `.x` and 
`.z` of `PauliStr`, respectively. `phase::`[`PhaseFactor`](@ref) assigns the 
value of field `.phase`. `nSite` assigns the value of field `.n`, which in default is to 
set to the inputs' full capacity, `8 * sizeof(UInt) * length(xWords)`. When `nSite` is less 
than the full capacity, only the leading  `cld(nSite, 8 * sizeof(UInt))` words are taken; 
any surplus words from the input are ignored, and bits beyond site `nSite` in the final 
word are cleared rather than validated. The components are copied into freshly allocated 
buffers, so the returned `PauliStr` does not reference `xWords` or `zWords`.

    PauliStr(list::AbstractVector{PauliSym}, phase::PhaseFactor=$posRea)

Build a `PauliStr` from a per-site list of Pauli symbols, one [`PauliSym`](@ref) (`symI`, 
`symX`, `symY`, `symZ`) per site, with `list[i]` acting on site `i`. The resulting string 
spans `length(list)` sites. `phase::`[`PhaseFactor`](@ref) determines the four optional 
phase attached to the Pauli string as `im^Int(phase)`, which in default is `+1`.
"""
mutable struct PauliStr <: DiscreteOperator
    const x::Memory{UInt}
    const z::Memory{UInt}
    phase  ::PhaseFactor
    const n::Int

    PauliStr(pStr::PauliStr, phase::PhaseFactor=pStr.phase) = 
    new(copy(pStr.x), copy(pStr.z), phase, pStr.n)

    function PauliStr(xWords::AbstractVector{UInt}, zWords::AbstractVector{UInt}, 
                      phase::PhaseFactor, nSite::Int=8*sizeof(UInt)*length(xWords))
        maxWordNum = length(xWords)

        if maxWordNum != length(zWords)
            throw(ArgumentError("`xWords` and `zWords` must have the same length."))
        end

        nBitPerWord = 8 * sizeof(UInt)
        maxSiteNum = nBitPerWord * maxWordNum
        if !(0 <= nSite <= maxSiteNum) #> `0` => empty string => identity
            throw(DomainError(nSite, "`nSite` must be in 0:$maxSiteNum."))
        end

        #> Ensure string fields are not linked to inputs
        nWord = cld(nSite, nBitPerWord)
        xStr = Memory{UInt}(undef, nWord)
        zStr = Memory{UInt}(undef, nWord)
        copyto!(xStr, firstindex(xStr), xWords, firstindex(xWords), nWord)
        copyto!(zStr, firstindex(zStr), zWords, firstindex(zWords), nWord)

        #> Enforce resetting the padding bits to be zero
        remSites = nSite & (nBitPerWord - 1) #>> More efficient than `nSite % nBitPerWord`
        if !iszero(remSites)
            maskStr = (one(UInt) << remSites) - one(UInt) #>> Last `remSites` bits are one
            xStr[end] &= maskStr
            zStr[end] &= maskStr
        end

        new(xStr, zStr, phase, nSite)
    end

    function PauliStr(list::AbstractVector{PauliSym}, phase::PhaseFactor=posRea)
        nSite = length(list) #> `0` => empty string => identity

        bitsPerWord = 8 * sizeof(UInt)
        nWord = cld(nSite, bitsPerWord)
        xStr = Memory{UInt}(undef, nWord); fill!(xStr, zero(UInt))
        zStr = Memory{UInt}(undef, nWord); fill!(zStr, zero(UInt))

        for i in 1:nSite
            ele = list[firstindex(list)+i-1]

            isZ, isX = if ele == symI
                false, false
            elseif ele == symZ
                true, false
            elseif ele == symX
                false,  true
            else #> symY
                true,  true
            end

            bitPos  = i - 1
            wordIdx = (bitPos ÷ bitsPerWord) + 1
            bitMask = one(UInt) << (bitPos % bitsPerWord)
            isX && (xStr[wordIdx] |= bitMask)
            isZ && (zStr[wordIdx] |= bitMask)
        end

        new(xStr, zStr, phase, nSite)
    end
end

PauliStr(siteNum::Integer=0, siteOp::PauliSym=symI) = PauliStr(fill(siteOp, siteNum))


function Base.hash(pStr::PauliStr, hashCode::UInt)
    code = hash(pStr.phase, hashCode)
    code = hash(pStr.n, code)
    code = hash(pStr.x, code)
    hash(pStr.z, code)
end

function Base.:(==)(pStr1::PauliStr, pStr2::PauliStr)
    isequal(pStr1.phase, pStr2.phase) && 
    isequal(pStr1.n    , pStr2.n    ) && 
    isequal(pStr1.x    , pStr2.x    ) && 
    isequal(pStr1.z    , pStr2.z    )
end

function Base.isless(a::PauliStr, b::PauliStr)
    a.n != b.n && return (a.n < b.n)

    @inbounds for i in 1:length(a.x)
        a.x[begin+i-1] != b.x[begin+i-1] && return (a.x[begin+i-1] < b.x[begin+i-1])
    end

    @inbounds for i in 1:length(a.z)
        a.z[begin+i-1] != b.z[begin+i-1] && return (a.z[begin+i-1] < b.z[begin+i-1])
    end

    UInt8(a.phase) < UInt8(b.phase)
end


function getPauliSymVec(str::Union{String, AbstractVector{<:Integer}})
    [getPauliSym(c::Union{Char, Integer}) for c in str]
end


"""
    pauli"..."

A custom string literal that builds a [`PauliStr`](@ref) from single-site Pauli symbols, 
one character per site (e.g. `'I'`, `'X'`, `'Y'`, `'Z'`), with the `i`-th character acting 
on site `i`. The resulting `PauliStr` is assigned the unit phase `posRea` (`+1`). 

# Example
```julia
pauli"IXYZ"
```
"""
macro pauli_str(ex)
    strOrVecExpr = esc(ex)
    :(getPauliSymVec($strOrVecExpr) |> PauliStr)
end

function phaseStr(p::PhaseFactor)
    if p === posRea
        "+"
    elseif p === posImg
        "+im*"
    elseif p === negRea
        "-"
    else
        "-im*"
    end
end


function printOperator(pStr::PauliStr)
    nBitPerWord = 8 * sizeof(UInt)
    body = ""

    for i in 1:pStr.n
        bitPos = i - 1
        w    = (bitPos ÷ nBitPerWord) + 1
        mask = one(UInt) << (bitPos % nBitPerWord)
        z = !iszero(pStr.z[w] & mask)
        x = !iszero(pStr.x[w] & mask)

        body *= if z & x
            'Y' * CONSTVAR!!subscriptNum[i]
        elseif z
            'Z' * CONSTVAR!!subscriptNum[i]
        elseif x
            'X' * CONSTVAR!!subscriptNum[i]
        else
            ""
        end
    end

     isempty(body) ? "I" : (phaseStr(pStr.phase) * body)
end

Base.show(io::IO, op::PauliStr) = print(io, printOperator(op))


"""
    PauliSum{T<:Real} <: DiscreteOperator

A linear combination of Pauli strings, i.e. a general operator

    ∑_k coeff_k * str_k,

stored as two parallel buffers: the coefficients `.coeff::Memory{Complex{T}}` and their
associated `PauliStr` `.str::Memory{PauliStr}`. 

# Fields
- `.coeff::Memory{Complex{T}}`: the coefficients associated with terms of Pauli strings.
- `.str::Memory{PauliStr}`: the Pauli strings with `length(.str) == length(.coeff)`.

≡≡≡ Initialization Method(s) ≡≡≡

    PauliSum(coeffs::AbstractVector{C}, strs::AbstractVector{PauliStr}, 
             mergeRedundancy::Bool=true) where {C<:Union{Real, Complex}}

Construct a `PauliSum{T}` with `T = real(C)` from `coeffs` and `strs` of equal length. The 
strings are deep-copied, and each string's phase is absorbed into its matching coefficient. 
When `mergeRedundancy=true` (by default), equal strings are combined into one term and any 
term whose coefficients sum to exactly zero is removed; when `mergeRedundancy=false`, 
duplicate strings are retained. In both cases the terms in the constructed `res::PauliSum` 
are stored in a deterministic canonical order such that for 
`res2=`[`canonicalize!`](@ref)`(deepcopy(res))`, 
    
    res2.coeff == res.coeff && res2.str == res.str

always returns `true`.

    PauliSum(::Type{T}, strs::AbstractVector{PauliStr}, 
             mergeRedundancy::Bool=true) where {T<:Real}

Construct a `PauliSum` with the coefficient of every Pauli string being `one(Complex{T})`.

    PauliSum(strs::AbstractVector{PauliStr}=PauliStr[])

Shorthand for `PauliSum(Int, strs)`.
"""
struct PauliSum{T<:Real} <: DiscreteOperator
    coeff::Memory{Complex{T}}
    str::Memory{PauliStr}

    function PauliSum(coeffs::AbstractVector{C}, 
                      strs::AbstractVector{PauliStr}, 
                      mergeRedundancy::Bool=true) where {C<:RealOrComplex}
        T = real(C)
        inputSize = length(coeffs)

        if inputSize != length(strs)
            throw(ArgumentError("`coeffs` and `strs` should have the same length."))
        end

        cInput = Memory{Complex{T}}(undef, inputSize)
        sInput = Memory{PauliStr}(undef, inputSize)
        copyto!(cInput, firstindex(cInput), coeffs, firstindex(coeffs), inputSize)
        for i in 1:inputSize; sInput[begin+i-1] = PauliStr(strs[begin+i-1]) end
        absorbPhases!(cInput, sInput)

        if mergeRedundancy
            perm = sortperm(sInput)

            #> Merge equal strings into buffers with upper-bound size, then trim once
            cBuffer = Memory{Complex{T}}(undef, inputSize)
            sBuffer = Memory{PauliStr}(undef, inputSize)
            mergedSize = 0
            k = 1

            @inbounds while k <= inputSize
                p = perm[begin+k-1]
                str = sInput[p]
                acc = cInput[p]

                k += 1
                while k <= inputSize && sInput[perm[begin+k-1]] == str
                    acc += cInput[perm[begin+k-1]]
                    k += 1
                end

                if !iszero(acc) #>> Drop terms with coefficients exactly equal zero
                    mergedSize += 1
                    sBuffer[begin+mergedSize-1] = str
                    cBuffer[begin+mergedSize-1] = acc
                end
            end

            if mergedSize == inputSize #>> No duplicates and no cancellation
                c = cBuffer
                s = sBuffer
            else
                c = Memory{Complex{T}}(undef, mergedSize)
                s = Memory{PauliStr}(undef, mergedSize)
                copyto!(c, firstindex(c), cBuffer, firstindex(cBuffer), mergedSize)
                copyto!(s, firstindex(s), sBuffer, firstindex(sBuffer), mergedSize)
            end
        else
            c = cInput
            s = sInput
            sortStrings!(c, s, true)
        end

        new{T}(c, s)
    end
end

function PauliSum(::Type{T}, str::AbstractVector{PauliStr}, mergeRedundancy::Bool=true
                  ) where {T<:Real}
    coeff = Memory{Complex{T}}(undef, length(str))
    coeff .= one(Complex{T})
    PauliSum(coeff, str, mergeRedundancy)
end

PauliSum(str::AbstractVector{PauliStr}=PauliStr[]) = PauliSum(Int, str)

function Base.hash(pSum::PauliSum, hashCode::UInt)
    code = hash(pSum.str, hashCode)
    hash(pSum.coeff, code)
end

function Base.:(==)(pSum1::PauliSum, pSum2::PauliSum)
    (pSum1.coeff == pSum2.coeff) && (pSum1.str == pSum2.str)
end


"""
    absorbPhases!(storage::AbstractVector{C}, 
                  strs::AbstractVector{PauliStr}) where {C<:Complex} -> Nothing

Absorb the phase of each Pauli string in `strs` into a parallel vectorized `storage`, in 
place. Specifically, for every index `i` whose `strs[i].phase` is not equal to one(C), 
multiply `storage[i]` by [`evalPhase`](@ref)`(strs[i].phase)` and reset `strs[i].phase` to 
`posRea`. Both `storage` and `strs` are mutated and must share the same length (an 
`ArgumentError` is thrown otherwise).
"""
function absorbPhases!(storage::AbstractVector{C}, 
                       strs::AbstractVector{PauliStr}) where {C<:Complex}
    nTerm = length(strs)
    if nTerm != length(storage)
        throw(ArgumentError("`storage` and `strs` should have the same length."))
    end

    for (i, str) in enumerate(strs)
        phase = evalPhase(str.phase)
        if !isone(phase)
            str.phase = posRea #>> Reset the phase to be one
            storage[begin+i-1] *= phase
        end
    end

    nothing
end


"""
    sortStrings!(weights::AbstractVector{C}, strs::AbstractVector{PauliStr}, 
                 considerWeight::Bool=true) where {C<:Union{Real, Complex}} -> Nothing

Sort `strs` into ascending order and apply the same permutation to the parallel `weights`, 
in place. When `considerWeight=true` (default), ties among equal strings are broken by 
`(abs(w), real(w), imag(w))` of the corresponding weight, yielding a total order even when 
duplicate strings are present; otherwise, the strings alone form the sort key ordered by 
`isless(::PauliStr, ::PauliStr)`. Both vector arguments are mutated and must share the same 
length (an`ArgumentError` is thrown otherwise).
"""
function sortStrings!(weights::AbstractVector{<:RealOrComplex}, 
                      strs::AbstractVector{PauliStr}, considerWeight::Bool=true)
    nTerm = length(strs)
    if nTerm != length(weights)
        throw(ArgumentError("`weights` and `strs` should have the same length."))
    end

    #> Sort the indices of the Pauli strings
    scope = collect(1:nTerm)
    sortFunc = if considerWeight
        function (i)
            coeff = weights[begin+i-1]
            (strs[begin+i-1], abs(coeff), real(coeff), imag(coeff))
        end
    else
        i -> strs[begin+i-1]
    end
    sort!(scope, by=sortFunc)

    #> Update elements in `weights` and `strs` using sorted `scope`
    #> The shifting of `scope` is for cases when `weights` & `strs` are not one-based indexed
    iFirst1 = firstindex(strs)
    iFirst2 = firstindex(weights)
    iFirst1 == 1 || (scope .+= iFirst1 - 1) #> Elements of `scope` are one-based indices
    strs .= strs[scope]
    iFirst2 == iFirst1 || (scope .+= iFirst2 - iFirst1)
    weights .= weights[scope]

    nothing
end


"""
    canonicalize!(ham::PauliSum) -> PauliSum

Rewrite `ham` into a canonical form in place and return it: absorb every string's phase into
its coefficient based on [`absorbPhases!`](@ref), then sort the Pauli terms into a 
deterministic total order based on [`sortStrings!`](@ref).

This function preserves the term count. In other words, it does **not** merge duplicate 
strings or drop zero coefficients. To obtain a (unlinked) merged form, rebuild the sum via 
`PauliSum(ham.coeff, ham.str, true)`.
"""
function canonicalize!(ham::PauliSum)
    coeffs = ham.coeff
    strs = ham.str
    absorbPhases!(coeffs, strs)
    sortStrings!(coeffs, strs)
    ham
end


struct PauliList <: DiscreteOperator
    str::Memory{PauliStr}
    function PauliList(strs::AbstractVector{PauliStr}, 
                      mergeRedundancy::Bool=true)
        inputSize = length(strs)
        sInput = Memory{PauliStr}(undef, inputSize)
        for i in 1:inputSize; sInput[begin+i-1] = PauliStr(strs[begin+i-1], posRea) end
        if mergeRedundancy
            perm = sortperm(sInput)
            #> Merge equal strings into buffers with upper-bound size, then trim once
            sBuffer = Memory{PauliStr}(undef, inputSize)
            mergedSize = 0
            k = 1
            @inbounds while k <= inputSize
                p = perm[begin+k-1]
                str = sInput[p]
                k += 1
                while k <= inputSize && sInput[perm[begin+k-1]] == str
                    k += 1
                end
                mergedSize += 1
                sBuffer[begin+mergedSize-1] = str
            end
            if mergedSize == inputSize #>> No duplicates
                s = sBuffer
            else
                s = Memory{PauliStr}(undef, mergedSize)
                copyto!(s, firstindex(s), sBuffer, firstindex(sBuffer), mergedSize)
            end
        else
            s = sInput
            sort!(s)
        end
        new(s)
    end
end