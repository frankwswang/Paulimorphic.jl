export PauliStr, @pauli_str, PauliSum, canonicalize!, curtail, sanitize!, shift!

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
             phase::PhaseFactor, nSite=8*sizeof(UInt)*length(xWords)) -> PauliStr

Construct a `PauliStr` on `nSite` sites from the bit-packed components, `xWords` and 
`zWords`. The layouts of `xWords` and `zWords` should follow the layout for fields `.x` and 
`.z` of `PauliStr`, respectively. `phase::`[`PhaseFactor`](@ref) assigns the 
value of field `.phase`. `nSite` assigns the value of field `.n`, which in default is to 
set to the inputs' full capacity, `8 * sizeof(UInt) * length(xWords)`. When `nSite` is less 
than the full capacity, only the leading  `cld(nSite, 8 * sizeof(UInt))` words are taken; 
any surplus words from the input are ignored, and bits beyond site `nSite` in the final 
word are cleared rather than validated. The components are copied into freshly allocated 
buffers, so the returned `PauliStr` does not reference `xWords` or `zWords`.

    PauliStr(list::AbstractVector{PauliSym}, phase::PhaseFactor=$posRea) -> PauliStr

Build a `PauliStr` from a per-site list of Pauli symbols, one [`PauliSym`](@ref) (`symI`, 
`symX`, `symY`, `symZ`) per site, with `list[i]` acting on site `i`. The resulting string 
spans `length(list)` sites. `phase::`[`PhaseFactor`](@ref) determines the four optional 
phase attached to the Pauli string as `im^Int(phase)`, which in default is `+1`.

    PauliStr(nSite::Integer=0, siteOp::PauliSym=symI, phase::PhaseFactor=PhaseFactor(0))

Build a uniform `PauliStr` on `nSite` sites in which every site carries the same 
single-site Pauli `siteOp` (one [`PauliSym`](@ref): `symI`, `symX`, `symY`, or `symZ`). 
`nSite` must be non-negative (an `ArgumentError` is thrown otherwise). 
`phase::`[`PhaseFactor`](@ref) determines the four optional phase attached to the Pauli 
string as `im^Int(phase)`, which in default is `+1`.
"""
mutable struct PauliStr <: DiscreteOperator
    const x::Memory{UInt}
    const z::Memory{UInt}
    phase  ::PhaseFactor
    const n::Int

    function PauliStr(nSite::Integer=0, siteOp::PauliSym=symI, phase::PhaseFactor=posRea)
        nSite < 0 && throw(ArgumentError("`nSite` must be non-negative."))
        nBitPerWord = 8 * sizeof(UInt)
        nWord = cld(nSite, nBitPerWord)
        allOneBits = ~zero(UInt) #> Same as `typemax(UInt)`
        xEle = (siteOp == symX || siteOp == symY) ? allOneBits : zero(UInt)
        zEle = (siteOp == symZ || siteOp == symY) ? allOneBits : zero(UInt)
        xStr = Memory{UInt}(undef, nWord); fill!(xStr, xEle)
        zStr = Memory{UInt}(undef, nWord); fill!(zStr, zEle)

        #> Enforce resetting the padding bits to be zero
        new(xStr, zStr, phase, Int(nSite)) |> sanitize!
    end

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
        new(xStr, zStr, phase, nSite) |> sanitize!
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
             simplification::Bool=true) where {T<:Real, C<:Union{Complex{T}, T}} -> 
    PauliSum{T}

Construct a `PauliSum{T}` with `T = real(C)` from `coeffs` and `strs` of equal length. The 
strings are deep-copied, and each string's phase is absorbed into its matching coefficient. 
When `simplification=true` (by default), equal strings are combined into one term and any 
term whose coefficients sum to exactly zero is removed; when `simplification=false`, 
duplicate strings are retained. In both cases the terms in the constructed `res::PauliSum` 
are stored in a deterministic canonical order such that for 
`res2=`[`canonicalize!`](@ref)`(deepcopy(res))`, 

    res2.coeff == res.coeff && res2.str == res.str

always returns `true`.

    PauliSum(::Type{T}, strs::AbstractVector{PauliStr}, 
             simplification::Bool=true) where {T<:Real} -> PauliSum{T}

Construct a `PauliSum` with the coefficient of every Pauli string being `one(Complex{T})`.

    PauliSum(strs::AbstractVector{PauliStr}=PauliStr[], 
             simplification::Bool=true) where {T<:Real} -> PauliSum{T}

Shorthand for `PauliSum(Int8, strs, simplification)`.

    PauliSum(selector::F, byCoeff::Bool, ham::PauliSum{T}) where {F, T<:Real} -> PauliSum{T}

Low-level constructor that builds a `PauliSum` from the subset of `ham`'s terms picked out 
by `selector`. The predicate is applied through `Base.findall` to `ham.coeff` when 
`byCoeff=true`, or to `ham.str` when `byCoeff=false`, and every term at a matching index is 
kept. Accordingly, `selector` must accept a `Complex{T}` (when `byCoeff=true`) or a 
`PauliStr` (when `byCoeff=false`) in the sole input and return a `Bool`. This constructor 
respects the term order of `ham` and performs no phase absorption, merging, or re-sorting.

Unlike other constructor methods, the retained strings are NOT reconstructed (deep-copied). 
In other words, the returned sum's `.str` entries are the same (subsets of) `PauliStr` held 
by `ham.str`.
"""
struct PauliSum{T<:Real} <: DiscreteOperator
    coeff::Memory{Complex{T}}
    str::Memory{PauliStr}

    function PauliSum(selector::F, byCoeff::Bool, ham::PauliSum{T}) where {F, T<:Real}
        coeffs = ham.coeff
        strs = ham.str
        indices = findall(selector, (byCoeff ? coeffs : strs))
        new{T}(coeffs[indices], strs[indices])
    end

    function PauliSum(coeffs::AbstractVector{C}, 
                      strs::AbstractVector{PauliStr}, 
                      simplification::Bool=true) where {C<:RealOrComplex}
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

        if !simplification || iszero(inputSize)
            c = cInput
            s = sInput
            sortStrings!(c, s, true)
        else
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
        end

        new{T}(c, s)
    end
end

function PauliSum(::Type{T}, str::AbstractVector{PauliStr}, simplification::Bool=true
                  ) where {T<:Real}
    coeff = Memory{Complex{T}}(undef, length(str))
    coeff .= one(Complex{T})
    PauliSum(coeff, str, simplification)
end

PauliSum(str::AbstractVector{PauliStr}=PauliStr[], simplification::Bool=true) = 
PauliSum(Int8, str, simplification)

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

    if nTerm > 0
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
        #> `scope` is shifted when `weights` & `strs` are not one-based indexed
        iFirst1 = firstindex(strs)
        iFirst2 = firstindex(weights)
        iFirst1 == 1 || (scope .+= iFirst1 - 1) #> Elements of `scope` are one-based indices
        strs .= strs[scope]
        iFirst2 == iFirst1 || (scope .+= iFirst2 - iFirst1)
        weights .= weights[scope]
    end

    nothing
end


"""
    sanitize!(str::PauliStr) -> PauliStr

Restore the [`PauliStr`](@ref) invariant that every bit after site `str.n` is zero, and 
then return the mutated `str`. Only the final word of each buffer can hold such surplus 
bits, which is subject to the sanitization. When `str.n` is an integer multiple of the word 
size, the final word holds no padding and the call is a no-op.
"""
function sanitize!(str::PauliStr)
    nBitPerWord = 8 * sizeof(UInt)
    remSites = str.n & (nBitPerWord - 1) #>> More efficient than `str.n % nBitPerWord`
    if !iszero(remSites)
        maskStr = (one(UInt) << remSites) - one(UInt) #>> Last `remSites` bits are one
        str.x[end] &= maskStr
        str.z[end] &= maskStr
    end

    str
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


"""
    curtail(ham::PauliSum{T}, tolerance::Real=(T<:Integer ? zero(T) : 8eps(T)); 
            relative::Bool=false, inclusive::Bool=false) where {T<:Real} -> 
    PauliSum{T}

Return a filtered shallow copy of `ham` with negligible terms removed, retaining only the 
terms whose coefficient exceeds a threshold. A term with coefficient `c` is kept when

    abs(c) >  tolerance * scale    (if inclusive == false)
    abs(c) >= tolerance * scale    (if inclusive == true )

where `scale` is `one(T)` when `relative=false` (default) and `maximum(abs, ham.coeff)` 
when `relative=true`. Thus, `tolerance` acts as an absolute magnitude floor by default, or 
as a fraction of the largest coefficient when `relative` is set. The `inclusive` keyword 
chooses whether coefficients landing exactly on the threshold are kept (`>=`) or dropped 
(`>`). For floating-point `T`, the default value of `tolerance` is `8eps(T)`: a small 
multiple of the machine epsilon, enough to discard the round-off residuals left by operator 
assembly when the coefficients are well conditioned; for integer `T`, the default value of 
`tolerance` is `zero(T)`. A negative `tolerance` throws an `ArgumentError`.

# String ownership
The retained terms remain in `ham`'s order, and the returned sum's coefficients are held in 
an independent buffer. Its `PauliStr`s, however, are NOT reconstructed (deep-copied). In 
other words, the returned sum's `.str` entries are the same (subsets of) `PauliStr` held 
by `ham.str`.
"""
function curtail(ham::PauliSum{T}, tolerance::Real=(T<:Integer ? zero(T) : 8eps(T)); 
                 relative::Bool=false, inclusive::Bool=false) where {T<:Real}
    tolerance < 0 && throw(ArgumentError("`tolerance` must be non-negative."))

    coeffs = ham.coeff
    scale = relative ? (isempty(coeffs) ? one(T) : maximum(abs, coeffs)) : one(T)
    threshold = tolerance * scale

    cap = if inclusive
        (c::Complex{T}) -> abs(c) >= threshold
    else
        (c::Complex{T}) -> abs(c) >  threshold
    end

    PauliSum(cap, true, ham)
end


"""
    shiftBits!(words::AbstractVector{T}, wordOffset::Int, bitOffset::Int, 
               forward::Bool=true) where {T<:$BitUInteger} -> 
    typeof(words)

Shift the bits packed in `words` in place by a total of `wordOffset` whole words plus
`bitOffset` bits, treating the buffer as one contiguous bit string in which the
least-significant bit of the lowest-indexed word comes first. The mutated `words` is
returned. The element type `T<:$BitUInteger` determines the word width at `8*sizeof(T)` 
bits.

The optional argument `forward` selects the direction of the shift:
- `forward = true` moves bits toward the most-significant end. The in-word shift (specified 
  by `bitOffset`) direction is from the least significant bit (LSB) to the most significant 
  bit (MSB) (e.g., a 2-bit shift sends `0b0011` to `0b1100`); the shift across words move 
  from the lower index to higher index (e.g., `[w1, w2, _]` becomes `[_, w1, w2]`).
- `forward = false` is the exact reverse, moving bits toward the least-significant end.

Bits pushed past either boundary are dropped and vacated positions are filled with zeros, 
so `length(words)` is preserved. `wordOffset` must be non-negative and `bitOffset` must 
satisfy `0 <= bitOffset < 8*sizeof(T)`, or a `DomainError` is thrown.
"""
function shiftBits!(words::AbstractVector{T}, wordOffset::Int, bitOffset::Int, forward::Bool
                    ) where {T<:BitUInteger}
    wordOffset < 0 && throw(DomainError(wordOffset, "`wordOffset` must be non-negative."))
    nBitPerWord = 8 * sizeof(T)
    if bitOffset >= nBitPerWord || bitOffset < 0
        throw(DomainError(bitOffset, "`0 <= bitOffset < $nBitPerWord` must hold true."))
    end

    comOffset = nBitPerWord - bitOffset #> Complementary shift for the inter-word carry
    nWord = length(words)
    carryOffset = 1
    rng = if forward
           comOffset *= -1
           bitOffset *= -1
          wordOffset *= -1
        carryOffset *= -1
        nWord:(-1):1
    else
        1:( 1):nWord
    end

    #> E.g., `nBitPerWord = 5; bitOffset = 2`
    #>> `forward` |`head`        |`tail`
    #>> `true `   | XXYYY->YYY00 | YYXXX -> 000YY
    #>> `false`   | YYYXX->00YYY | XXXYY -> YY000
    @inbounds for i in rng
        sourcePos = i + wordOffset
        head = (1 <= sourcePos <= nWord) ? (words[begin+sourcePos-1] >> bitOffset) : zero(T)
        tail = if !iszero(bitOffset) && 1 <= (sourcePos + carryOffset) <= nWord
            words[begin+(sourcePos + carryOffset)-1] << comOffset
        else
            zero(T)
        end

        words[begin+i-1] = head | tail
    end

    words
end


"""
    shift!(str::PauliStr, n::Integer, toHigher::Bool=false) -> PauliStr

Shift the single-site operators in `str` along the site axis by `n` sites, in place,
and return the mutated `str`. When `toHigher=true` (default) the operators move toward 
higher site indices (i.e., a *right* shift): the operator on site `k` is moved to site 
`k + n`, and anything pushed above site `str.n` is discarded. Similarly, when 
`toHigher=false` the operators move toward lower site indices (a *left* shift). Vacated 
sites are refilled with the identity, the site count `str.n` is preserved, and the overall 
`.phase` is left unchanged.

# Example
```julia
julia> shift!(pauli"IIXXII", 3, false)   #> left  shift: site 4 → site 1, site 3 dropped
+X₁

julia> shift!(pauli"IIXXII", 3, true)    #> right shift: site 3 → site 6, site 4 dropped
+X₆
```
"""
function shift!(str::PauliStr, n::Integer, toHigher::Bool=true)
    n < 0 && throw(ArgumentError("`n` must be non-negative."))
    (iszero(n) || iszero(str.n)) && return str #> Nothing to shift

    nBitPerWord = 8 * sizeof(UInt)
    wordOffset, bitOffset = divrem(n, nBitPerWord)
    shiftBits!(str.x, wordOffset, bitOffset, toHigher)
    shiftBits!(str.z, wordOffset, bitOffset, toHigher)

    sanitize!(str) #> Clear any bits moved into the padding region past site `str.n`
end