export PauliStr, @pauli_str, PauliSum, countSites, canonicalize!, curtail, sanitize!, 
       shift!, paste!, stamp!, reframe

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
`.z` of `PauliStr`, respectively. `phase::`[`PhaseFactor`](@ref) assigns the value of field 
`.phase`. `nSite` assigns the value of field `.n`, which in default is set to the inputs' 
full capacity, `8 * sizeof(UInt) * length(xWords)`. `nSite` may fall on either side of that 
capacity: when it is smaller, only the leading `cld(nSite, 8 * sizeof(UInt))` words of each 
input are taken — any surplus words are ignored; when it is larger, every site past `nSite` 
carries the identity. The components are copied into freshly allocated buffers, so the 
returned `PauliStr` does not reference `xWords` or `zWords`.

    PauliStr(list::AbstractVector{PauliSym}, phase::PhaseFactor=$posRea) -> PauliStr

Build a `PauliStr` from a per-site list of Pauli symbols, one [`PauliSym`](@ref) (`symI`, 
`symX`, `symY`, `symZ`) per site, with `list[i]` acting on site `i`. The resulting string 
spans `length(list)` sites. `phase::`[`PhaseFactor`](@ref) determines the four optional 
phase attached to the Pauli string as `im^Int(phase)`, which in default is `+1`.

    PauliStr(nSite::Integer=0, siteOp::PauliSym=symI, phase::PhaseFactor=PhaseFactor(0))

Build a uniform `PauliStr` on `nSite` sites in which every site carries the same 
single-site Pauli `siteOp::`[`PauliSym`](@ref). `nSite` must be non-negative (an 
`DomainError` is thrown otherwise). `phase::`[`PhaseFactor`](@ref) determines the four 
optional phase attached to the Pauli string as `im^Int(phase)`, which in default is `+1`.

    PauliStr(pStr::PauliStr, nSite::Int=pStr.n, phase::PhaseFactor=pStr.phase) -> PauliStr

Construct a copy of `pStr` rebuilt to explicitly act on `nSite` sites and in default 
preserves the site count of `pStr`. When `nSite > pStr.n` every site above `pStr.n` carries 
the identity; when `nSite < pStr.n`, the sites above `nSite` are cropped away. `phase` 
assigns the value of field `.phase`. The returned `PauliStr` holds freshly allocated 
buffers and does not reference the data in `pStr`.
"""
mutable struct PauliStr <: DiscreteOperator
    const x::Memory{UInt}
    const z::Memory{UInt}
    phase  ::PhaseFactor
    const n::Int

    function PauliStr(nSite::Integer=0, siteOp::PauliSym=symI, phase::PhaseFactor=posRea)
        nSite < 0 && throw(DomainError(nSite, "`nSite` must be non-negative."))
        nSitePerWord = 8 * sizeof(UInt)
        nWord = cld(nSite, nSitePerWord)
        allOneBits = ~zero(UInt) #> Same as `typemax(UInt)`
        xEle = (siteOp == symX || siteOp == symY) ? allOneBits : zero(UInt)
        zEle = (siteOp == symZ || siteOp == symY) ? allOneBits : zero(UInt)
        xStr = Memory{UInt}(undef, nWord); fill!(xStr, xEle)
        zStr = Memory{UInt}(undef, nWord); fill!(zStr, zEle)

        #> Enforce resetting the padding bits to be zero
        new(xStr, zStr, phase, Int(nSite)) |> sanitize!
    end

    function PauliStr(xWords::AbstractVector{UInt}, zWords::AbstractVector{UInt}, 
                      phase::PhaseFactor, nSite::Int=8*sizeof(UInt)*length(xWords))
        nSite < 0 && throw(DomainError(nSite, "`nSite` must be non-negative."))
        maxWordNum = length(xWords)
        if maxWordNum != length(zWords)
            throw(ArgumentError("`xWords` and `zWords` must have the same length."))
        end
        #> Ensure string fields are not linked to inputs
        nWord = cld(nSite, 8sizeof(UInt))
        xStr = Memory{UInt}(undef, nWord)
        zStr = Memory{UInt}(undef, nWord)

        #> Zero only the words above the loaded prefix
        nWordLoad = min(nWord, maxWordNum)
        @inbounds for i in (nWordLoad + 1):nWord; xStr[begin+i-1] = zero(UInt) end
        @inbounds for i in (nWordLoad + 1):nWord; zStr[begin+i-1] = zero(UInt) end

        copyto!(xStr, firstindex(xStr), xWords, firstindex(xWords), nWordLoad)
        copyto!(zStr, firstindex(zStr), zWords, firstindex(zWords), nWordLoad)

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

PauliStr(pStr::PauliStr, nSite::Int=pStr.n, phase::PhaseFactor=pStr.phase) = 
PauliStr(pStr.x, pStr.z, phase, nSite)


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
    nSitePerWord = 8 * sizeof(UInt)
    body = ""

    for i in 1:pStr.n
        bitPos = i - 1
        w    = (bitPos ÷ nSitePerWord) + 1
        mask = one(UInt) << (bitPos % nSitePerWord)
        z = !iszero(pStr.z[w] & mask)
        x = !iszero(pStr.x[w] & mask)

        body *= if z & x
            'Y' * getSubscriptStr(i)
        elseif z
            'Z' * getSubscriptStr(i)
        elseif x
            'X' * getSubscriptStr(i)
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

Construct a `res::PauliSum{T}` with `T=real(C)` from `coeffs` and `strs` of equal length. 
The strings are deep-copied and rebuilt to have a common site count: the maximum site count 
over `strs`. Therefore, every string in `res` explicitly acts on the same number of sites 
(equal to [`countSites`](@ref)`(res)`). Each string's phase is absorbed into its matching 
coefficient. When `simplification=true` (by default), equal strings — including strings 
that become equal only after the rebuild (e.g., `X` and `XI`) — are combined into one term 
and any term whose coefficient is exactly zero is removed; when `simplification=false`, 
such equal strings are retained. In both cases the terms in the constructed `res` are 
stored in a deterministic canonical order such that for 
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

!!! warning
    Unlike other constructor methods, the retained strings are NOT reconstructed 
    (deep-copied). In other words, the entires in the field `.str` of the returned 
    `PauliStr` by this constructor are the same (subset of) `PauliStr` held by `ham.str`.
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
        nSite = iszero(inputSize) ? 0 : maximum(countSites, strs)
        for i in 1:inputSize; sInput[begin+i-1] = PauliStr(strs[begin+i-1], nSite) end
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
    countSites(str::PauliStr) -> Int

Return the number of sites `str` acts on, i.e., its site count (stored in its field `.n`).
"""
countSites(str::PauliStr) = str.n

"""
    countSites(ham::PauliSum) -> Int

Return the overall site count of `ham`, i.e., the maximum site count over its Pauli 
strings. Therefore, this method does not require each string in `ham.str` to explicitly 
act on the same number of sites. When `ham` holds no terms, `0` is returned.
"""
function countSites(ham::PauliSum)
    isempty(ham.str) ? 0 : maximum(countSites, ham.str)
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
    nSitePerWord = 8 * sizeof(UInt)
    remSites = str.n & (nSitePerWord - 1) #>> More efficient than `str.n % nSitePerWord`
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
`tolerance` is `zero(T)`.

# String ownership
The retained terms remain in `ham`'s order, and the returned sum's coefficients are held in 
an independent buffer. Its `PauliStr`s, however, are NOT reconstructed (deep-copied). In 
other words, the returned sum's `.str` entries are the same (subsets of) `PauliStr` held 
by `ham.str`.
"""
function curtail(ham::PauliSum{T}, tolerance::Real=(T<:Integer ? zero(T) : 8eps(T)); 
                 relative::Bool=false, inclusive::Bool=false) where {T<:Real}
    tolerance < 0 && throw(DomainError(tolerance, "`tolerance` must be non-negative."))

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


#> Bit order style: MSB-left (MSB ... LSB)
function alignWords(offset::Int, sourceLSB::T, sourceMSB::T) where {T<:BitUInteger}
    nBitPerWord = 8 * sizeof(T)
    piece1 = (sourceLSB >> offset)                #> EDCBA -> 00EDC (offset = 2)
    piece2 =  sourceMSB << (nBitPerWord - offset) #> SRQPO -> PO000
    piece1 | piece2                               #>          POEDC
end

function mergeWords(offset::Int, sourceLSB::T, sourceMSB::T) where {T<:BitUInteger}
    keptLSB = ~(typemax(T) << offset) & sourceLSB #> EDCBA -> 000BA (offset = 2)
    keptMSB =  (typemax(T) << offset) & sourceMSB #> SRQPO -> SRQ00
    keptMSB | keptLSB                             #>          SRQBA
end


"""
    shiftBits!(words::AbstractVector{T}, wordShift::Int, bitOffset::Int, 
               forward::Bool=true) where {T<:$BitUInteger} -> 
    typeof(words)

Shift the bits packed in `words` in place by a total of `wordShift` whole words plus
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
so `length(words)` is preserved. `wordShift` must be non-negative and `bitOffset` must 
satisfy `0 <= bitOffset < 8*sizeof(T)`, or a `DomainError` is thrown.
"""
function shiftBits!(words::AbstractVector{T}, wordShift::Int, bitOffset::Int, 
                    forward::Bool=true) where {T<:BitUInteger}
    wordShift < 0 && throw(DomainError(wordShift, "`wordShift` must be non-negative."))
    nBitPerWord = 8 * sizeof(T)
    if bitOffset >= nBitPerWord || bitOffset < 0
        throw(DomainError(bitOffset, "`0 <= bitOffset < $nBitPerWord` must hold true."))
    end

    nWord = length(words)
    offset, shift, wordIndexRange = if forward
        (mod(-bitOffset, nBitPerWord), -!iszero(bitOffset) - wordShift, nWord:-1:1)
    else
        (     bitOffset,                                     wordShift, 1: 1:nWord)
    end

    #> Alignment for bit words `w[bitOrder, wordIndex]` (`offset=1`, `nBitPerWord=4`):
    #>> ...|w[1,i      ] w[2,i      ] w[3,i      ] w[4,i        ]|w[1,i+1      ] ...
    #>> ... w[2,i+shift] w[3,i+shift] w[4,i+shift]|w[1,i+1+shift] w[2,i+1+shift] ...
    @inbounds for i in wordIndexRange
        j = i + shift
        pieceLo = (1 <= j <= nWord) ? words[begin+j-1] : zero(T)
        words[begin+i-1] = if iszero(offset)
            pieceLo
        else
            pieceHi = (1 <= j + 1 <= nWord) ? words[begin+j] : zero(T)
            alignWords(offset, pieceLo, pieceHi)
        end
    end

    words
end


"""
    shift!(str::PauliStr, n::Integer, toHigher::Bool=true) -> PauliStr

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
    n < 0 && throw(DomainError(n, "`n` must be non-negative."))
    (iszero(n) || iszero(str.n)) && return str #> Nothing to shift

    nSitePerWord = 8 * sizeof(UInt)
    wordShift, bitOffset = divrem(n, nSitePerWord)
    shiftBits!(str.x, wordShift, bitOffset, toHigher)
    shiftBits!(str.z, wordShift, bitOffset, toHigher)

    sanitize!(str) #> Clear any bits moved into the padding region past site `str.n`
end


"""
    pasteBits!(dstWords::AbstractVector{T}, dstBitIdxLo::Signed, 
               srcWords::AbstractVector{T}, srcBitIdxLo::Signed, 
               nBit::Signed) where {T<:$BitUInteger} -> 
    typeof(dstWords)
 
Copy `nBit` consecutive bits from `srcWords`, beginning at the 1-based bit index 
`srcBitIdxLo`, into `dstWords` beginning at the bit index `dstBitIdxLo`, in place, and 
then return the mutated `dstWords`. Both `dstWords` and `srcWords` are treated as one 
contiguous bit string in which the least-significant bit of the lowest-indexed word comes 
first, with the word width set by the element type at `8*sizeof(T)` bits. Every bit of 
`dstWords` outside the written window keeps its original value.
 
`nBit` must be non-negative, and each of `dstBitIdxLo` and `srcBitIdxLo` must be within 
the bit capacity of its buffer (a `DomainError` is thrown otherwise). `nBit` is allowed to 
exceed the number of bits available in `dstWords` or `srcWords`, in which case the 
excessive bits (on the more significant side) are dropped upon pasting.
 
!!! warning
    `dstWords` and `srcWords` must not share any underlying data; otherwise, the 
    copy-pasted result may be corrupted.
"""
function pasteBits!(dstWords::AbstractVector{T}, dstBitIdxLo::Signed, 
                    srcWords::AbstractVector{T}, srcBitIdxLo::Signed, nBit::Signed) where 
                   {T<:BitUInteger}
    nBitPerWord = 8 * sizeof(T)
    nWordDst = length(dstWords)
    nWordSrc = length(srcWords)
    nBitMaxDst = (nBitPerWord * nWordDst)
    nBitMaxSrc = (nBitPerWord * nWordSrc)

    0 <= nBit || throw(DomainError(nBit, "`nBit` must be non-negative."))
    (1 <= dstBitIdxLo <= nBitMaxDst) || 
    throw(DomainError(dstBitIdxLo, "`dstBitIdxLo` must be in 1:$nBitMaxDst."))
    (1 <= srcBitIdxLo <= nBitMaxSrc) || 
    throw(DomainError(srcBitIdxLo, "`srcBitIdxLo` must be in 1:$nBitMaxSrc."))

    iszero(nBit) && (return dstWords)

    #> `nBit` shall not go out of bound for both `dstWords` and `srcWords`. This clamp is 
    #>  the sole guarantor of every `@inbounds` fetch below being safe.
    nBit = min(nBit, nBitMaxDst-dstBitIdxLo+1, nBitMaxSrc-srcBitIdxLo+1)

    dstWordIdxLo, dstBitPosLo = fldmod1(dstBitIdxLo,        nBitPerWord)
    dstWordIdxHi, dstBitPosHi = fldmod1(dstBitIdxLo+nBit-1, nBitPerWord)
    wordShift, bitOffset = fldmod(srcBitIdxLo - dstBitIdxLo, nBitPerWord) #> bitOffset >= 0
    #>> srcBitIdx = dstBitIdx + wordShift * nBitPerWord + bitOffset

    oldHead = dstWords[begin+dstWordIdxLo-1]
    oldTail = dstWords[begin+dstWordIdxHi-1]

    #> Least-significant and most-significant words
    singleWordPaste = (dstWordIdxLo == dstWordIdxHi)
    @inbounds for (isMSW, dstWordIdx, nCover) in zip((false, true), 
                                                     (dstWordIdxLo, dstWordIdxHi), 
                                                     (dstBitPosLo-1, dstBitPosHi))
        srcWordIdx = dstWordIdx + wordShift

        newWord = if singleWordPaste && isMSW #> Tail pass re-reads the head pass's write
            dstWords[begin+dstWordIdx-1]
        elseif iszero(bitOffset) #> Word-by-word aligned case
            srcWords[begin+srcWordIdx-1]
        else
            pieceLo = ((isMSW || 1<=srcWordIdx) ? srcWords[begin+srcWordIdx-1] : zero(T))
            pieceHi = ( (srcWordIdx < nWordSrc) ? srcWords[begin+srcWordIdx  ] : zero(T))
            alignWords(bitOffset, pieceLo, pieceHi)
        end

        wordLo, wordHi = isMSW ? (newWord, oldTail) : (oldHead, newWord)
        newWord = mergeWords(nCover, wordLo, wordHi)
        dstWords[begin+dstWordIdx-1] = newWord
    end

    #> Sandwiched words
    if iszero(bitOffset) #> Word-by-word aligned case
        nSandwiched = dstWordIdxHi - dstWordIdxLo - 1
        if nSandwiched > 0 
            copyto!(dstWords, firstindex(dstWords)+dstWordIdxLo, 
                    srcWords, firstindex(srcWords)+dstWordIdxLo+wordShift, nSandwiched)
        end
    else
        @inbounds for dstWordIdx in (dstWordIdxLo+1):(dstWordIdxHi-1)
            srcWordIdx = dstWordIdx + wordShift
            pieceLo = srcWords[begin+srcWordIdx-1]
            pieceHi = srcWords[begin+srcWordIdx  ]
            newWord = alignWords(bitOffset, pieceLo, pieceHi)
            dstWords[begin+dstWordIdx-1] = newWord
        end
    end

    dstWords
end


"""
    paste!(dst::PauliStr, dstStart::Integer, src::PauliStr, 
           srcRange::UnitRange{<:Integer}=1:src.n; toHigher::Bool=true) -> PauliStr
 
Overwrite a contiguous window of `dst`'s sites with the single-site Pauli operators that 
`src` holds over the site range `srcRange`, in place, and then return the mutated `dst`. 
The phase of `dst` (`dst.phase`) is left untouched. When `toHigher=true` (default), site 
`first(srcRange)` of `src` lands on site `dstStart` of `dst`, with the remaining selected 
sites extending toward higher site indices; when `toHigher=false`, site `last(srcRange)` of 
`src` lands on site `dstStart`, with the remaining selected sites extending toward lower 
site indices. In either direction, the selected sites of `src` that fall outside `1:dst.n` 
are truncated. `dstStart` must be in `1:dst.n`, and a non-empty `srcRange` must be within 
`1:src.n` (a `DomainError` is thrown otherwise).

# Mechanism illustration (e.g., `[b1, b2, b3]` represents a three-site `dst`):
 
    toHigher = true  (dstStart=2):    toHigher = false (dstStart=1):
     dst: [b1, b2, b3] (initial)       dst:         [b1, b2, b3] (initial)
     src:     [c1, c2, c3]             src: [c1, c2, c3]
     dst: [b1, c1, c2] (result)        dst:         [c3, b2, b3] (result)
 
    toHigher = true  (dstStart=2, srcRange=2:3):
     dst: [b1, b2, b3] (initial)
     src:     [c2, c3] (site 1 of `src` deselected)
     dst: [b1, c2, c3] (result)
"""
function paste!(dst::PauliStr, dstStart::Integer, 
                src::PauliStr, srcRange::UnitRange{<:Integer}=1:src.n; toHigher::Bool=true)
    nSiteDst = dst.n
    if !(1 <= dstStart <= nSiteDst)
        throw(DomainError(dstStart, "`dstStart` must be in 1:$(nSiteDst)."))
    end

    nSitePasted = length(srcRange)
    iszero(nSitePasted) && (return dst)

    nSiteSrc = src.n
    srcSiteLo = (Int∘first)(srcRange)
    if !(1 <= srcSiteLo && last(srcRange) <= nSiteSrc)
        throw(DomainError(srcRange, "A non-empty `srcRange` must be within 1:$nSiteSrc."))
    end

    dst === src && (src = PauliStr(src)) #> Ensure buffer ownership invariant

    shiftedStart = Int(dstStart) - (toHigher ? 0 : nSitePasted)
    dstSiteLo = if shiftedStart < 0 #> Lower-side truncation for `src`
             nSitePasted += shiftedStart
            srcSiteLo -= shiftedStart
            1
        else
            shiftedStart + Int(!toHigher)
        end

    nSitePasted = min(nSitePasted, nSiteDst - dstSiteLo + 1) #> Truncate sites at `dst.n`

    pasteBits!(dst.x, dstSiteLo, src.x, srcSiteLo, nSitePasted)
    pasteBits!(dst.z, dstSiteLo, src.z, srcSiteLo, nSitePasted)

    dst
end


"""
    stampBits!(dstWords::AbstractVector{T}, dstBitIdxLo::Signed, bitVal::Bool, 
               nBit::Signed) where {T<:$BitUInteger} -> 
    typeof(dstWords)

Overwrite `nBit` consecutive bits of `dstWords` with the constant `bitVal`, beginning at 
the 1-based bit index `dstBitIdxLo`, in place, and then return the mutated `dstWords`. 
`dstWords` is treated as one contiguous bit string in which the least-significant bit of 
the lowest-indexed word comes first, with the word width set by the element type at 
`8*sizeof(T)` bits. Every bit of `dstWords` outside the written window keeps its original 
value.

`nBit` must be non-negative, and `dstBitIdxLo` must be within the bit capacity of 
`dstWords` (a `DomainError` is thrown otherwise). `nBit` is allowed to exceed the number 
of bits available in `dstWords`, in which case the excessive bits (on the more significant 
side) are dropped upon stamping.
"""
function stampBits!(dstWords::AbstractVector{T}, dstBitIdxLo::Signed, bitVal::Bool, 
                    nBit::Signed) where {T<:BitUInteger}
    nBitPerWord = 8 * sizeof(T)
    nBitMaxDst = nBitPerWord * length(dstWords)

    0 <= nBit || throw(DomainError(nBit, "`nBit` must be non-negative."))
    (1 <= dstBitIdxLo <= nBitMaxDst) || 
    throw(DomainError(dstBitIdxLo, "`dstBitIdxLo` must be in 1:$nBitMaxDst."))

    iszero(nBit) && (return dstWords)

    #> `nBit` shall not go out of bound for `dstWords`. This clamp is the sole guarantor 
    #>  of every `@inbounds` access below being safe.
    nBit = min(nBit, nBitMaxDst-dstBitIdxLo+1)

    dstWordIdxLo, dstBitPosLo = fldmod1(dstBitIdxLo,        nBitPerWord)
    dstWordIdxHi, dstBitPosHi = fldmod1(dstBitIdxLo+nBit-1, nBitPerWord)

    wordVal = bitVal ? typemax(T) : zero(T)
    oldHead = dstWords[begin+dstWordIdxLo-1]
    oldTail = dstWords[begin+dstWordIdxHi-1]

    #> Least-significant and most-significant words
    singleWordStamp = (dstWordIdxLo == dstWordIdxHi)
    @inbounds for (isMSW, dstWordIdx, nCover) in zip((false, true), 
                                                     (dstWordIdxLo, dstWordIdxHi), 
                                                     (dstBitPosLo-1, dstBitPosHi))
        #> Tail pass re-reads the head pass's write
        newWord = (singleWordStamp && isMSW) ? dstWords[begin+dstWordIdx-1] : wordVal
        dstWords[begin+dstWordIdx-1] = if iszero(nCover % nBitPerWord)
            newWord
        else
            wordLo, wordHi = isMSW ? (newWord, oldTail) : (oldHead, newWord)
            mergeWords(nCover, wordLo, wordHi)
        end
    end

    #> Sandwiched words
    @inbounds for dstWordIdx in (dstWordIdxLo+1):(dstWordIdxHi-1)
        dstWords[begin+dstWordIdx-1] = wordVal
    end

    dstWords
end


"""
    stamp!(dst::PauliStr, startSite::Integer, opSym::PauliSym, nSite::Integer=1; 
           toHigher::Bool=true) -> PauliStr

Overwrite a contiguous window of `nSite` sites of `dst` with the single-site Pauli 
operator `opSym::`[`PauliSym`](@ref), in place, and then return the mutated `dst`. The 
phase of `dst` (`dst.phase`) is left untouched. When `toHigher=true` (default), the window 
starts at site `startSite` and extends toward higher site indices; when `toHigher=false`, 
the window ends at site `startSite` and extends toward lower site indices. In either 
direction, the part of the window that falls outside `1:dst.n` is truncated. `startSite` 
must be in `1:dst.n`, and `nSite` must be non-negative (a `DomainError` is thrown 
otherwise).

# Mechanism illustration (e.g., `[b1, b2, b3]` represents a three-site `dst`):

    toHigher = true  (startSite=2, nSite=3):   toHigher = false (startSite=1, nSite=3):
     dst: [b1, b2, b3]     (initial)            dst:         [b1, b2, b3] (initial)
     op:      [op, op, op]                      op:  [op, op, op]
     dst: [b1, op, op]     (result)             dst:         [op, b2, b3] (result)
"""
function stamp!(dst::PauliStr, startSite::Integer, opSym::PauliSym, nSite::Integer=1; 
                toHigher::Bool=true)
    nSiteDst = dst.n
    if !(1 <= startSite <= nSiteDst)
        throw(DomainError(startSite, "`startSite` must be in 1:$(nSiteDst)."))
    end

    nSite < 0 && throw(DomainError(nSite, "`nSite` must be non-negative."))
    iszero(nSite) && (return dst)
    nSiteStamped = Int(nSite)

    shiftedStart = Int(startSite) - (toHigher ? 0 : nSiteStamped)
    dstSiteLo = if shiftedStart < 0 #> Lower-side truncation for the window
            nSiteStamped += shiftedStart
            1
        else
            shiftedStart + Int(!toHigher)
        end

    nSiteStamped = min(nSiteStamped, nSiteDst - dstSiteLo + 1) #> Truncate sites at `dst.n`

    xBit = (opSym == symX || opSym == symY)
    zBit = (opSym == symZ || opSym == symY)
    stampBits!(dst.x, dstSiteLo, xBit, nSiteStamped)
    stampBits!(dst.z, dstSiteLo, zBit, nSiteStamped)

    dst
end


"""
    reframe(ham::PauliSum{T}, nSite::Integer=countSites(ham); lowToHigh::Bool=true, 
            filler::PauliSym=symI, simplification::Bool=true) where {T<:Real} -> 
    PauliSum{T}

Return a `res::PauliSum{T}` by rebuilding each `PauliStr` in `ham.str` as a `PauliStr` in 
`res.str` within a fixed frame of exactly `nSite` sites, i.e., a reframed string. `nSite` 
must be non-negative (a `DomainError` is thrown otherwise), and when `length(res.str) > 0`, 
`res` holds the following property:

    minimum(countSites, res.str) == countSites(res) == nSite

When `lowToHigh=true` (default), each single-site operator from the original string is 
placed site-by-site starting from the frame's low edge: the site-`k` operator of the 
reframed string `strR` is set to be the operator acting on site `k` in the original string 
`strO`. Conversely, when `lowToHigh=false`, the site-`k` operator in `strR` is set to be 
the operator acting on site [`countSites`](@ref)`(strO) - nSite + k` in `strO`. In either 
case, every site in the frame not covered by the operators (including single-site 
identities) in the original string is assigned a `filler::`[`PauliSym`](@ref) operator, 
while any of those operators falling outside the frame is cropped away. Regardless of the 
value of `nSite`, each original string's phase information is transferred to the 
corresponding reframed string.

After the reframed strings are constructed, they are then collected (along with the 
coefficients associated with their respective original strings held by `ham`) to construct 
`res::PauliSum{T}` in canonical order. If `simplification=true`, the same simplification 
procedure used by the constructor of [`PauliSum`](@ref) is applied to `res` before it is 
returned. Depending on the specific choice of `filler`, or the value of `nSite` compared to 
`countSites(ham)`, `res` may not be algebraically equivalent to `ham`. Only when 
`filler=symI` and `nSite >= countSites(ham)` is `res` guaranteed to represent the same 
operator as `ham` (up to numerical round-off errors if `simplification=true`).

# Mechanism illustration (`[c1, c2, c3]` represents a three-site string, `fl` the filler):

    lowToHigh = true  (nSite=5):    lowToHigh = false (nSite=5):
     old: [c1, c2, c3]               old:         [c1, c2, c3]
     new: [c1, c2, c3, fl, fl]       new: [fl, fl, c1, c2, c3]

    lowToHigh = true  (nSite=2):    lowToHigh = false (nSite=2):
     old: [c1, c2, c3]               old: [c1, c2, c3]
     new: [c1, c2]                   new:     [c2, c3]

# String ownership
The returned sum does not reference any data in `ham`: its coefficients are held in an 
independent buffer, and its `PauliStr`s are freshly constructed.

# Example
```julia
julia> ham = PauliSum([1, 2], [pauli"XZ", pauli"YIX"]);

julia> countSites(ham) #> The constructor has rebuilt `pauli"XZ"` onto 3 sites as `XZI`
3

julia> reframe(ham, 5) == PauliSum([1, 2], [pauli"XZIII", pauli"YIXII"])
true

julia> reframe(ham, 5, lowToHigh=false) == PauliSum([1, 2], [pauli"IIXZI", pauli"IIYIX"])
true
```
"""
function reframe(ham::PauliSum, nSite::Integer=countSites(ham); lowToHigh::Bool=true, 
                 filler::PauliSym=symI, simplification::Bool=true)
    nSite < 0 && throw(DomainError(nSite, "`nSite` must be non-negative."))
    nSite = Int(nSite)

    coeffs = ham.coeff
    strs = map(ham.str) do oldStr
        newStr = PauliStr(nSite, filler, oldStr.phase)
        if nSite > 0
            iSiteStart = ifelse(lowToHigh, 1, nSite)
            paste!(newStr, iSiteStart, oldStr, toHigher=lowToHigh)
        end
        newStr
    end
    PauliSum(coeffs, strs, simplification)
end