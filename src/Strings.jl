export PauliStr, PauliSum, @pauli_str

using LinearAlgebra: dot
import Base: hash, ==, show

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


function hash(pStr::PauliStr, hashCode::UInt)
    code = hash(pStr.phase, hashCode)
    code = hash(pStr.n, code)
    code = hash(pStr.x, code)
    hash(pStr.z, code)
end

function ==(pStr1::PauliStr, pStr2::PauliStr)
    isequal(pStr1.phase, pStr2.phase) && 
    isequal(pStr1.n    , pStr2.n    ) && 
    isequal(pStr1.x    , pStr2.x    ) && 
    isequal(pStr1.z    , pStr2.z    )
end


function getPauliSymVec(str::Union{String, AbstractVector{<:Integer}})
    [getPauliSym(c::Union{Char, Integer}) for c in str]
end

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


struct PauliSum{T<:RealOrComplex} <: DiscreteOperator
    coeff::Memory{T}
    string::Memory{PauliStr}

    function PauliSum(coeff::AbstractVector{T}, string::AbstractVector{PauliStr}) where {T}
        if length(coeff) != length(string)
            throw(ArgumentError("`coeff` and `string` should have the same length."))
        end

        new{T}(convert(Memory{T}, coeff), convert(Memory{PauliStr}, string))
    end
end

function hash(pSum::PauliSum, hashCode::UInt)
    code = hash(pSum.string, hashCode)
    hash(pSum.coeff, code)
end

function ==(pSum1::PauliSum, pSum2::PauliSum)
    (pSum1.coeff == pSum2.coeff) && (pSum1.string == pSum2.string)
end

function PauliSum(::Type{T}, str::AbstractVector{PauliStr}) where {T<:RealOrComplex}
    coeff = Memory{T}(undef, length(str))
    coeff .= one(T)
    PauliSum(coeff, str)
end

PauliSum(str) = PauliSum(Int, str)