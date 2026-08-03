export PauliSym, symI, symX, symY, symZ, PhaseFactor, evalPhase, mul, toMatrix

const PauliXMatEntries = (0,   1,    1,  0) #> (m11, m21, m12, m22)
const PauliYMatEntries = (0, 1im, -1im,  0)
const PauliZMatEntries = (1,   0,    0, -1)


"""
    PauliSym <: Enum{UInt8}

A `UInt8`-based enumeration of the four single-site Pauli operators. Each instance's 
integer value is the operator's two-bit binary code `xz` (i.e., `2x + z`), where the high 
bit `x` marks the X-component and the low bit `z` marks the Z-component:

    symI::PauliSym  =>  0 # binary 00: identity (no component)
    symZ::PauliSym  =>  1 # binary 01: Z (Z-component only)
    symX::PauliSym  =>  2 # binary 10: X (X-component only)
    symY::PauliSym  =>  3 # binary 11: Y (both components)

Hence, `PauliSym(num)` decodes `num` in `0:3` directly as the bit pair and imposes the 
value order: `symI < symZ < symX < symY`.
"""
@enum PauliSym::UInt8 begin #> Assigned values align with X-Z layout
    symI=0 #> 00
    symZ=1 #> 01
    symX=2 #> 10
    symY=3 #> 11
end

function genPauliSym(ele::AbstractChar)
    num = if ele == 'I'
        0
    elseif ele == 'Z'
        1
    elseif ele == 'X'
        2
    elseif ele == 'Y'
        3
    else
        throw(ArgumentError("\'$ele\' is not a valid letter for a Pauli operator"))
    end

    genPauliSym(num)
end

genPauliSym(num::Integer) = PauliSym(num)


"""

    toMatrix(::Type{T}, sym::PauliSym) where {T<:Real} -> Matrix{Complex{T}}

    toMatrix(sym::PauliSym) -> Matrix{Complex{Int}}

Return the 2×2 matrix representation (in the Pauli-Z eigenbasis) of the single-site 
Pauli operator tagged by `sym::`[`PauliSym`](@ref), with element type as `Complex{T}`. When 
`T` is unspecified, method `toMatrix(sym::PauliSym)` defaults it to `Int`. `T` is 
disallowed to be `Bool`. The returned matrix is newly allocated on every call and shares no 
data with any global state.
"""
function toMatrix(::Type{T}, sym::PauliSym) where {T<:Real}
    (T <: Bool) && throw(ArgumentError("T = $T is disallowed."))
    mat = zeros(Complex{T}, 2, 2)
    matVec = vec(mat)

    if sym == symI
        matVec[begin] = mat[end] = one(T)
    elseif sym == symX
        matVec .= PauliXMatEntries
    elseif sym == symY
        matVec .= PauliYMatEntries
    else
        matVec .= PauliZMatEntries
    end

    mat
end

toMatrix(sym::PauliSym) = toMatrix(Int, sym)


"""
    PhaseFactor <: Enum{UInt8}

A `UInt8`-backed enumeration of the four phase coefficients `im^k` that a Pauli-group
element may carry (the fourth roots of unity). Each instance's integer value is the
exponent `k` of `im`:

    posRea::PhaseFactor  =>  im^0 # +1
    posImg::PhaseFactor  =>  im^1 # +im
    negRea::PhaseFactor  =>  im^2 # -1
    negImg::PhaseFactor  =>  im^3 # -im

Use [`evalPhase`](@ref) to obtain the value for the corresponding complex coefficient.
"""
@enum PhaseFactor::UInt8 begin
    posRea=0 #> i^0 == +1
    posImg=1 #> i^1 == +im
    negRea=2 #> i^2 == -1
    negImg=3 #> i^3 == -im
end

const PhaseOrCoeff = Union{PhaseFactor, RealOrComplex}


"""
    mul(l::PhaseFactor, r::PhaseFactor) -> PhaseFactor

Multiply two phase factors and return their product as another `PhaseFactor`.
"""
mul(l::PhaseFactor, r::PhaseFactor) = PhaseFactor((UInt8(l) + UInt8(r)) & 3)

Base.:*(l::PhaseFactor, r::PhaseFactor) = mul(l, r)


"""
    evalPhase(phase::PhaseFactor) -> Complex{Int}

Return the complex phase coefficient represented by `phase`.
"""
evalPhase(phase::PhaseFactor) = im^Int(phase)


const CONSTVAR!!subscriptNum = 
      Dict(['0'=>'₀', '1'=>'₁', '2'=>'₂', '3'=>'₃', '4'=>'₄', 
            '5'=>'₅', '6'=>'₆', '7'=>'₇', '8'=>'₈', '9'=>'₉'])

function getSubscriptStr(num::Signed)
    num < 0 && throw(DomainError(num, "`num` must be non-negative."))
    mapreduce(c->CONSTVAR!!subscriptNum[c], *, string(num), init="")
end