export PauliSym, symI, symX, symY, symZ, PhaseFactor, toString, toPhase, evalPhase

public mul

const PauliXMatEntries = (0,   1,    1,  0) #> (m11, m21, m12, m22)
const PauliYMatEntries = (0, 1im, -1im,  0)
const PauliZMatEntries = (1,   0,    0, -1)


"""
    PauliSym <: Enum{UInt8}

A `UInt8`-based enumeration of the four single-site Pauli operators. Each instance's 
integer value is the operator's two-bit binary code `xz` (i.e., `2x + z`), where the high 
bit `x` marks the ``x``-component and the low bit `z` marks the ``z``-component:

    Int(symI::PauliSym)  ==  0 # binary 00: identity (no component)
    Int(symZ::PauliSym)  ==  1 # binary 01: Z (``z``-component only)
    Int(symX::PauliSym)  ==  2 # binary 10: X (``x``-component only)
    Int(symY::PauliSym)  ==  3 # binary 11: Y (both components)

Hence, `PauliSym(num)` serves as a symbolic representation of the (single-site) Pauli 
operators for `num` in `0:3`:

```math
\\mathtt{symI} \\mapsto \\hat{I}, \\quad
\\mathtt{symZ} \\mapsto \\hat{Z}, \\quad
\\mathtt{symX} \\mapsto \\hat{X}, \\quad
\\mathtt{symY} \\mapsto \\hat{Y};
```

and `PauliSym` also imposes a strict total order: `symI < symZ < symX < symY`.
"""
@enum PauliSym::UInt8 begin #> Assigned values align with X-Z layout
    symI=0 #> 00
    symZ=1 #> 01
    symX=2 #> 10
    symY=3 #> 11
end

const PauliCharacters = ('I', 'Z', 'X', 'Y')

function genPauliSym(ele::AbstractChar)
    idx = findfirst(==(ele), PauliCharacters)
    if idx === nothing
        throw(ArgumentError("'$ele' is not a valid character for a Pauli operator."))
    end
    genPauliSym(idx - 1)
end

genPauliSym(num::Integer) = PauliSym(num)


"""
    PhaseFactor <: Enum{UInt8}

A `UInt8`-based enumeration of the four phase coefficients `im^k` that a Pauli-group
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

    toString(p::PhaseFactor; omitPlusSign::Bool=false) -> String

Return the `String` representation of `p::`[`PhaseFactor`](@ref). This method produces the 
representation consistent with that of the phase showing up in the string returned by 
[`toString(::PauliStr)`](@ref).
"""
function toString(p::PhaseFactor; omitPlusSign::Bool=false)
    ps = omitPlusSign ? "" : "+"
    if p === posRea
        ps
    elseif p === posImg
        ps * "i"
    elseif p === negRea
        "-"
    else
        "-i"
    end
end


const ValidPhaseMarkers = unique(
    toString(p; omitPlusSign) for p in instances(PhaseFactor), omitPlusSign in (false, true)
)

"""
    toPhase(input::Union{Complex, Real, AbstractString}) -> PhaseFactor

Return a `PhaseFactor` which can be parsed from a valid `input`. If `input` is a `Real`, 
it should be equal to one of the return values of [`evalPhase(p)`](@ref `evalPhase`), in 
which case `p` is returned; if `input` is a `AbstractString`, its should be equal to one of 
the valid markers: $(join(("`\"$s\"`" for s in ValidPhaseMarkers), ", ")); otherwise, 
`input` is not valid, and an `ArgumentError` is thrown.
"""
function toPhase(input::RealOrComplex)
    if isone(input)
        posRea
    elseif isone(-im * input)
        posImg
    elseif isone(-input)
        negRea
    elseif isone(im * input)
        negImg
    else
        throw(ArgumentError("`input=$input` is not a numerical value representable by "*
                            "`PhaseFactor`."))
    end
end

function toPhase(input::AbstractString)
    for p in instances(PhaseFactor), omitPlusSign in (false, true)
        input == toString(p; omitPlusSign) && (return p)
    end
    throw(ArgumentError("`input=\"$input\"` is not one of valid phase markers:\n" * 
                        join(("`\"$s\"`" for s in ValidPhaseMarkers), ", ") * "."))
end


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
