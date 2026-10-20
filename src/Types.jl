abstract type LinearOperator <: Any end
abstract type DiscreteOperator <: LinearOperator end

abstract type StructuredType <: Any end

#> Types whose instances can properly have methods for negation
const SignedReal = Union{AbstractFloat, Signed, Rational{<:Signed}}

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

const SRealOrComplex{T<:SignedReal} = Union{T, Complex{T}}

const FloatOrComplex{T<:AbstractFloat} = SRealOrComplex{T}

const BitUInteger = Union{UInt8, UInt16, UInt32, UInt64, UInt128}

const SameTypePair{T} = Pair{T, T}

const MissingOr{T} = Union{Missing, T}

const NonEmptyTuple{T, M} = Tuple{T, Vararg{T, M}}

const Interface = Union{Function, Type, Module}


#> Pre-declared functions for forward referencing in docstrings <#
function toAdjoint end #> From Operations.jl
