abstract type LinearOperator <: Any end
abstract type DiscreteOperator <: LinearOperator end

abstract type StructuredType <: Any end

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

const FloatOrComplex{T<:AbstractFloat} = RealOrComplex{T}

const SameTypePair{T} = Pair{T, T}

const BitUInteger = Union{UInt8, UInt16, UInt32, UInt64, UInt128}

const MissingOr{T} = Union{Missing, T}

const Interface = Union{Function, Type, Module}

const NonEmptyTuple{T, M} = Tuple{T, Vararg{T, M}}


#> Pre-declared functions for forward referencing in docstrings <#
function toAdjoint end #> From Operations.jl
