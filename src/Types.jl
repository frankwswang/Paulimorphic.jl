abstract type LinearOperator end
abstract type DiscreteOperator <: LinearOperator end

const RealOrComplex{T<:Real} = Union{T, Complex{T}}