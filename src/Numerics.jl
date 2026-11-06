public extendType, evalMagnitude

"""
    toBitIntData(::Type{T}, val::Integer, boundName::AbstractString="val"
                 ) where {T<:$BitSInteger} -> 
    T

    toBitIntData(val::Integer, boundName::AbstractString="val") -> Int

Convert `val` to `T` (by default, `Int`), or throw a `DomainError` that refers to `val` as 
`\$boundName` when `val` is not representable by `T`. 
"""
function toBitIntData(::Type{T}, val::Integer, boundName::AbstractString="val"
                        ) where {T<:BitSInteger}
    if !(applicable(T, val) && typemin(T) <= val <= typemax(T))
        throw(DomainError(val, "Aside from being an `Integer`, `$boundName` must also "*
                               "be representable by `$T`."))
    end

    T(val)
end

toBitIntData(val::Integer, boundName::AbstractString="val") = 
toBitIntData(Int, val, boundName)

"""
    toBitIntData(::Type{T}, rng::UnitRange{<:Integer}, boundName::AbstractString="rng"
                 ) where {T<:$BitSInteger} -> 
    UnitRange{T}

    toBitIntData(rng::UnitRange{<:Integer}, boundName::AbstractString="rng") -> 
    UnitRange{Int}

Convert `rng` to `UnitRange{T}`, for which an individual error message is thrown wether 
either the start or the stop index of `rng` (referred as `\$boundName`) is not 
representable by `T`.
"""
function toBitIntData(::Type{T}, rng::UnitRange{<:Integer}, 
                        boundName::AbstractString="rng") where {T<:BitSInteger}
    lo = toBitIntData(T, first(rng), "first($boundName)")
    hi = toBitIntData(T, last(rng), "last($boundName)")
    lo:hi
end

toBitIntData(rng::UnitRange{<:Integer}, boundName::AbstractString="rng") = 
toBitIntData(Int, rng, boundName)


#> Adaptive overflow-check arithmetics for numeric types based on fixed-width signed 
#> integers. Every other type falls back to `+`, `*`, `conj`, and `-` (so floating-point 
#> arithmetic remains unchanged). Different numeric types are first promoted before the 
#> operation, where an overflow can be automatically detected via type conversion, followed 
#> by an `InexactError` throw.
function adaptAdd(num1::T, num2::T) where {T<:RealOrComplex}
    if T <: BitSInteger
        Base.checked_add(num1, num2)
    elseif T <: Complex{<:BitSInteger}
        x = adaptAdd(real(num1), real(num2))
        y = adaptAdd(imag(num1), imag(num2))
        Complex(x, y)
    else
        num1 + num2
    end
end

adaptAdd(num1::RealOrComplex, num2::RealOrComplex) = adaptAdd(promote(num1, num2)...)


function adaptMul(num1::T, num2::T) where {T<:RealOrComplex}
    if T <: BitSInteger
        Base.checked_mul(num1, num2)
    elseif T <: Complex{<:BitSInteger}
        x1, y1 = reim(num1)
        x2, y2 = reim(num2)
        x = Base.checked_sub(adaptMul(x1, x2), adaptMul(y1, y2))
        y = adaptAdd(adaptMul(x1, y2), adaptMul(y1, x2))
        Complex(x, y)
    else
        num1 * num2
    end
end

adaptMul(num1::RealOrComplex, num2::RealOrComplex) = adaptMul(promote(num1, num2)...)


function adaptConj(num::T) where {T<:RealOrComplex}
    if T <: Complex{<:BitSInteger}
        Complex(real(num), adaptNeg( imag(num) ))
    else
        conj(num)
    end
end


function adaptNeg(num::T) where {T<:RealOrComplex}
    if T <: BitSInteger
        Base.checked_neg(num)
    elseif T <: Complex{<:BitSInteger}
        Complex(adaptNeg( real(num) ), adaptNeg( imag(num) ))
    else
        -num
    end
end


#>-- Reference(s) --<#
#> [DOI] 10.1002/zamm.19740540106 (Improved Kahan--Babuska algorithm proposed by Neumaier)
@inline function neumaierStep(x1::T, x2::T) where {T<:AbstractFloat}
    total = x1 + x2
    residue = ifelse(abs(x1) < abs(x2), (x2 - total) + x1, (x1 - total) + x2)
    (total, residue)
end

function neumaierStep(x1::FloatOrComplex{T}, x2::FloatOrComplex{T}) where {T<:AbstractFloat}
    rTotal, rResidue = neumaierStep(real(x1), real(x2))
    iTotal, iResidue = neumaierStep(imag(x1), imag(x2))
    (Complex(rTotal, iTotal), Complex(rResidue, iResidue))
end


function neumaierAdd(total::C, val::C, residue::C=zero(C)) where {C<:FloatOrComplex}
    total, localResidue = neumaierStep(total, val)
    (total, residue + localResidue)
end

neumaierAdd(total::C, val::C, residue::C=zero(C)) where {C<:RealOrComplex} = 
(adaptAdd(total, val), residue)


function neumaierSum(mapper::F, ::Type{T}, iterable::S) where {F, T<:RealOrComplex, S}
    eleT = T <: Real ? T : typeintersect(T, Complex)
    if !(isconcretetype(eleT) && (isconcretetype∘real)(eleT))
        throw(ArgumentError("The Neumaier sum based on `T = $T` is not well-defined."))
    end

    if Base.IteratorSize(S) isa Base.IsInfinite
        throw(ArgumentError("`iterable` must not be an infinite iterator."))
    end

    naiveSum = zero(eleT)
    sumResidue = zero(eleT)

    for item in iterable
        formattedVal = eleT(item|>mapper)
        naiveSum, sumResidue = neumaierAdd(naiveSum, formattedVal, sumResidue)
    end

    naiveSum + sumResidue
end

neumaierSum(::Type{T}, iterable) where {T<:RealOrComplex} = 
neumaierSum(Base.identity, T, iterable)

neumaierSum(arr::AbstractArray{T}) where {T<:RealOrComplex} = 
neumaierSum(T, arr)


#>-- Reference(s) --<#
#> [DOI] 10.48550/arXiv.1904.09481
"""
    evalMagnitude(c::Union{Complex{T}, T}, allowNaN::Bool=true) where {T<:Real} -> float(T)

Return the absolute value of `c` using an 
[unfused error-mitigated method](https://doi.org/10.48550/arXiv.1904.09481). 
Unlike [`Base.abs`](https://docs.julialang.org/en/v1/base/math/#Base.abs), this method only 
relies on IEEE-754 defined operations separately (with correct rounding behaviors) for 
the standard numeric types in Julia (e.g., `Float64`, `Int`). Therefore, if no 
[type piracy](https://docs.julialang.org/en/v1/manual/style-guide/#avoid-type-piracy) is 
committed for those numeric types, the returned values are bitwise reproducible across 
different hardware platforms, except potentially for the bit patterns of `NaN` results.

When `allowNaN=true`, NaN inputs propagate to the result unless another component is 
infinite (verified by `isinf`), in which case positive infinity is returned.
Otherwise, a `DomainError` is thrown if any input component is NaN.
"""
function evalMagnitude(c::Complex{T}, allowNaN::Bool=true) where {T<:AbstractFloat}
    comps = real(c), imag(c)
    if !allowNaN
        for (str, comp) in zip(("real", "imag"), comps)
            isnan(comp) && throw(DomainError(comp, "`$str(c)` is disallowed to be `NaN`."))
        end
    end

    any(isinf, comps) && (return T(Inf)) #> Early exit 1 to comply with IEEE-754

    x, y = abs.(comps)
    x < y && ((x, y) = (y, x))           #> Ensure `x >= y` unless either is `NaN`

    iszero(y) && (return x)              #> Early exit 2
    isnan( x) && (return x)              #> Early exit 3
    isnan( y) && (return y)              #> Early exit 4

    y <= x * sqrt(eps(T)/2) && (return x) #> Widely varying operands

    if !(T <: Float32) && promote_type(T, Float32) == Float32
        xW, yW = Float32(x), Float32(y) #> Avoid infinite `scale`
        T(sqrt(xW*xW + yW*yW))
    else
        minT = floatmin(T)
        scale = eps(T) * sqrt(minT)     #> Rescale to avoid overflow/underflow

        if x > sqrt(floatmax(T)/2)
            x *= scale
            y *= scale
            factor = inv(scale)
        elseif y < sqrt(minT)
            x /= scale
            y /= scale
            factor = scale
        else
            factor = one(T)
        end

        h = sqrt(x*x + y*y)

        if h <= 2y
            delta = h - y
            correction = x*(2delta - x) + (delta - 2(x-y))*delta
        else
            delta = h - x
            correction = 2delta*(x - 2y) + (4delta - y)*y + delta*delta
        end

        (h - correction/(2h)) * factor
    end
end

function evalMagnitude(c::Complex{T}, allowNaN::Bool=true) where {T<:Real}
    coreT::Type{<:AbstractFloat} = float(T)
    evalMagnitude(Complex(coreT(c|>real), coreT(c|>imag)), allowNaN)
end

function evalMagnitude(c::AbstractFloat, allowNaN::Bool=true)
    !allowNaN && isnan(c) && throw(DomainError(c, "NaN is disallowed."))
    abs(c)
end

evalMagnitude(c::Real, allowNaN::Bool=true) = evalMagnitude(float(c), allowNaN)


"""
    extendType(::Type{C}, ::Type{T}) where {C<:SignedReal, T<:SignedReal} -> Type

    extendType(::Type{C}, ::Type{Complex{T}}) where {C<:SignedReal, T<:SignedReal} -> 
    Type

Return a new data type `E::Type` that is always at least as accurate as `T` (hence can 
losslessly represent `T`) based on a core type `C`.

# Examples
```jldoctest
julia> $(extendType|>repr)(Int, Float64)
Float64

julia> $(extendType|>repr)(Int16, Complex{Rational{Int8}})
Complex{Rational{Int16}}

julia> $(extendType|>repr)(Float64, Rational{Int16})
Rational{Int16}
```
"""
function extendType(::Type{C}, ::Type{T}) where {C<:SignedReal, T<:SignedReal}
    bl1 = C <: AbstractFloat
    bl2 = T <: AbstractFloat
    if (bl1 && bl2) || (!bl1 && !bl2)
        promote_type(C, T)
    else
        T
    end
end

extendType(::Type{C}, ::Type{Complex{T}}) where {C<:SignedReal, T<:SignedReal} = 
(complex∘extendType)(C, T)
