public extendType, evalMagnitude

"""
    toBitIntData(::Type{T}, val::Integer, objName::Symbol=:val, 
                 accessorName::Union{Missing, Symbol}=missing
                 ) where {T<:$BitSInteger} -> 
    T

    toBitIntData(val::Integer, objName::Symbol=:val, 
                 accessorName::Union{Missing, Symbol}=missing) -> 
    Int

Convert `val` to `T` (by default, `Int`), or throw a `DomainError` when `val` is not 
representable by `T`. The error message refers to `val` as `\$objName` if `accessorName` 
is `missing` (the default), and as `\$accessorName(\$objName)` otherwise.
"""
function toBitIntData(::Type{T}, val::Integer, objName::Symbol=:val, 
                      accessorName::MissingOr{Symbol}=missing) where {T<:BitSInteger}
    if !(applicable(T, val) && typemin(T) <= val <= typemax(T))
        valName = ismissing(accessorName) ? "$objName" : "$accessorName($objName)"
        throw(DomainError(val, "Aside from being an `Integer`, `$valName` must also "*
                               "be representable by `$T`."))
    end

    T(val)
end

toBitIntData(val::Integer, objName::Symbol=:val, 
             accessorName::MissingOr{Symbol}=missing) = 
toBitIntData(Int, val, objName, accessorName)

"""
    toBitIntData(::Type{T}, rng::UnitRange{<:Integer}, objName::Symbol=:rng
                 ) where {T<:$BitSInteger} -> 
    UnitRange{T}

    toBitIntData(rng::UnitRange{<:Integer}, objName::Symbol=:rng) -> UnitRange{Int}

Convert `rng` to `UnitRange{T}`, throwing an individual error message when either the 
start or the stop index of `rng` (referred to as `first(\$objName)` or 
`last(\$objName)`, respectively) is not representable by `T`.
"""
function toBitIntData(::Type{T}, rng::UnitRange{<:Integer}, objName::Symbol=:rng
                      ) where {T<:BitSInteger}
    lo = toBitIntData(T, first(rng), objName, :first)
    hi = toBitIntData(T, last(rng), objName, :last)
    lo:hi
end

toBitIntData(rng::UnitRange{<:Integer}, objName::Symbol=:rng) = 
toBitIntData(Int, rng, objName)


#> Adaptive overflow-check arithmetics for numeric types based on fixed-width signed 
#> integers. Every other type falls back to `+`, `*`, `conj`, and `-` (so floating-point 
#> arithmetic remains unchanged). Different numeric types are first promoted before the 
#> operation, where an overflow can be automatically detected via type conversion, followed 
#> by an `InexactError` throw.
function adaptAdd(num1::T, num2::T) where {T<:RealOrComplex}
    if T <: BitSInteger
        Base.Checked.checked_add(num1, num2)
    elseif T <: ComplexBitSInteger
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
        Base.Checked.checked_mul(num1, num2)
    elseif T <: ComplexBitSInteger
        x1, y1 = reim(num1)
        x2, y2 = reim(num2)
        x = Base.Checked.checked_sub(adaptMul(x1, x2), adaptMul(y1, y2))
        y = adaptAdd(adaptMul(x1, y2), adaptMul(y1, x2))
        Complex(x, y)
    else
        num1 * num2
    end
end

adaptMul(num1::RealOrComplex, num2::RealOrComplex) = adaptMul(promote(num1, num2)...)


function adaptConj(num::T) where {T<:RealOrComplex}
    if T <: ComplexBitSInteger
        Complex(real(num), adaptNeg( imag(num) ))
    else
        conj(num)
    end
end


function adaptNeg(num::T) where {T<:RealOrComplex}
    if T <: BitSInteger
        Base.Checked.checked_neg(num)
    elseif T <: ComplexBitSInteger
        Complex(adaptNeg( real(num) ), adaptNeg( imag(num) ))
    else
        -num
    end
end


#> A conditional caller of `adaptNeg`. It returns `adaptNeg(num)` if `flag` is `true`, and 
#> `num` otherwise. For numeric types based on fixed-width signed integers (e.g., `Int64`), 
#> the result is computed without a branch on `flag`; as a result, this function is often 
#> faster than branch-based execution of `adaptNeg` when the branch predictions fail.
function condNeg(num::T, flag::Bool) where {T<:RealOrComplex}
    if T <: BitSInteger
        mask = -T(flag) #> `0` (no bits) if `flag` is `false`, and `-1` (all bits) otherwise
        res, hasOverflow = Base.Checked.sub_with_overflow(xor(num, mask), mask)
        #> The subtraction overflows only when the negation does (under `num=typemin(T)`)
        hasOverflow ? adaptNeg(num) : res #> `adaptNeg` reports the `OverflowError`
    elseif T <: ComplexBitSInteger
        Complex(condNeg( real(num), flag ), condNeg( imag(num), flag ))
    else
        flag ? adaptNeg(num) : num
    end
end


#> Value-based check of `num1 == -num2` where negation `-(x)` operation is avoided when 
#> possible to avoid edge cases like `-0xff === 0x01` (where the latter is the additive 
#> inverse of the former only when they are passed to `Base.:+`).
function checkNeg(num1::T, num2::T) where {T<:Real}
    if T <: Union{BitSInteger, BitUInteger}
        #> Two integers are opposite numbers iff they add up to zero without an overflow
        total, hasOverflow = Base.Checked.add_with_overflow(num1, num2)
        !hasOverflow & iszero(total)
    elseif T <: Rational
        #> A `Rational` is stored in lowest terms with a non-negative denominator
        denominator(num1) == denominator(num2) && 
        checkNeg(numerator(num1), numerator(num2))
    else
        num1 == -num2
    end
end

checkNeg(num1::T, num2::T) where {T<:Complex} = 
checkNeg(real(num1), real(num2)) && checkNeg(imag(num1), imag(num2))

#> Numbers of different types are first promoted (potentially leading to rounding for 
#> floating-point types), which can throw an `InexactError` if either cannot be converted.
function checkNeg(num1::RealOrComplex, num2::RealOrComplex)
    pNum1, pNum2 = promote(num1, num2)
    pNum1 == num1 && pNum2 == num2 && checkNeg(pNum1, pNum2)
end


#> Value-based check of `num1 == conj(num2)` where component-wise negation `-(x)` operation 
#> is avoided when possible to avoid edge cases like `-0xff === 0x01` (where the latter is 
#> the additive inverse of the former only when they are passed to `Base.:+`).
checkConj(num1::Real, num2::Real) = (num1 == num2)

function checkConj(num1::RealOrComplex, num2::RealOrComplex)
    re1, im1 = reim(num1)
    re2, im2 = reim(num2)
    re1 == re2 && checkNeg(im1, im2)
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


#> Conclude a Neumaier summation by adding the accumulated residue to the naive sum. The 
#> addition is skipped for a residue that is zero (keeping a signed zero intact) and for a 
#> residue that is not finite. The latter only arises when the naive sum is not finite 
#> either: such a sum has no rounding error to compensate for. As a result, the result of 
#> a Neumaier summation is consistent with the `Base.:+` for non-finite results. For 
#> instance: `neumaierSum([1e308, 1e308]) == 1e308 + 1e308 == Inf`
function absorbResidue(naiveSum::T, sumResidue::T) where {T<:AbstractFloat}
    isSkipped = iszero(sumResidue) | !isfinite(sumResidue)
    ifelse(isSkipped, naiveSum, naiveSum + sumResidue)
end

absorbResidue(naiveSum::C, sumResidue::C) where {C<:Complex{<:AbstractFloat}} = 
Complex(absorbResidue( real(naiveSum), real(sumResidue) ), 
        absorbResidue( imag(naiveSum), imag(sumResidue) ))

#> The residue of any other type stays zero (calling the fallback method of `neumaierAdd`)
absorbResidue(naiveSum::C, sumResidue::C) where {C<:RealOrComplex} = 
adaptAdd(naiveSum, sumResidue)


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

    absorbResidue(naiveSum, sumResidue)
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
