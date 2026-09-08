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
(total + val, residue)

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
