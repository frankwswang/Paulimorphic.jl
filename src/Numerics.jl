#>-- Reference(s) --<#
#> [DOI] 10.1002/zamm.19740540106 (Improved Kahan--Babuska algorithm proposed by Neumaier)
@inline function neumaierAdd(x1::T, x2::T) where {T<:AbstractFloat}
    total = x1 + x2
    residue = ifelse(abs(x1) < abs(x2), (x2 - total) + x1, (x1 - total) + x2)
    (total, residue)
end

function neumaierAdd(x1::FloatOrComplex{T}, x2::FloatOrComplex{T}) where {T<:AbstractFloat}
    rTotal, rResidue = neumaierAdd(real(x1), real(x2))
    iTotal, iResidue = neumaierAdd(imag(x1), imag(x2))
    (Complex(rTotal, iTotal), Complex(rResidue, iResidue))
end

function neumaierSum(vals::AbstractVector{<:FloatOrComplex})
    eleT = eltype(vals)
    T = eleT <: AbstractFloat ? eleT : typeintersect(eleT, Complex)
    if !(isconcretetype(T) && (isconcretetype∘real)(T))
        throw(ArgumentError("The Neumaier sum of `vals::$(typeof(vals))` is not "*
                            "well-defined."))
    end

    accumulation = zero(T)
    compensation = zero(T)

    for val in vals
        accumulation, correction = neumaierAdd(accumulation, val)
        compensation += correction
    end

    accumulation + compensation
end