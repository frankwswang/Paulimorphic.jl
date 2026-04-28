export PauliStr, PauliSum, @pauli_str

using LinearAlgebra: dot
import Base: hash, ==, show

struct PauliStr <: DiscreteOperator
    zStr::Memory{Bool}
    xStr::Memory{Bool}
    phase::PhaseFactor
    weight::UInt

    function PauliStr(zStr::AbstractVector{Bool}, xStr::AbstractVector{Bool})
        if length(zStr) != length(xStr)
            throw(ArgumentError("`zStr` and `xStr` should have the same length."))
        end

        flipCount = dot(zStr, xStr)
        phase = PhaseFactor((flipCount - 1) & 3)
        weight = mapreduce(|, +, zStr, xStr)
        new(convert(Memory{Bool}, zStr), convert(Memory{Bool}, xStr), phase, weight)
    end
end

function hash(pStr::PauliStr, hashCode::UInt)
    code = hash(pStr.phase, hashCode)
    hash(pStr.zStr, hash(pStr.xStr, code))
end

function ==(pStr1::PauliStr, pStr2::PauliStr)
    (pStr1.phase == pStr2.phase) && 
    (pStr1.xStr  == pStr2.xStr ) && 
    (pStr1.zStr  == pStr2.zStr )
end

function PauliStr(list::AbstractVector{PauliSym})
    len = length(list)
    zStr = Memory{Bool}(undef, len)
    xStr = Memory{Bool}(undef, len)

    for i in 1:len
        ele = list[begin+i-1]

        isZ, isX = if ele == symI
            false, false
        elseif ele == symZ
             true, false
        elseif ele == symX
            false,  true
        else
             true,  true
        end

        xStr[begin+i-1] = isX
        zStr[begin+i-1] = isZ
    end

    PauliStr(zStr, xStr)
end

PauliStr(siteNum::Integer=0, siteOp::PauliSym=symI) = PauliStr(fill(siteOp, siteNum))


function getPauliSymVec(str::Union{String, AbstractVector{<:Integer}})
    [getPauliSym(c::Union{Char, Integer}) for c in str]
end

macro pauli_str(ex)
    strOrVecExpr = esc(ex)
    :(getPauliSymVec($strOrVecExpr) |> PauliStr)
end


function printStr(pStr::PauliStr)
    if isempty(pStr.zStr) || iszero(pStr.weight)
        "I"
    else
        i = 0
        str = ""

        for (z, x) in zip(pStr.zStr, pStr.xStr)
            i += 1
            str *= if z && x
                'Y' * CONSTVAR!!subscriptNum[i]
            elseif z && !x
                'Z' * CONSTVAR!!subscriptNum[i]
            elseif x && !z
                'X' * CONSTVAR!!subscriptNum[i]
            else
                ""
            end
        end

        str
    end
end

Base.show(io::IO, pStr::PauliStr) = print(io, printStr(pStr))


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