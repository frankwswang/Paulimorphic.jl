export PauliSym, PauliX, PauliY, PauliZ, PhaseFactor

const PauliXMatEntries = (0,   1,    1,  0) #> (m11, m21, m12, m22)
const PauliYMatEntries = (0, 1im, -1im,  0)
const PauliZMatEntries = (1,   0,    0, -1)


@enum PauliSym::UInt8 begin
    symI=0
    symX=1
    symY=2
    symZ=3
end


function getPauliSym(ele::Char)
    num = if ele in 'I'
        0
    elseif ele == 'X'
        1
    elseif ele == 'Y'
        2
    elseif ele == 'Z'
        3
    else
        throw(ArgumentError("\'$ele\' is not a valid letter for a Pauli operator"))
    end

    getPauliSym(num)
end

getPauliSym(num::Integer) = PauliSym(num)


struct PauliX{L<:Unsigned} <: DiscreteOperator
    label::L
end

struct PauliY{L<:Unsigned} <: DiscreteOperator
    label::L
end

struct PauliZ{L<:Unsigned} <: DiscreteOperator
    label::L
end


@enum PhaseFactor::UInt8 begin
    negImg=0 #> 00: -im
    negRea=1 #> 01: -1
    posImg=2 #> 10: +im
    posRea=3 #> 11: +1
end


const CONSTVAR!!subscriptNum = 
      Dict([0=>'₀', 1=>'₁', 2=>'₂', 3=>'₃', 4=>'₄', 5=>'₅', 6=>'₆', 7=>'₇', 8=>'₈', 9=>'₉'])