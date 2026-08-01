using Test
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg

@testset "Strings.jl" begin

m = "X"
@test (@pauli_str "$m") == pauli"X" == pauli"X"
@test (@pauli_str [2,3,2,0,1]) == pauli"XYXIZ" #> 0->I, 1->Z, 2->X, 3->Y

@test string(pauli"")   == "+I"
@test string(pauli"I")  == "+I"
@test string(pauli"II") == "+I"
@test toString(pauli"IIX") == "+X₃"
@test toString(pauli"IIX", true) == "IIX"
@test toString(pauli"XXIX", true) == "XXIX" #> Exact macro round-trip
@test toString(pauli"XXIX", true, omitPlusSign=false) == "+XXIX"
@test toString(pauli"X", omitPlusSign=true) == "X₁"
@test toString(PauliStr(2, symX, posImg)) == "+im*X₁X₂"
@test toString(PauliStr(2, symX, posImg), true) == "im*XX"
@test toString(PauliStr(3, symI, negRea)) == "-I"
@test toString(PauliStr(3, symI, negRea), true) == "-III"
@test toString(PauliStr(0, symI, negImg)) ==
      toString(PauliStr(0, symI, negImg), true) == "-im*I"

end