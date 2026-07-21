using Test
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg

@testset "Strings.jl" begin

m = "X"
@test (@pauli_str "$m") == pauli"X" == pauli"X"
@test (@pauli_str [1,2,1,0,3]) == pauli"XYXIZ"

@test string(pauli"")   == "+I"
@test string(pauli"I")  == "+I"
@test string(pauli"II") == "+I"
@test toString(pauli"IIX") == "+X₃"
@test toString(pauli"IIX", denseString=true) == "IIX"
@test toString(pauli"XXIX", denseString=true) == "XXIX" #> Exact macro round-trip
@test toString(pauli"XXIX", denseString=true, omitPlusSign=false) == "+XXIX"
@test toString(pauli"X", omitPlusSign=true) == "X₁"
@test toString(PauliStr(2, symX, posImg)) == "+im*X₁X₂"
@test toString(PauliStr(2, symX, posImg), denseString=true) == "im*XX"
@test toString(PauliStr(3, symI, negRea)) == "-I"
@test toString(PauliStr(3, symI, negRea), denseString=true) == "-III"
@test toString(PauliStr(0, symI, negImg)) ==
      toString(PauliStr(0, symI, negImg), denseString=true) == "-im*I"

end