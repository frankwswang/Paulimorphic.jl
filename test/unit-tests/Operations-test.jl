using Test
using Paulimorphic

@testset "Operations.jl" begin

str1 = pauli"XXIZ"
str2 = pauli"XXIY"
H1 = PauliSum([str1, str2])
H2 = PauliSum([str1, str2])
@test checkAntiCom(str1, str2)
@test H1 == H2

@test string(pauli"") == "I"
@test string(pauli"I") == "I"
@test string(pauli"II") == "I"
@test string(pauli"IIX") == "X₃"

@test checkCommute(pauli"IX", pauli"IX")
@test checkCommute(pauli"IX", pauli"I")
@test checkCommute(pauli"IX", pauli"I")

end