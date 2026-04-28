using Test
using Paulimorphic

@testset "Strings.jl" begin

m = "X"
@test (@pauli_str "$m") == pauli"X" == pauli"X"
@test (@pauli_str [1,2,1,0,3]) == pauli"XYXIZ"

end