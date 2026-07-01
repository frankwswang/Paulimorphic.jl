using Test
using Paulimorphic

@testset "hamiltonians.jl" begin

n = Int32(6)

@testset "Encoding Hamiltonian (JW + Ternary, random)" begin
    encodings = [Jordan_Wigner_encoding(n), Ternary_Tree_encoding(n)]
    ham = hamiltonian(n, encodings; coeff_type=:random)
    @test !isempty(ham.str)
    @test length(ham.coeff) == length(ham.str)
    @test all(s -> s.n == n, ham.str)
end

@testset "4-regular Hamiltonian (uniform)" begin
    ham = hamiltonian(n, :FourRegularGraph; coeff_type=:uniform)

    @test !isempty(ham.str)
    @test length(ham.coeff) == length(ham.str)
    @test all(c -> isone(c), ham.coeff)
end

@testset "2-local Hamiltonian (zeropmone)" begin
    ham = hamiltonian(n, :TwoLocal; coeff_type=:zeropmone)

    @test !isempty(ham.str)
    @test length(ham.coeff) == length(ham.str)
    @test all(c -> (imag(c) == 0) && (real(c) in (-1.0, 0.0, 1.0)), ham.coeff)
end

@testset "Kagome Hamiltonian (normal)" begin
    ham = hamiltonian(n, :Lattice; type2=:Kagome2D, coeff_type=:normal)

    @test !isempty(ham.str)
    @test length(ham.coeff) == length(ham.str)
    @test all(s -> s.n == n, ham.str)
end

end
