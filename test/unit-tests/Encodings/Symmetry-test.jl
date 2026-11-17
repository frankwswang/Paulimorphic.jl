using Test
using Paulimorphic
using LinearAlgebra: Hermitian, Symmetric

@testset "Symmetry.jl" begin

#> Shared fixtures
h1a = [1.0 0.5-im; 0.5+im 2.0] #> Complex Hermitian
h1s = [1.0 0.5; 0.5 2.0]       #> Real symmetric
v = [1.0 0.25; 0.25 0.5]
w = [2.0 0.5 0.0; 0.5 1.0 0.25; 0.0 0.25 3.0]
g2 = [v[i, j] * v[m, n] for i in 1:2, j in 1:2, m in 1:2, n in 1:2] #> 8-fold symmetric
gvw = [v[i, j] * w[m, n] for i in 1:2, j in 1:2, m in 1:3, n in 1:3] #> 4-fold cross tensor

gB = zeros(ComplexF64, 2, 2, 2, 2)
gB[1, 2, 1, 2] =  im
gB[2, 1, 2, 1] = -im

gD = zeros(2, 2, 2, 2) #> Real, Hermitian, exchange symmetric; pair transposition fails
gD[1, 2, 1, 2] = 1.0
gD[2, 1, 2, 1] = 1.0

gF = zeros(2, 2, 2, 2) #> Hermitian and pair transposable, but not exchange symmetric
gF[1, 2, 1, 1] = 1.0
gF[2, 1, 1, 1] = 1.0

@testset "checkNBodyInteTensor" begin
    @test  checkNBodyInteTensor(g2, (2, 2), (1, 1), (true, true))
    @test  checkNBodyInteTensor(gvw, (2, 3), (1, 2), (true, true))
    @test_throws ArgumentError checkNBodyInteTensor(gD, (2, 2), (1, 1), (false, true))
    @test !checkNBodyInteTensor(gD, (2, 2), (1, 1), (true, true))
    @test  checkNBodyInteTensor(gD, (2, 2), (1, 1), (false, false))
    @test !checkNBodyInteTensor(gF, (2, 2), (1, 1), (false, false))
    @test  checkNBodyInteTensor(gF, (2, 2), (1, 1), (false, false); particleExch=false)
    @test  checkNBodyInteTensor(gB, (2, 2), (1, 1), (false, false))
    @test !checkNBodyInteTensor(Hermitian(h1a), (2,), (1,), (true,))
    @test  checkNBodyInteTensor(Symmetric(h1s), (2,), (1,), (true,))
end

@testset "Particle positions" begin
    #> Both checkers convert the positions to `Int` before examining their ranges. Hence, 
    #> a position that is not representable by `Int` is reported as such by either of them.
    bigPos = big(2)^70
    @test_throws DomainError checkIntePairTransposition(g2, bigPos)
    @test_throws "`pos` must also be representable" checkIntePairTransposition(g2, bigPos)
    @test_throws DomainError checkInteParticleExchange(g2, (1, bigPos))
    @test_throws "`last(posPair)` must also" checkInteParticleExchange(g2, (1, bigPos))

    #> A representable position must be within `1:2`, since `g2` holds two particles
    @test_throws DomainError checkIntePairTransposition(g2, 3)
    @test_throws "`pos` must be within `1:2`" checkIntePairTransposition(g2, 3)
    @test_throws DomainError checkInteParticleExchange(g2, (0, 2))
    @test_throws "`posPair` must be within `1:2`" checkInteParticleExchange(g2, (0, 2))

    #> The integer type of an in-range position does not affect the result
    @test  checkIntePairTransposition(g2, UInt8(2))
    @test !checkIntePairTransposition(gD, big(1))
    @test  checkInteParticleExchange(g2, (2, 0x01))
    @test !checkInteParticleExchange(gF, (big(1), 2))
end

end
