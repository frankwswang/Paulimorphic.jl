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

end