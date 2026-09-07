using Test
using Paulimorphic: neumaierStep, neumaierAdd, neumaierSum

@testset "Numerics.jl" begin
    @testset "neumaierStep" begin
        @test neumaierStep(1e16, 1.0) == (1e16, 1.0)
        @test neumaierStep(1.0, 1e16) == (1e16, 1.0)
        @test neumaierStep(ComplexF64(1e16, 1.0), ComplexF64(1.0, 1e16)) == 
              (ComplexF64(1e16, 1e16), ComplexF64(1.0, 1.0))
    end

    @testset "neumaierAdd" begin
        #> Residue-step methods: the float path collects the addition residue
        @test neumaierAdd(1e16, 1.0, 0.0) == neumaierAdd(1e16, 1.0) == (1e16, 1.0)
        @test neumaierAdd(1e16, -1e16, 1.0) == (0.0, 1.0)
        #> Exact fallback: plain addition with an untouched residue slot
        @test neumaierAdd(1//2, 1//3, 0//1) == neumaierAdd(1//2, 1//3) == (5//6, 0//1)
        @test neumaierAdd(Complex(1, 2), Complex(3, 4), Complex(0, 0)) == 
                         (Complex(4, 6), Complex(0, 0))
    end

    @testset "neumaierSum" begin
        @test neumaierSum([1e16, 1.0, -1e16]) == 1.0
        @test neumaierSum([1e16, 1.0, -1e16, -1.0]) == 0.0
        @test neumaierSum(ComplexF64[1e16+im, 1.0+im, -1e16-2im]) == ComplexF64(1.0, 0.0)
        @test neumaierSum(Float64[]) == 0.0
        @test_throws ArgumentError neumaierSum(AbstractFloat[1.0, 2.0])
    end
end
