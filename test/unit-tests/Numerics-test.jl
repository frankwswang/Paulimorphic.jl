using Test
using Paulimorphic: neumaierStep, neumaierAdd, neumaierSum, evalMagnitude

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

        @test neumaierSum(Float64, Iterators.filter(>(1), [0.5, 1.5])) == 1.5
        @test neumaierSum(Float64, skipmissing([1e16, missing, 1.0, -1e16])) == 1.0
        @test neumaierSum(abs, Float64, [-1.0, 2.0]) == 3.0
        @test_throws ArgumentError neumaierSum(Complex, ComplexF64[1.0im])
        @test_throws ArgumentError let scope = (x for x in Iterators.countfrom(1.0))
            neumaierSum(Float64, scope)
        end
    end

    @testset "evalMagnitude" begin
        #> Exact cases, including the real and imaginary axes (where it equals `abs`)
        @test evalMagnitude(ComplexF64(3, 4)) == evalMagnitude(ComplexF64(-4, 3)) == 5
        @test evalMagnitude(ComplexF64(-2.0, 0)) == evalMagnitude(ComplexF64(0, -2.0)) == 2
        @test evalMagnitude(ComplexF64(0, 0)) == 0.0

        #> `Inf` and `NaN` handling
        @test evalMagnitude(ComplexF64(Inf, NaN)) == Inf
        @test evalMagnitude(ComplexF64(-Inf,  1)) == Inf
        @test isnan(evalMagnitude(ComplexF64(NaN, 1)))
        @test isnan(evalMagnitude(ComplexF64(1, NaN)))
        @test_throws DomainError evalMagnitude(complex(Inf, NaN), false)
        @test_throws DomainError evalMagnitude(complex(NaN, 1.0), false)

        #> No spurious overflow or underflow of the squared components
        @test evalMagnitude(ComplexF64(1e300, 1e300)) == 1e300 * sqrt(2.0)
        @test evalMagnitude(ComplexF64(1e-300, 1e-300)) == Float64(sqrt(big(2.0)) * 1e-300)

        #> Generic values: within a few ulps of `abs` (`hypot`), whose last bit is 
        #> hardware-dependent, whereas `evalMagnitude` is reproducible everywhere
        for c in (0.23665594309283294 + 0.03906366194959983im, 
                  0.006570505490472972 - 0.01410983396709257im, 1.0 - 1.0im, 1f0 + 2f0im)
            @test abs(evalMagnitude(c) - abs(c)) <= 2eps(abs(c))
        end

        #> Exact coefficient types are converted to floating-point numbers first
        @test evalMagnitude(Complex(3//1, 4//1)) === evalMagnitude(Complex(3, 4)) === 5.0

        #> Data types besides `ComplexF64`
        @test evalMagnitude(ComplexF16(200, 150)) == Float16(250)
        @test evalMagnitude(ComplexF32(3, 4)) == 5f0
        @test evalMagnitude(3) == 3.0
        @test evalMagnitude(-3.0f0) == 3.0f0
    end
end
