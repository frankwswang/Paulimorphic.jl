using Test
using Paulimorphic: neumaierStep, neumaierAdd, neumaierSum, evalMagnitude, extendType
using Paulimorphic: adaptAdd, adaptMul, adaptConj, adaptNeg, condNeg, toBitIntData
using Paulimorphic: checkNeg, checkConj, absorbResidue

@testset "Numerics.jl" begin
    @testset "adaptAdd / adaptMul / adaptConj / adaptNeg" begin
        #> Fixed-width signed integers (real, or as complex components) throw on overflow
        @test adaptAdd(typemax(Int) - 1, 1) === typemax(Int)
        @test_throws OverflowError adaptAdd(typemax(Int), 1)
        @test_throws OverflowError adaptAdd(Int8(100), Int8(100))
        @test adaptAdd(Complex(1, 2), Complex(3, 4)) === Complex(4, 6)
        @test_throws OverflowError adaptAdd(Complex(0, typemax(Int)), Complex(0, 1))
        @test adaptMul(2^31, 2^31) === 2^62
        @test_throws OverflowError adaptMul(2^32, 2^32)
        @test adaptMul(Complex(1, 2), Complex(3, 4)) === Complex(-5, 10)
        @test_throws OverflowError adaptMul(Complex(2^32, 0), Complex(0, 2^32))
        @test_throws OverflowError adaptMul(Complex(typemin(Int), 0), Complex(-1, 0))
        @test adaptConj(Complex(1, 2)) === Complex(1, -2)
        @test_throws OverflowError adaptConj(Complex(0, typemin(Int)))
        @test_throws OverflowError adaptConj(Complex(Int8(0), typemin(Int8)))
        @test adaptNeg(5) === -5 && adaptNeg(typemin(Int) + 1) === typemax(Int)
        @test_throws OverflowError adaptNeg(typemin(Int))
        @test_throws OverflowError adaptNeg(typemin(Int8))
        @test adaptNeg(Complex(1, -2)) === Complex(-1, 2)
        @test adaptNeg(Complex(typemax(Int), typemin(Int) + 1)) === 
              Complex(typemin(Int) + 1, typemax(Int))
        @test_throws OverflowError adaptNeg(Complex(typemin(Int), 0))
        @test_throws OverflowError adaptNeg(Complex(Int8(0), typemin(Int8)))

        #> Operands of different types are promoted first: an overflow is detected in the 
        #> promoted type, and a narrower result is left to the caller's conversion
        @test adaptMul(Int8(100), 2) === 200
        @test adaptMul(Complex(Int8(1), Int8(2)), 3) === Complex(3, 6)
        @test_throws OverflowError adaptMul(Int8(2), 2^62)
        @test adaptMul(2.0, 3) === 6.0

        #> Every other type keeps its own arithmetic (plain IEEE operations for floats)
        @test adaptAdd(1.0, 2.0) === 3.0 && adaptMul(1.5, 2.0) === 3.0
        @test adaptAdd(1e308, 1e308) === Inf
        @test adaptConj(1.0 + 2.0im) === 1.0 - 2.0im
        @test adaptNeg(1.5) === -1.5 && adaptNeg(0.0) === -0.0 #> Signed zero kept
        @test adaptNeg(1.0 + 2.0im) === -1.0 - 2.0im
        @test adaptMul(1//2, 2//3) === 1//3
        @test adaptNeg(1//2) === -1//2
        @test adaptNeg(Complex(1//2, -1//3)) === Complex(-1//2, 1//3)
        @test adaptAdd(big(typemax(Int)), big(1)) == big(typemax(Int)) + 1
        @test adaptNeg(big(typemin(Int))) == -big(typemin(Int))

        #> `@allocated` is evaluated twice, so that the first one absorbs the compilation.
        #> For the second evaluation, each adaptive operation should be allocation free.
        for x in (3, 1.5, 1//2, Complex(1, 2), 1.0 + 2.0im, Complex(1//2, -1//3))
            for f in (adaptAdd, adaptMul)
                @allocated f(x, x)
                @test (@allocated f(x, x)) == 0
            end
            for f in (adaptConj, adaptNeg)
                @allocated f(x)
                @test (@allocated f(x)) == 0
            end
        end
    end

    @testset "condNeg" begin
        #> `condNeg(num, true)` returns `adaptNeg(num)`; `condNeg(num, false)` returns `num`
        @test condNeg(5, true) === -5 && condNeg(5, false) === 5
        @test condNeg(0, true) === 0 && condNeg(-7, true) === 7
        @test condNeg(Complex(1, -2), true) === Complex(-1, 2)
        @test condNeg(Complex(1, -2), false) === Complex(1, -2)
        @test condNeg(1.5, true) === -1.5 && condNeg(1.5, false) === 1.5
        @test condNeg(0.0, true) === -0.0 && condNeg(-0.0, false) === -0.0
        @test condNeg(1.0 + 2.0im, true) === -1.0 - 2.0im
        @test condNeg(1//2, true) === -1//2 && condNeg(1//2, false) === 1//2
        @test condNeg(Complex(1//2, -1//3), true) === Complex(-1//2, 1//3)
        @test condNeg(big(typemin(Int)), true) == -big(typemin(Int))

        #> For every fixed-width signed integer type, the result is that of the negation up 
        #> to the limits of the type. The one value whose negation overflows throws the 
        #> error of `adaptNeg`, but only if it is to be negated
        for T in (Int8, Int16, Int32, Int64, Int128)
            for num in (typemin(T) + one(T), -one(T), zero(T), one(T), typemax(T))
                @test condNeg(num, true) === -num && condNeg(num, false) === num
                @test condNeg(Complex(num, -num), true) === Complex(-num, num)
            end

            low = typemin(T)
            @test condNeg(low, false) === low
            @test condNeg(Complex(low, low), false) === Complex(low, low)

            for num in (low, Complex(low, one(T)), Complex(one(T), low), Complex(low, low))
                message = try adaptNeg(num) catch err; sprint(showerror, err) end
                @test_throws OverflowError condNeg(num, true)
                @test_throws message condNeg(num, true)
            end
        end

        #> As for the other adaptive operations, the second evaluation allocates nothing
        for x in (3, 1.5, 1//2, Complex(1, 2), 1.0 + 2.0im, Complex(1//2, -1//3))
            for flag in (true, false)
                @allocated condNeg(x, flag)
                @test (@allocated condNeg(x, flag)) == 0
            end
        end
    end

    @testset "checkNeg / checkConj" begin
        #> `checkNeg(num1, num2)` judges `num1 == -num2` and `checkConj(num1, num2)` judges 
        #> `num1 == conj(num2)`, both at the value level
        @test  checkNeg(5, -5) && checkNeg(0, 0) && !checkNeg(5, 5) && !checkNeg(5, -4)
        @test  checkNeg(1.5, -1.5) && checkNeg(0.0, -0.0) && checkNeg(0.0, 0.0)
        @test  checkNeg(Inf, -Inf) && !checkNeg(Inf, Inf) && !checkNeg(NaN, NaN)
        @test  checkNeg(1//2, -1//2) && !checkNeg(1//2, 1//2) && !checkNeg(1//2, -1//3)
        @test  checkNeg(big(2)^70, -big(2)^70) && !checkNeg(big(2)^70, big(2)^70)
        @test  checkNeg(false, false) && !checkNeg(true, true) && !checkNeg(true, false)
        @test  checkNeg(Complex(1, -2), Complex(-1, 2))
        @test !checkNeg(Complex(1, -2), Complex(-1, -2))
        @test  checkConj(Complex(1, -2), Complex(1, 2))
        @test !checkConj(Complex(1, -2), Complex(1, -2))
        @test  checkConj(1.0 + 2.0im, 1.0 - 2.0im) && checkConj(1.0 + 0.0im, 1.0 - 0.0im)
        @test  checkConj(3, 3) && !checkConj(3, -3) && checkConj(1//2, 1//2)
        @test  checkConj(Complex(1//2, -1//3), Complex(1//2, 1//3))

        #> Where the opposite number (or the conjugate) is not representable, such that `-` 
        #> wraps around or throws, the judgment is `false` without any error
        for T in (Int8, Int16, Int32, Int64, Int128)
            low, top = typemin(T), typemax(T)
            @test -low === low && !checkNeg(low, low) && !checkNeg(top, top)
            @test  checkNeg(top, low + one(T)) && checkNeg(low + one(T), top)
            @test !checkConj(Complex(top, low), Complex(top, low))
            @test  checkConj(Complex(low, top), Complex(low, low + one(T)))

            ratLow = low // one(T)
            cmpLow = Complex(ratLow, ratLow)
            @test_throws OverflowError conj(cmpLow)
            @test !checkNeg(ratLow, ratLow) && !checkConj(cmpLow, cmpLow)
            @test  checkNeg(top // one(T), (low + one(T)) // one(T))
        end

        for T in (UInt8, UInt16, UInt32, UInt64, UInt128)
            top = typemax(T)
            @test -top === one(T) && !checkNeg(one(T), top) && !checkNeg(top, one(T))
            @test  checkNeg(zero(T), zero(T)) && !checkNeg(top, top)
            @test !checkConj(Complex(zero(T), one(T)), Complex(zero(T), top))
            @test  checkConj(Complex(top, zero(T)), Complex(top, zero(T)))
        end

        #> Every pair of `Int8` numbers and of `UInt8` numbers, against the same judgment 
        #> made within `Int`
        for T in (Int8, UInt8)
            vals = typemin(T):typemax(T)
            @test all(checkNeg(x, y) == (Int(x) == -Int(y)) for x in vals, y in vals)
        end

        #> Numbers of different types are first promoted, like in the adaptive operations, 
        #> so the promotion bounds them: a number that it cannot convert throws, and a pair 
        #> with a number that it rounds (to a floating-point type) is not opposite
        @test  checkNeg(Int8(-128), Int16(128)) && checkNeg(1, -1.0)
        @test !checkNeg(0xff, 0x0001) && !checkNeg(0x0001, 0xff)
        @test_throws InexactError checkNeg(0x01, Int8(-1))
        pow60 = Int64(2)^60 #> `pow60 + 1` is rounded to `2.0^60` as a `Float64`
        @test Float64(pow60 + 1) === 2.0^60
        @test  checkNeg(pow60, -2.0^60) && !checkNeg(pow60 + 1, -2.0^60)
        @test !checkConj(Complex(1, pow60 + 1), Complex(1.0, -2.0^60))
        @test  checkConj(Complex{Int8}(1, -128), Complex{Int16}(1, 128))
        @test  checkConj(2.0, Complex(2, 0)) && !checkConj(2.0, Complex(2, 1))

        #> As for the adaptive operations, the second evaluation allocates nothing
        for x in (3, 1.5, 1//2, Complex(1, 2), 1.0 + 2.0im, Complex(1//2, -1//3))
            for f in (checkNeg, checkConj)
                @allocated f(x, x)
                @test (@allocated f(x, x)) == 0
            end
        end
    end

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
        #> Exact fallback: (overflow-checked) plain addition with an untouched residue slot
        @test neumaierAdd(1//2, 1//3, 0//1) == neumaierAdd(1//2, 1//3) == (5//6, 0//1)
        @test neumaierAdd(Complex(1, 2), Complex(3, 4), Complex(0, 0)) == 
                         (Complex(4, 6), Complex(0, 0))
        @test_throws OverflowError neumaierAdd(typemax(Int), 1)
        @test_throws OverflowError neumaierAdd(Complex(typemax(Int), 0), Complex(1, 0))
        @test neumaierAdd(big(typemax(Int)), big(1)) == (big(typemax(Int)) + 1, big(0))
    end

    @testset "absorbResidue" begin
        #> The residue is added unless it is zero (so that a signed zero stays intact) or 
        #> not finite (which it only is when the naive sum is not finite either)
        @test absorbResidue(0.0, 1.0) === 1.0
        @test absorbResidue(1.0, 2.0^-52) === 1.0 + 2.0^-52
        @test absorbResidue(-0.0, 0.0) === -0.0 && absorbResidue(0.0, -0.0) === 0.0
        @test absorbResidue(Inf, -Inf) === Inf && absorbResidue(-Inf, NaN) === -Inf
        @test isnan(absorbResidue(NaN, 1.0)) && isnan(absorbResidue(NaN, NaN))
        @test absorbResidue(ComplexF64(Inf, -0.0), ComplexF64(-Inf, 0.0)) === 
              ComplexF64(Inf, -0.0)
        @test absorbResidue(ComplexF64(0.0, Inf), ComplexF64(1.0, NaN)) === 
              ComplexF64(1.0, Inf)
        #> Exact fallback: the residue slot of any other type stays zero
        @test absorbResidue(5//6, 0//1) === 5//6 && absorbResidue(3, 0) === 3
        @test absorbResidue(Complex(4, 6), Complex(0, 0)) === Complex(4, 6)
    end

    @testset "neumaierSum" begin
        @test neumaierSum([1e16, 1.0, -1e16]) == 1.0
        @test neumaierSum([1e16, 1.0, -1e16, -1.0]) == 0.0
        @test neumaierSum(ComplexF64[1e16+im, 1.0+im, -1e16-2im]) == ComplexF64(1.0, 0.0)
        @test neumaierSum(Float64[]) == 0.0
        #> A sum that is not finite is the one that plain additions give, instead of a 
        #> `NaN` made by a residue that is not finite either
        @test neumaierSum([1e308, 1e308]) === 1e308 + 1e308 === Inf
        @test neumaierSum([Inf, 1.0]) === Inf && neumaierSum([1.0, -Inf, 2.0]) === -Inf
        @test isnan(neumaierSum([Inf, -Inf])) && isnan(neumaierSum([1.0, NaN]))
        @test neumaierSum(ComplexF64[1e308 + 1.0im, 1e308 + 1e16im, -1e16im]) === 
              ComplexF64(Inf, 1.0)
        @test_throws ArgumentError neumaierSum(AbstractFloat[1.0, 2.0])

        @test neumaierSum(Float64, Iterators.filter(>(1), [0.5, 1.5])) == 1.5
        @test neumaierSum(Float64, skipmissing([1e16, missing, 1.0, -1e16])) == 1.0
        @test neumaierSum(abs, Float64, [-1.0, 2.0]) == 3.0
        @test_throws OverflowError neumaierSum([typemax(Int), 1])
        @test neumaierSum(Int128, [typemax(Int), 1]) == Int128(typemax(Int)) + 1
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

    @testset "extendType" begin
        #> Same category (both floating-point or both exact): promoted; otherwise, `T`
        @test extendType(Float32, Float64) === extendType(Float64, Float32) === Float64
        @test extendType(Int16, Rational{Int8}) === Rational{Int16}
        @test extendType(Int, Float64) === Float64
        @test extendType(Float64, Rational{Int16}) === Rational{Int16}
        #> A complex `T` has its real and imaginary parts extended alike
        @test extendType(Int16, Complex{Rational{Int8}}) === Complex{Rational{Int16}}
        @test extendType(Float64, Complex{Int}) === Complex{Int}
        @test extendType(Float64, ComplexF64) === ComplexF64

        #> The returned type sets the element type of accumulation buffers (e.g., in the 
        #> constructors of `PauliSum` and in `mul`), so every method must be inferable
        @test (@inferred extendType(Int, Float64)) === Float64
        @test (@inferred extendType(Float64, ComplexF64)) === ComplexF64
        @test (@inferred extendType(Float32, Complex{BigFloat})) === Complex{BigFloat}
    end

    @testset "toBitIntData" begin
        @test toBitIntData(UInt8(3)) === 3
        @test toBitIntData(true) === 1
        @test toBitIntData(Int8, 3) === Int8(3)
        @test toBitIntData(UInt(1):UInt(3)) === 1:3
        @test toBitIntData(Int8, 5:4) === Int8(5):Int8(4) #> Empty range
        @test_throws DomainError toBitIntData(Int8, 128)
        @test_throws DomainError toBitIntData(typemax(UInt))
        @test_throws DomainError toBitIntData(big(2)^70)
        @test_throws DomainError toBitIntData(Int8, 1:200)

        #> The error message refers to the unrepresentable value by the provided names
        @test_throws "`val`" toBitIntData(Int8, 128)
        @test_throws "`nSite`" toBitIntData(Int8, 128, :nSite)
        @test_throws "`first(posPair)`" toBitIntData(Int8, 128, :posPair, :first)
        @test_throws "`nSite`" toBitIntData(typemax(UInt), :nSite)
        @test_throws "`last(posPair)`" toBitIntData(typemax(UInt), :posPair, :last)
        @test_throws "`first(rng)`" toBitIntData(Int8, 200:300)
        @test_throws "`last(rng)`" toBitIntData(Int8, 1:200)
        @test_throws "`first(srcRange)`" toBitIntData(Int8, 200:300, :srcRange)
        @test_throws "`last(srcRange)`" toBitIntData(UInt(1):typemax(UInt), :srcRange)

        #> `@allocated` is evaluated twice, so that the first one absorbs the compilation. 
        #> For the second evaluation, converting a representable range should be 
        #> allocation free (i.e., no part of the error message is prepared in advance).
        for rng in (2:5, UInt(2):UInt(5))
            @allocated toBitIntData(rng, :srcRange)
            @test (@allocated toBitIntData(rng, :srcRange)) == 0
        end
    end
end
