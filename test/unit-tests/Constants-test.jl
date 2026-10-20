using Test
using Paulimorphic
using Paulimorphic: PauliCharacters, posRea, posImg, negRea, negImg

@testset "Constants.jl" begin

#> `evalPhase`: inferred return type and exact values (`===` pins `Complex{Int}` and value)
@test (@inferred evalPhase(posRea)) === Complex( 1,  0)
@test (@inferred evalPhase(posImg)) === Complex( 0,  1)
@test (@inferred evalPhase(negRea)) === Complex(-1,  0)
@test (@inferred evalPhase(negImg)) === Complex( 0, -1)

#> The bindings between Pauli characters and their respective integer codes
@test map(x->PauliCharacters[begin+Int(x)], instances(PauliSym)) == PauliCharacters

#> `toPhase`
for p in instances(PhaseFactor)
    @test toPhase(evalPhase(p)) == p
    @test toPhase(toString(p)) == toPhase(toString(p; omitPlusSign=true)) == p
end
@test toPhase(-1.0) == negRea && toPhase(Complex{Float64}(0, -1)) == negImg
@test_throws ArgumentError toPhase(2)
@test_throws ArgumentError toPhase(1 + 1im)

end
