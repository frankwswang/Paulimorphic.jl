using Test
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg

@testset "Constants.jl" begin

#> `evalPhase`: inferred return type and exact values (`===` pins `Complex{Int}` and value)
@test (@inferred evalPhase(posRea)) === Complex( 1,  0)
@test (@inferred evalPhase(posImg)) === Complex( 0,  1)
@test (@inferred evalPhase(negRea)) === Complex(-1,  0)
@test (@inferred evalPhase(negImg)) === Complex( 0, -1)

end
