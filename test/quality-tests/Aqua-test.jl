using Test
using Aqua
using Paulimorphic

@testset "Aqua.jl-Required Test" begin
    Aqua.test_all(Paulimorphic)
end
