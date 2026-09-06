using Test
using Paulimorphic: isIndexLabel

@testset "Tensors.jl" begin

@testset "isIndexLabel" begin
    #> Default start (`iStart == first(candidate)`): anchor-free structural check
    @test  isIndexLabel((2,))
    @test  isIndexLabel((0,))
    @test  isIndexLabel((-7,))
    @test  isIndexLabel((3, 4, 3))
    @test  isIndexLabel((0, 1, 2))
    @test  isIndexLabel((1, 2, 2))
    @test  isIndexLabel((1, 1, 1))
    @test !isIndexLabel((3, 1))    #> Drop below the first element
    @test !isIndexLabel((2, 1, 2))
    @test !isIndexLabel((5, 7))    #> Skip an integer

    #> Explicit starts; the first element must equal `iStart`
    @test  isIndexLabel((1, 2, 1), 1)
    @test  isIndexLabel((1, 2, 3), 1)
    @test  isIndexLabel((2, 3, 2), 2)
    @test  isIndexLabel((0, 1, 0), 0)
    @test  isIndexLabel((-1, 0, -1), -1)
    @test !isIndexLabel((0,), 1)
    @test !isIndexLabel((2, 1), 1)
    @test !isIndexLabel((1, 2), 2)

    #> `Bool` elements participate through `true == 1` and `false == 0`
    @test  isIndexLabel((false, true))
    @test  isIndexLabel(( true, true), 1)
    @test !isIndexLabel((false, true), 1)
end

end
