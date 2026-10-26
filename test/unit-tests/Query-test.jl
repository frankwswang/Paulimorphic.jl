using Test
using Paulimorphic: MemoryChain, genMemory, updateNeuAccumulator!, NeuAccumulator

@testset "Query.jl" begin
    @testset "MemoryChain" begin
        m1 = genMemory([1, 2, 3])
        m2 = genMemory([4.0, 5.0])
        chain = MemoryChain(m1, m2)
        @test chain isa AbstractVector{Float64}
        @test length(chain) == 5
        @test chain[1] === 1.0 && chain[3] === 3.0 && chain[4] === 4.0 && chain[end] === 5.0
        @test_throws BoundsError chain[0]
        @test_throws BoundsError chain[6]
        #> Iteration must visit every element exactly once, including the last one
        @test collect(chain) == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test [x for x in chain] == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test sum(chain) == 15.0
        @test maximum(chain) == 5.0 && 5.0 in chain
        #> Writing through the chain modifies the underlying segment
        chain[4] = 9.0
        @test m2[1] == 9.0
        #> Explicit element type and conversions
        @test collect(MemoryChain{Int}(m1)) == [1, 2, 3]
        @test_throws ArgumentError MemoryChain{Int}(genMemory(["a"]))
        @test_throws InexactError MemoryChain{Int}(genMemory([0.5]))[1]
        #> Empty chains
        @test isempty(MemoryChain{Int}()) && collect(MemoryChain{Int}()) == Int[]
        @test_throws BoundsError MemoryChain{Int}()[1]
    end

    @testset "genMemory" begin
        @test genMemory([1 2; 3 4]) == Memory{Int}([1, 3, 2, 4])
        @test genMemory(Float64, [1, 2]) isa Memory{Float64}
        @test genMemory(3) isa Memory{Int} && genMemory(3)[] == 3
    end

    @testset "updateNeuAccumulator!" begin
        acc = NeuAccumulator{Symbol, Float64}()
        @test updateNeuAccumulator!(acc, :a=>1e16)
        @test !updateNeuAccumulator!(acc, :a=>1.0)
        @test !updateNeuAccumulator!(acc, :a=>-1e16)
        @test sum(acc[:a]) == 1.0 #> Naive accumulation would give 0.0
    end
end
