using Test
using Paulimorphic: isIndexLabel, dumpTo!, PauliSum, @pauli_str

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

@testset "dumpTo!" begin
    #> Minimal zero-based vector (axes start at 0) for testing the axis-mismatch tolerance
    struct ZeroBasedVector{T} <: AbstractVector{T}
        data::Vector{T}
    end

    Base.size(v::ZeroBasedVector) = size(v.data)
    Base.axes(v::ZeroBasedVector) = (Base.IdentityUnitRange(0:(length(v.data) - 1)),)
    Base.getindex(v::ZeroBasedVector, i::Int) = v.data[begin+i]

    src = ZeroBasedVector([1.0, 2.0])
    @test firstindex(src) == 0 #> Sanity check: `src` is genuinely offset-indexed
    @test_throws DimensionMismatch [0, 0] .= src #> `dst .= src` fails when `length(src)>1`

    dst = zeros(3)
    @test dumpTo!(dst, src) === dst
    @test dst == [src..., 0.0]
    @test dumpTo!(dst, src, 2) == [1.0, 1.0, 2.0] #> Destination offset via `iStart`
    @test dumpTo!(dst, 7.0) == fill(7.0, 3)       #> Scalar source fills the whole array

    #> Offset coefficient vectors flow through both `PauliSum` constructor branches
    refSum = PauliSum([pauli"X", pauli"Y"], [0.5, 0.25])
    @test PauliSum(Float64, [pauli"X", pauli"Y"], ZeroBasedVector([0.5, 0.25])) == refSum
    @test PauliSum([pauli"X", pauli"Y"], ZeroBasedVector([0.5, 0.25]), false) == refSum
end

end
