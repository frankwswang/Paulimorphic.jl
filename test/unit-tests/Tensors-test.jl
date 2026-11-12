using Test
using Random: Xoshiro
using LinearAlgebra: kron
using SparseArrays: SparseMatrixCSC, nnz, rowvals, nzrange
using Paulimorphic
using Paulimorphic: isIndexLabel, dumpTo!, posRea, posImg, negRea, negImg, setCoeff!

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

@testset "toMatrix" begin
    syms = [symI, symX, symY, symZ]
    phases = [posRea, posImg, negRea, negImg]
    rng = Xoshiro(1234)
    randStr(nSite::Int) = PauliStr([rand(rng, syms) for _ in 1:nSite], rand(rng, phases))

    #> Independent dense reference of the documented basis convention: site `k` is the 
    #> `k`-th factor from the right of the Kronecker product (least significant bit first)
    function kronRef(str::PauliStr, nSite::Int=countSites(str))
        mats = [Matrix(toMatrix(indexSite(str, k, true))) for k in nSite:-1:1]
        evalPhase(str) .* foldl(kron, mats; init=ones(Complex{Int}, 1, 1))
    end
    function sumRef(h::PauliSum{T}, nSite::Int=countSites(h)) where {T}
        res = zeros(Complex{T}, 2^nSite, 2^nSite)
        for (str, coeff) in listTerms(h)
            res .+= coeff .* kronRef(str, nSite)
        end
        res
    end

    #> Even for single-site Pauli operators, their matrices should be sparse
    @test all(nnz(sym|>toMatrix)==2 for sym in syms)

    #> Every string on up to three sites with every phase: a signed permutation matrix
    for nSite in 0:3, code in 0:(4^nSite - 1), phase in phases
        str = PauliStr(PauliSym[syms[begin + ((code >> (2k - 2)) & 3)] for k in 1:nSite], 
                       phase)
        mat = toMatrix(str)
        bl = true
        bl *= mat isa SparseMatrixCSC{Complex{Int}, Int}
        bl *= size(mat) == (2^nSite, 2^nSite) && nnz(mat) == 2^nSite
        bl *= Matrix(mat) == kronRef(str)
        bl *= toMatrix(Float64, str) == mat
        bl *= toMatrix(Float64, str) isa SparseMatrixCSC{ComplexF64, Int}
        @test bl
    end
    @test Matrix(toMatrix(pauli"-")) == fill(-1, 1, 1) #> Zero-site string: its phase
    @test Matrix(toMatrix(pauli"XI")) == [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]
    @test Matrix(toMatrix(pauli"IX")) == [0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]

    #> Random sums: duplicates, zero coefficients, and phases exercise the merging in the 
    #> constructor, the phase folding, and the dropping of exactly cancelled entries
    bl1 = bl2 = bl3 = bl4 = bl5 = true
    for _ in 1:300
        nSite = rand(rng, 1:5)
        nTerm = rand(rng, 1:8)
        strs = [randStr(nSite) for _ in 1:nTerm]
        coeffs = [Complex{Int}(rand(rng, -3:3), rand(rng, -3:3)) for _ in 1:nTerm]
        h = PauliSum(strs, coeffs)
        n = countSites(h) #> `0` when every term cancels
        ref = sumRef(h)
        mat = toMatrix(h)
        bl1 *= size(mat) == (2^n, 2^n) && Matrix(mat) == ref
        bl2 *= nnz(mat) == count(!iszero, ref) #> No stored zeros and no missing entries
        bl3 *= all(j->issorted(rowvals(mat)[nzrange(mat, j)]), 1:2^n) #> Valid CSC layout
        bl4 *= toMatrix!(zeros(Complex{Int}, 2^n, 2^n), h) == ref
        bl5 *= toMatrix(Float64, h) == ref
    end
    @test bl1
    @test bl2
    @test bl3
    @test bl4
    @test bl5

    #> Floating-point coefficients against the exact evaluation of the same terms: the 
    #> exact entries are accumulated as `Complex{Rational{Int}}` (as `extendType(Float64, 
    #> Rational{Int}) == Rational{Int}`) and rounded to `ComplexF64` only at the end; 
    #> comparing rational-valued matrices directly would overflow inside `norm` (`abs2` 
    #> squares the numerators, which can be as large as `2^53`)
    bl6 = bl7 = true
    for _ in 1:100
        nSite = rand(rng, 1:4)
        nTerm = rand(rng, 1:6)
        strs = [randStr(nSite) for _ in 1:nTerm]
        coeffs = [Complex(rand(rng) - 0.5, rand(rng) - 0.5) for _ in 1:nTerm]
        h = PauliSum(strs, coeffs)
        n = countSites(h)
        hExact = PauliSum(Rational{Int}, h) #> Exact conversion of the binary fractions
        bl6 *= toMatrix(h) ≈ toMatrix(Float64, hExact)
        bl7 *= toMatrix!(zeros(ComplexF64, 2^n, 2^n), h) ≈ 
               toMatrix!(zeros(ComplexF64, 2^n, 2^n), hExact)
    end
    @test bl6
    @test bl7

    #> Core type of the result: the entries are accumulated at the precision level of 
    #> `extendType(T, T1)` and converted to `Complex{T}` last
    hInt = PauliSum([pauli"XI", pauli"IZ"], [1, 2])
    @test toMatrix(Float32, hInt) isa SparseMatrixCSC{ComplexF32, Int}
    @test toMatrix(Float32, hInt) == toMatrix(hInt) == toMatrix(Rational{Int}, hInt)
    @test_throws ArgumentError toMatrix(Signed, hInt) #> Not a concrete core type
    @test_throws MethodError toMatrix(Bool, hInt)
    @test_throws InexactError toMatrix(Int, PauliSum([pauli"X"], [0.5]))
    #>> Narrowing to `Float32` rounds the entries to exact zero, which are then not stored
    hTiny = PauliSum([pauli"X"], [1e-50])
    @test nnz(toMatrix(hTiny)) == 2 && nnz(toMatrix(Float32, hTiny)) == 0
    #>> A fixed-width integer overflow throws instead of wrapping around (also on 
    #>> negation); a wider `T` accumulates the same terms safely
    hBig = PauliSum([pauli"I", pauli"Z"], [typemax(Int), 1])
    @test_throws OverflowError toMatrix(hBig)
    @test toMatrix(Int128, hBig) == [Int128(typemax(Int)) + 1 0; 0 typemax(Int) - 1]
    hNeg = PauliSum([pauli"Z"], [typemin(Int)])
    @test_throws OverflowError toMatrix(hNeg)
    @test_throws OverflowError toMatrix(Float64, hNeg) #> `extendType(Float64, Int) == Int`
    @test toMatrix(Int128, hNeg) == [typemin(Int) 0; 0 -Int128(typemin(Int))]

    #> When `canonicalization=false`, `toMatrix` requires input `PauliSum` to be already in 
    #> the canonical form
    hd = PauliSum([pauli"X", pauli"Y"], [1.0, 2.0])
    hd.str[2] = pauli"X" #> Artificial non-canonical from
    @test toMatrix(hd) == 3 .* toMatrix(pauli"X")
    setCoeff!(hd, 3.0, 1)
    @test !isCanonical(hd)
    @test_throws ArgumentError toMatrix(hd, canonicalization=false)
    @test_throws ArgumentError toMatrix!(zeros(ComplexF64,2,2), hd, canonicalization=false)
    #>> By default, non-canonical `PauliSum` is canonicalized upon input
    @test toMatrix(hd) == 5 .* toMatrix(pauli"X")
    @test toMatrix!(zeros(ComplexF64, 2, 2), hd) == 5 .* toMatrix(pauli"X")
    @test toMatrix(PauliSum(hd)) == 5 .* toMatrix(pauli"X")
    #>> A phase left inside a stored string (through the aliasing form of `indexTerm`)
    hp = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    indexTerm(hp, 1, false).first.phase = negRea #> The first stored term is `Z`
    @test_throws ArgumentError toMatrix(hp, canonicalization=false)
    @test toMatrix(hp) == toMatrix(PauliSum(hp)) == [-2 1; 1 2] #> X - 2Z
    #>> Strings of different site counts (the buffer of `h` can hold them while `h` stays 
    #>> canonical) are implicitly padded
    hm = PauliSum([pauli"XI", pauli"IZ"], [1, 2])
    hm.str[begin] = pauli"X"
    @test !isCanonical(hm) && countSites(hm) == 2
    @test toMatrix(hm) == toMatrix(PauliSum([pauli"XI", pauli"IZ"], [1, 2]))

    #> The empty sum is the 1×1 zero matrix; the site count is bounded by the `Int` range
    @test toMatrix(PauliSum(Float64)) == zeros(1, 1) && nnz(toMatrix(PauliSum(Int))) == 0
    @test_throws DomainError toMatrix(PauliStr(8sizeof(Int) - 1))
    @test_throws DomainError toMatrix!(zeros(1, 1), PauliStr(8sizeof(Int) - 1))
end

@testset "toMatrix!" begin
    h = PauliSum([pauli"XI", pauli"IZ", pauli"YY"], [1.0, 2.0, 0.5im])
    ref = Matrix(toMatrix(h))

    #> Overwrite semantics and the returned object
    dst = fill(7.0 + 0.0im, 4, 4)
    @test toMatrix!(dst, h) === dst && dst == ref

    #> Element-type conversion upon assignment
    @test toMatrix!(zeros(ComplexF32, 4, 4), h) == ref
    @test toMatrix!(zeros(Rational{Int}, 4, 4), pauli"ZZ") == Matrix(toMatrix(pauli"ZZ"))
    @test toMatrix!(zeros(Int, 2, 2), PhaseFactor(1) * pauli"Y") == [0 1; -1 0] #> Real
    @test_throws InexactError toMatrix!(zeros(2, 2), pauli"Y")
    @test_throws InexactError toMatrix!(zeros(Int, 2, 2), PauliSum([pauli"X"], [0.5]))

    #> Size check (before any write), including the `PauliSym` and zero-site cases
    @test_throws ArgumentError toMatrix!(zeros(ComplexF64, 4, 5), h)
    @test_throws ArgumentError toMatrix!(zeros(ComplexF64, 2, 2), h)
    for sym in (symI, symX, symY, symZ)
        @test toMatrix!(zeros(Complex{Int}, 2, 2), sym) == Matrix(toMatrix(sym))
        @test_throws ArgumentError toMatrix!(zeros(Complex{Int}, 1, 1), sym)
    end
    @test toMatrix!(fill(1, 1, 1), pauli"-") == fill(-1, 1, 1)
    @test toMatrix!(fill(1.0, 1, 1), PauliSum(Float64)) == zeros(1, 1)

    #> A view addresses a block of a larger matrix; the rest of the matrix is untouched
    big = zeros(ComplexF64, 6, 6)
    @test toMatrix!(view(big, 2:5, 3:6), h) == ref
    @test big[2:5, 3:6] == ref && count(!iszero, big) == count(!iszero, ref)

    #> Offset axes: the first element along each axis hosts the first row (column)
    struct ZeroBasedMatrix{T} <: AbstractMatrix{T}
        data::Matrix{T}
    end

    Base.size(m::ZeroBasedMatrix) = size(m.data)
    Base.axes(m::ZeroBasedMatrix) = map(n->Base.IdentityUnitRange(0:(n - 1)), size(m.data))
    Base.getindex(m::ZeroBasedMatrix, i::Int, j::Int) = m.data[begin+i, begin+j]
    Base.setindex!(m::ZeroBasedMatrix, v, i::Int, j::Int) = (m.data[begin+i, begin+j] = v)

    zb = ZeroBasedMatrix(fill(3.0 + 0.0im, 4, 4))
    @test firstindex(zb, 1) == 0 && firstindex(zb, 2) == 0 #> Sanity check of the fixture
    @test toMatrix!(zb, h) === zb && zb.data == ref
end

end
