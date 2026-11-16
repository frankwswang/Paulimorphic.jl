using Test
using Random: Xoshiro
using LinearAlgebra: kron
using SparseArrays: SparseMatrixCSC, nnz, rowvals, nzrange
using Paulimorphic
using Paulimorphic: isIndexLabel, dumpTo!, posRea, posImg, negRea, negImg, setCoeff!, 
                    formatPauliOpMatInfo, genPauliOpMatEntryEvaluator, 
                    traverseMatrixEntries

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

@testset "formatPauliOpMatInfo" begin
    #> One `x`-mask, one `z`-mask, and one coefficient (with the `im` of every `Y` folded 
    #> in) per term in the stored order, preceded by the number of distinct `x`-masks, 
    #> which bounds the number of non-zero entries in each column of the matrix
    h = PauliSum([pauli"XI", pauli"IZ", pauli"YI", pauli"ZZ"], [1, 2, 3, 4])
    #>> By default, site `k` is bit `nSite-k` of the masks, which thus depend on `nSite`
    info = 2 => (UInt[2, 2, 0, 0], UInt[0, 2, 1, 3], [1, 3im, 2, 4])
    @test formatPauliOpMatInfo(Int, h) == formatPauliOpMatInfo(Int, h, 2) == info
    info3 = 2 => (UInt[4, 4, 0, 0], UInt[0, 4, 2, 6], [1, 3im, 2, 4])
    @test formatPauliOpMatInfo(Int, h, 3) == info3 #> Identity on the additional site
    #>> With `reverseKron=true`, site `k` is bit `k-1` of the masks, as in a `PauliStr`
    infoR = 2 => (UInt[1, 1, 0, 0], UInt[0, 1, 2, 3], [1, 3im, 2, 4])
    @test formatPauliOpMatInfo(Int, h, reverseKron=true) == infoR
    @test formatPauliOpMatInfo(Int, h, 3, reverseKron=true) == infoR
    #> The coefficients are converted to the given core type before the phases are folded
    @test formatPauliOpMatInfo(Float64, h).second[end] isa Memory{ComplexF64}
    #> A site count below the one of the input clears the bits above it. The public 
    #> functions reject such a site count before this internal function is reached
    infoLow = 2 => (UInt[1, 1, 0, 0], UInt[0, 1, 0, 1], [1, 3im, 2, 4])
    @test formatPauliOpMatInfo(Int, h, 1) == infoLow
end

@testset "toMatrix" begin
    syms = [symI, symX, symY, symZ]
    phases = [posRea, posImg, negRea, negImg]
    rng = Xoshiro(1234)
    randStr(nSite::Int) = PauliStr([rand(rng, syms) for _ in 1:nSite], rand(rng, phases))

    #> Independent dense reference of the documented basis convention: by default, site 
    #> `k` is the `k`-th factor from the left of the Kronecker product (most significant 
    #> bit first); with `reverseKron=true`, it is the `k`-th factor from the right
    function kronRef(str::PauliStr, nSite::Int=countSites(str); reverseKron::Bool=false)
        sites = reverseKron ? (nSite:-1:1) : (1:1:nSite)
        mats = [Matrix(toMatrix(indexSite(str, k, true))) for k in sites]
        evalPhase(str) .* foldl(kron, mats; init=ones(Complex{Int}, 1, 1))
    end
    function sumRef(h::PauliSum{T}, nSite::Int=countSites(h); reverseKron::Bool=false
                    ) where {T}
        res = zeros(Complex{T}, 2^nSite, 2^nSite)
        for (str, coeff) in listTerms(h)
            res .+= coeff .* kronRef(str, nSite; reverseKron)
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
        bl *= Matrix(toMatrix(str, reverseKron=true)) == kronRef(str, reverseKron=true)
        bl *= toMatrix(Float64, str) == mat
        bl *= toMatrix(Float64, str) isa SparseMatrixCSC{ComplexF64, Int}
        @test bl
    end
    @test Matrix(toMatrix(pauli"-")) == fill(-1, 1, 1) #> Zero-site string: its phase
    @test Matrix(toMatrix(pauli"XI")) == [0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]
    @test Matrix(toMatrix(pauli"IX")) == [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]
    #>> `reverseKron=true` reverses the order of the factors of the Kronecker product
    matZZY = kron(toMatrix(symZ), toMatrix(symZ), toMatrix(symY))
    @test toMatrix(pauli"ZZY") == matZZY == toMatrix(pauli"YZZ", reverseKron=true)
    @test toMatrix(pauli"ZZY", reverseKron=true) == toMatrix(pauli"YZZ") != matZZY

    #> An `nSite` above the site count pads the operator with the identity on the upper 
    #> sites, i.e., on the less significant bits of the basis-state index by default, and 
    #> on the more significant ones with `reverseKron=true`
    for str in (pauli"-", pauli"X", pauli"iYZ", pauli"-iZXY"), nPad in 0:2
        nSite = countSites(str) + nPad
        mat = toMatrix(str, nSite)
        @test size(mat) == (2^nSite, 2^nSite) && Matrix(mat) == kronRef(str, nSite)
        matR = toMatrix(str, nSite, reverseKron=true)
        @test Matrix(matR) == kronRef(str, nSite, reverseKron=true)
    end
    @test toMatrix(pauli"X", 2) == toMatrix(pauli"XI")
    @test Matrix(toMatrix(pauli"-", 2)) == [-1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 -1]
    for sym in syms
        @test toMatrix(sym, 1) == toMatrix(sym)
        @test Matrix(toMatrix(sym, 3)) == kronRef(PauliStr(1, sym), 3)
        @test Matrix(toMatrix(sym, 3, reverseKron=true)) == 
              kronRef(PauliStr(1, sym), 3, reverseKron=true)
        @test toMatrix(Float64, sym, 2) isa SparseMatrixCSC{ComplexF64, Int}
    end
    @test Matrix(toMatrix(symZ, 2)) == [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    @test Matrix(toMatrix(symZ, 2, reverseKron=true)) == 
          [1 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
    hPad = PauliSum([pauli"XI", pauli"IZ", pauli"YY"], [1.0, 2.0, 0.5im])
    @test Matrix(toMatrix(hPad, 3)) == sumRef(hPad, 3) == kron(sumRef(hPad), [1 0; 0 1])
    @test Matrix(toMatrix(hPad, 3, reverseKron=true)) == 
          kron([1 0; 0 1], sumRef(hPad, reverseKron=true))
    @test toMatrix(hPad, 3) == toMatrix(reframe(hPad, 3))
    @test toMatrix(Float32, hPad, 3) isa SparseMatrixCSC{ComplexF32, Int}

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
        #>> The same checks in the reversed order of the Kronecker product
        refR = sumRef(h, reverseKron=true)
        matR = toMatrix(h, reverseKron=true)
        bl1 *= Matrix(matR) == refR
        bl3 *= all(j->issorted(rowvals(matR)[nzrange(matR, j)]), 1:2^n)
        bl4 *= toMatrix!(zeros(Complex{Int}, 2^n, 2^n), h, reverseKron=true) == refR
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
    @test_throws ArgumentError toMatrix(Signed, symX)
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
    #>> The phase of a string is folded into its coefficient at the same precision level: 
    #>> the entries `±128` of `-128 * YY` only require a `T` wider than `Int8`
    hYY = PauliSum([pauli"YY"], Int8[-128])
    refYY = [0 0 0 128; 0 0 -128 0; 0 -128 0 0; 128 0 0 0]
    @test_throws OverflowError toMatrix(hYY)
    @test toMatrix(Int16, hYY) == refYY
    @test toMatrix!(zeros(Int16, 4, 4), hYY) == refYY
    hZZ = PauliSum([pauli"ZZ"], Int8[-128]) #> The signs of `ZZ` involve no phase
    @test toMatrix(Int16, hZZ) == [-128 0 0 0; 0 128 0 0; 0 0 128 0; 0 0 0 -128]

    #> By default (`requireCanonical=true`), the input `PauliSum` is required to be in the 
    #> canonical form; with `requireCanonical=false`, it is canonicalized (on a copy)
    hd = PauliSum([pauli"X", pauli"Y"], [1.0, 2.0])
    hd.str[2] = pauli"X" #> Artificial non-canonical form
    @test toMatrix(hd, requireCanonical=false) == 3 .* toMatrix(pauli"X")
    setCoeff!(hd, 3.0, 1)
    @test !isCanonical(hd)
    @test_throws ArgumentError toMatrix(hd)
    @test_throws ArgumentError toMatrix!(zeros(ComplexF64,2,2), hd)
    #>> The rejection precedes any write to the destination
    dst = fill(7.0 + 0.0im, 2, 2)
    @test_throws ArgumentError toMatrix!(dst, hd)
    @test dst == fill(7, 2, 2)
    #>> The default value of `requireCanonical` should be the same for every method
    @test_throws ArgumentError toMatrix(Float64, hd)
    @test_throws ArgumentError toMatrix(Float64, hd, requireCanonical=true)
    #>> With `requireCanonical=false`, a non-canonical `PauliSum` is canonicalized on input
    @test toMatrix(hd, requireCanonical=false) == 5 .* toMatrix(pauli"X")
    @test toMatrix!(zeros(ComplexF64, 2, 2), hd, requireCanonical=false) == 
          5 .* toMatrix(pauli"X")
    @test toMatrix(PauliSum(hd)) == 5 .* toMatrix(pauli"X")
    @test toMatrix(Float64, hd, requireCanonical=false) == 5 .* toMatrix(pauli"X")
    #>> A phase left inside a stored string (through the aliasing form of `indexTerm`)
    hp = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    indexTerm(hp, 1, false).first.phase = negRea #> The first stored term is `Z`
    @test_throws ArgumentError toMatrix(hp)
    @test toMatrix(hp, requireCanonical=false) == toMatrix(PauliSum(hp)) == 
          [-2 1; 1 2] #> X - 2Z
    #>> Strings of different site counts (the buffer of `h` can hold them, which breaks the 
    #>> canonical form) are implicitly padded
    hm = PauliSum([pauli"XI", pauli"IZ"], [1, 2])
    hm.str[begin] = pauli"X"
    @test !isCanonical(hm) && countSites(hm) == 2
    @test toMatrix(hm, requireCanonical=false) == 
          toMatrix(PauliSum([pauli"XI", pauli"IZ"], [1, 2]))
    #>> Terms that are merely out of order are sorted on a shallow copy of the sum, which 
    #>> leaves the input (including the order of its stored terms) untouched
    hs = PauliSum([pauli"XI", pauli"IZ"], [1.0, 2.0])
    refS = [2 0 1 0; 0 -2 0 1; 1 0 2 0; 0 1 0 -2]
    @test toMatrix(hs) == refS
    hs.str[1], hs.str[2] = hs.str[2], hs.str[1]
    hs.coeff[1], hs.coeff[2] = hs.coeff[2], hs.coeff[1]
    strsBefore, coeffsBefore = copy(hs.str), copy(hs.coeff)
    @test !isCanonical(hs)
    @test_throws ArgumentError toMatrix(hs)
    @test toMatrix(hs, requireCanonical=false) == refS
    @test toMatrix!(zeros(ComplexF64, 4, 4), hs, requireCanonical=false) == refS
    @test all(hs.str .=== strsBefore) && hs.coeff == coeffsBefore && !isCanonical(hs)

    #> The empty sum is the 1×1 zero matrix; the site count is bounded by the `Int` range
    @test toMatrix(PauliSum(Float64)) == zeros(1, 1) && nnz(toMatrix(PauliSum(Int))) == 0
    @test_throws DomainError toMatrix(PauliStr(8sizeof(Int) - 1))
    @test_throws DomainError toMatrix!(zeros(1, 1), PauliStr(8sizeof(Int) - 1))
    #>> An explicit `nSite` (of any integer type) is subject to the same upper bound, and 
    #>> it sets the order of the zero matrix representing the empty sum
    matEmpty = toMatrix(PauliSum(Float64), 2)
    @test size(matEmpty) == (4, 4) && nnz(matEmpty) == 0
    @test_throws DomainError toMatrix(pauli"X", 8sizeof(Int) - 1)
    @test_throws DomainError toMatrix(pauli"X", -1)
    @test_throws DomainError toMatrix(pauli"X", BigInt(2)^80)
    @test_throws DomainError toMatrix!(zeros(2, 2), pauli"X", -1)
    #>> An `nSite` below the site count is rejected instead of cropping the operator
    @test_throws DomainError toMatrix(pauli"XZ", 1)
    @test_throws DomainError toMatrix(symX, 0)
    @test_throws DomainError toMatrix(Float64, PauliSum([pauli"XZ"], [1.0]), 1)
    #>> An `nSite` of any integer type is accepted as long as an `Int` can represent it
    @test toMatrix(pauli"X", Int8(2)) == toMatrix(pauli"X", 2)
    @test toMatrix(Float64, hInt, UInt8(3)) == toMatrix(Float64, hInt, 3)
    @test toMatrix!(zeros(Int, 4, 4), symZ, big(2)) == toMatrix!(zeros(Int, 4, 4), symZ, 2)
    @test traverseMatrixEntries((_, _) -> nothing, Int, hInt, Int8(2)) == 
          traverseMatrixEntries((_, _) -> nothing, Int, hInt, 2)

    #> The transient buffers of `toMatrix` are bounded by the number of distinct `x`-masks 
    #> among the terms (`1` for a diagonal operator) instead of the number of terms (`66` 
    #> below).
    nDiag = 12
    genZZStr(i::Int, j::Int) = PauliStr([k == i || k == j ? symZ : symI for k in 1:nDiag])
    hDiag = PauliSum([genZZStr(i, j) for i in 1:nDiag for j in (i + 1):nDiag], 
                     ones(nDiag * (nDiag - 1) ÷ 2))
    nXMask, (xMasksDiag, zMasksDiag, coeffsDiag) = formatPauliOpMatInfo(Float64, hDiag)
    @test nXMask == 1 && length(coeffsDiag) == countTerms(hDiag) == 66
    matDiag = toMatrix(hDiag) #> This call also compiles the one measured below
    @test nnz(matDiag) == 2^nDiag
    @test (@allocated toMatrix(hDiag)) < 2^20 #> 1 MiB
    #>> The evaluator of the entries is concretely typed (otherwise, every column pays a 
    #>> run-time dispatch), and evaluating a column with it allocates nothing
    evalArgTypes = (Type{Float64}, Memory{UInt}, Memory{UInt}, Memory{ComplexF64})
    @test isconcretetype(only(Base.return_types(genPauliOpMatEntryEvaluator, evalArgTypes)))
    evalDiag = genPauliOpMatEntryEvaluator(Float64, xMasksDiag, zMasksDiag, coeffsDiag)
    skipEntry(::Pair, ::Int) = nothing
    colLabel = UInt(5)
    @test evalDiag(skipEntry, colLabel, 0) == 1 #> Also sets up the accumulator's storage
    @test evalDiag(skipEntry, colLabel, 3) == 4
    @test (@allocated evalDiag(skipEntry, colLabel, 0)) == 0
end

@testset "toMatrix!" begin
    h = PauliSum([pauli"XI", pauli"IZ", pauli"YY"], [1.0, 2.0, 0.5im])
    ref = Matrix(toMatrix(h))

    #> Overwrite semantics and the returned object
    dst = fill(7.0 + 0.0im, 4, 4)
    @test toMatrix!(dst, h) === dst && dst == ref
    #>> Every element of `dst` is assigned, also by the methods for a single-site operator
    for sym in (symI, symX, symY, symZ)
        @test toMatrix!(fill(7 + 7im, 2, 2), sym) == Matrix(toMatrix(sym))
        @test toMatrix!(fill(7 + 7im, 4, 4), sym, 2) == Matrix(toMatrix(sym, 2))
    end
    @test toMatrix!(fill(7, 2, 2), symI) == [1 0; 0 1]

    #> Element-type conversion upon assignment
    @test toMatrix!(zeros(ComplexF32, 4, 4), h) == ref
    @test toMatrix!(zeros(Rational{Int}, 4, 4), pauli"ZZ") == Matrix(toMatrix(pauli"ZZ"))
    @test toMatrix!(zeros(Int, 2, 2), PhaseFactor(1) * pauli"Y") == [0 1; -1 0] #> Real
    @test_throws InexactError toMatrix!(zeros(2, 2), pauli"Y")
    @test_throws InexactError toMatrix!(zeros(Int, 2, 2), PauliSum([pauli"X"], [0.5]))
    #>> A non-real entry for a real `dst` raises the same error through every method
    @test_throws InexactError toMatrix!(zeros(2, 2), symY)
    @test_throws InexactError toMatrix!(zeros(2, 2), symY, 1)

    #> Size check (before any write), including the `PauliSym` and zero-site cases
    @test_throws ArgumentError toMatrix!(zeros(ComplexF64, 4, 5), h)
    @test_throws ArgumentError toMatrix!(zeros(ComplexF64, 2, 2), h)
    for sym in (symI, symX, symY, symZ)
        @test toMatrix!(zeros(Complex{Int}, 2, 2), sym) == Matrix(toMatrix(sym))
        @test_throws ArgumentError toMatrix!(zeros(Complex{Int}, 1, 1), sym)
    end
    @test toMatrix!(fill(1, 1, 1), pauli"-") == fill(-1, 1, 1)
    @test toMatrix!(fill(1.0, 1, 1), PauliSum(Float64)) == zeros(1, 1)

    #> An explicit `nSite` above the site count pads the operator with the identity on the 
    #> upper sites; the order of `dst` must then be `2^nSite`
    @test toMatrix!(fill(7.0 + 0.0im, 8, 8), h, 3) == kron(ref, [1 0; 0 1])
    @test toMatrix!(fill(7.0 + 0.0im, 4, 4), pauli"X", 2) == Matrix(toMatrix(pauli"XI"))
    @test toMatrix!(zeros(Int, 4, 4), symZ, 2) == [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    #>> `reverseKron=true` reverses the order of the Kronecker product for every method 
    #>> that takes `nSite`, with the identity padded on the more significant bits
    refR = Matrix(toMatrix(h, reverseKron=true))
    @test refR != ref && toMatrix!(fill(7.0 + 0.0im, 4, 4), h, reverseKron=true) == refR
    @test toMatrix!(fill(7.0 + 0.0im, 8, 8), h, 3, reverseKron=true) == 
          kron([1 0; 0 1], refR)
    @test toMatrix!(fill(7.0 + 0.0im, 4, 4), pauli"XI", reverseKron=true) == 
          Matrix(toMatrix(pauli"IX"))
    @test toMatrix!(zeros(Int, 4, 4), symZ, 2, reverseKron=true) == 
          [1 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
    @test_throws ArgumentError toMatrix!(zeros(ComplexF64, 4, 4), h, 3)
    @test_throws ArgumentError toMatrix!(zeros(4, 4), pauli"X")
    #>> An `nSite` below the site count is rejected before any write to `dst`
    dstLow = fill(7.0, 2, 2)
    @test_throws DomainError toMatrix!(dstLow, pauli"XZ", 1)
    @test dstLow == fill(7.0, 2, 2)

    #> The core data type of `dst` is checked by every method
    dstAbstract = Matrix{Signed}(undef, 2, 2)
    @test_throws ArgumentError toMatrix!(dstAbstract, PauliSum([pauli"Z"], [1]))
    @test_throws ArgumentError toMatrix!(dstAbstract, pauli"Z")
    @test_throws ArgumentError toMatrix!(dstAbstract, symZ)

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
