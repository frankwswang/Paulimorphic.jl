using Test
using Random: Xoshiro
using LinearAlgebra: kron
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg, mul, add, combine, sub, toAdjoint

@testset "Operations.jl" begin

syms = [symI, symX, symY, symZ]
phases = [posRea, posImg, negRea, negImg]
#> Ground-truth reference: dense matrix representation
matI = toMatrix(symI)
matX = toMatrix(symX)
matY = toMatrix(symY)
matZ = toMatrix(symZ)

function randStr(rng, nSite::Int)
    PauliStr([rand(rng, syms) for _ in 1:nSite], rand(rng, phases))
end

function randSum(rng, nSite::Int, nTerm::Int)
    strs = [randStr(rng, nSite) for _ in 1:nTerm]
    coeffs = [Complex{Int}(rand(rng, -3:3), rand(rng, -3:3)) for _ in 1:nTerm]
    PauliSum(strs, coeffs) #> Zero coefficients and duplicate strings exercise merging
end

function strToMat(pStr::PauliStr, nSite::Int=pStr.n)
        nSitePerWord = 8 * sizeof(UInt)
        res = ones(ComplexF64, 1, 1)
        for i in 1:nSite
            w, b = fldmod(i-1, nSitePerWord)
            zWord = w < length(pStr.z) ? pStr.z[begin+w] : zero(UInt)
            xWord = w < length(pStr.x) ? pStr.x[begin+w] : zero(UInt)
            z = !iszero(zWord & (one(UInt) << b))
            x = !iszero(xWord & (one(UInt) << b))
            m = (z && x) ? matY : (z ? matZ : (x ? matX : matI))
            res = kron(m, res)
        end
        evalPhase(pStr.phase) .* res
    end

function sumToMat(h::PauliSum, nSite::Int=countSites(h))
    res = zeros(ComplexF64, 2^nSite, 2^nSite)
    for i in eachindex(h.str)
        res .+= h.coeff[begin+i-1] .* strToMat(h.str[begin+i-1], nSite)
    end
    res
end

@testset "add" begin
    #> `add`: named-function layer
    #>> 1. Str + Str merges under simplification; default coefficient type is Int
    @test add(pauli"X", pauli"X") == PauliSum([pauli"X"], [2])
    @test add(pauli"X", pauli"X") isa PauliSum{Int}

    #>> 2. Explicit coefficient type (value and type)
    @test add(Float64, pauli"X", pauli"Z") == PauliSum([pauli"X", pauli"Z"], [1.0, 1.0])
    @test add(Float64, pauli"X", pauli"Z") isa PauliSum{Float64}

    #>> 3. Identity-padding rebuild: `X` and `XI` become equal after site-count promotion
    @test add(pauli"X", pauli"XI") == PauliSum([pauli"XI"], [2])

    #>> 4. Sum + Sum with cross-type promotion (Int ∪ Float64 -> Float64)
    h1 = PauliSum([pauli"XZ"], [2])
    h2 = PauliSum([pauli"XZ", pauli"YI"], [1.0, 0.5])
    @test add(h1, h2) == PauliSum([pauli"XZ", pauli"YI"], [3.0, 0.5])
    @test add(h1, h2) isa PauliSum{Float64}

    #>> 5. Sum + Sum with the empty sum: additive identity (and empty-buffer `vcat` edge)
    @test add(PauliSum(Int), h2) == h2

    #>> 6. Phase absorption through the Pair method: (-im * X) => 2.0 contributes -2.0im
    @test add(PauliSum(Float64), mul(pauli"X", negImg)=>2.0) == 
          PauliSum([pauli"X"], [-2.0im])

    #>> 7. Pair method with cross-type promotion, incl. site-count rebuild of the added `X`
    @test add(h1, pauli"X"=>0.5) == PauliSum([pauli"XZ", pauli"XI"], [2.0, 0.5])
    @test add(h1, pauli"X"=>0.5) isa PauliSum{Float64}

    #>> 8. Sum + Str: nominal coefficient `one(T)` merged into the matching term
    @test add(h2, pauli"YI") == PauliSum([pauli"XZ", pauli"YI"], [1.0, 1.5])

    #>> 9. Exact cancellation yields the empty (zero) operator
    resCancel = add(pauli"X", mul(pauli"X", negRea))
    @test resCancel == PauliSum(Int)
    @test isempty(resCancel.str)

    #>> 10. simplification=false merges equal strings all the same (canonical form), but 
    #>> retains the terms whose merged coefficient is exactly zero, per method
    @test add(pauli"X", pauli"X", false) == PauliSum([pauli"X"], [2])
    resZero = add(pauli"X", mul(pauli"X", negRea), false)
    @test resZero.str == [pauli"X"] && resZero.coeff == Complex{Int}[0]
    @test isCanonical(resZero) && !isCanonical(resZero, true)
    @test PauliSum(resZero, true) == PauliSum(Int)
    @test countTerms(add(h1, h1, false)) == countTerms(add(h1, mul(h1, negRea), false)) == 1
    @test countTerms(add(h1, pauli"XZ"=>-2, false)) == 1
    hOne = PauliSum([pauli"XZ"], [1])
    @test countTerms(add(hOne, mul(pauli"XZ", negRea), false)) == 1

    #>> 11. Round trip with indexTerm
    h3 = PauliSum([pauli"XZ"], [0.5])
    @test add(h3, indexTerm(h3, 1)) == PauliSum([pauli"XZ"], [1.0])

    #>> 12. Ownership: result must not reference either input's data (buffer level)
    resOwn = add(h1, h2)
    @test all(s -> all(t -> s.x !== t.x && s.z !== t.z, vcat(h1.str, h2.str)), resOwn.str)

    #> `Base.:+`: operator layer
    #>> Binary `+` on strings is equivalent to the default-`T` `add` method
    @test pauli"X" + pauli"Z" == add(pauli"X", pauli"Z")
    @test pauli"X" + pauli"Z" == PauliSum([pauli"X", pauli"Z"], [1, 1])
    @test pauli"X" + pauli"X" == PauliSum([pauli"X"], [2])

    #>> Chained all-string addition (single-pass n-ary method, arity 4)
    @test pauli"X" + pauli"Y" + pauli"Z" + pauli"X" == 
          PauliSum([pauli"X", pauli"Y", pauli"Z"], [2, 1, 1])

    #>> Sum ↔ Str in both orders; Sum + Sum routes through `add`
    h = PauliSum([pauli"Z"], [1.0])
    @test pauli"X" + h == PauliSum([pauli"X", pauli"Z"], [1.0, 1.0])
    @test h + pauli"X" == pauli"X" + h
    @test h1 + h2 == add(h1, h2)

    #>> Mixed chain uses the n-ary method with correct promotion
    @test (PauliSum(Float64) + pauli"X" + pauli"Y") isa PauliSum{Float64}

    #>> Intentionally unsupported combinations stay unsupported
    @test_throws MethodError +(pauli"X")            #>> No unary `+`
    @test_throws MethodError h + (pauli"Y"=>2.0)        #>> `Pair` terms are not operators
    @test_throws MethodError (pauli"Y"=>2.0) + h
    @test_throws MethodError pauli"X" + (pauli"Y"=>2.0)
    @test_throws MethodError add(h, pauli"Y"=>im)      #>> `im` is a `Complex{Bool}`

    #>> A non-concrete `T` is rejected by the core-type check in every method
    @test_throws ArgumentError add(Signed, h1, h2)
    @test_throws ArgumentError add(Rational{<:Signed}, h1, h2)
    @test_throws ArgumentError add(Rational{<:Signed}, h, pauli"X"=>1)
    @test_throws ArgumentError add(Rational{<:Signed}, h, pauli"X")
    @test_throws ArgumentError add(Rational{<:Signed}, pauli"X", pauli"Z")

    #> Type-conversion methods
    h1 = PauliSum([pauli"X"], [3])
    h2 = PauliSum([pauli"Z"], [0.5])
    @test add(Float32, h1, h2) isa PauliSum{Float32}
    @test add(Float32, h2, pauli"X") == add(PauliSum(Float32, h2), pauli"X")
    #>> A `PauliStr` takes no part in the precision level at which the terms are merged, 
    #>> which makes `add(T, h, s)` agree with `combine(T, [h, s])`
    hThird = PauliSum([pauli"I"], [-1//3])
    @test add(Float64, hThird, pauli"I") == PauliSum([pauli"I"], [Float64(2//3)])
    @test add(Float64, hThird, pauli"I") == 
          combine(Float64, Union{PauliStr, PauliSum}[hThird, pauli"I"])
    #>> `extendType(Float64, Int8) == Int8` for both of them, as for `add(Float64, h, h)`
    hEdge = PauliSum([pauli"X"], Int8[127])
    @test_throws OverflowError add(Float64, hEdge, pauli"X")
    @test_throws OverflowError add(Float64, hEdge, hEdge)
    @test_throws OverflowError combine(Float64, Union{PauliStr, PauliSum}[hEdge, pauli"X"])
end

@testset "combine" begin
    #> n-ary methods: all the terms are merged in one compensated summation
    s = pauli"X"
    ha = PauliSum([s], [1.0])
    hb = PauliSum([s], [1e-17])
    hc = PauliSum([s], [-1.0])
    #>> A chain of three or more `+` is one n-ary call (its merge semantics and order 
    #>> independence are pinned in `quality-tests/Determinism-test.jl`)
    @test (ha + hb + hc) isa PauliSum{Float64}
    #>> Promotion of mixed core types; exact cancellation drops the term
    hi = PauliSum([pauli"Z"], [2])
    @test combine([hi, ha, hc]) == PauliSum([pauli"Z"], [2.0])
    @test combine([hi, ha, hc]) isa PauliSum{Float64}
    @test hi + ha + hc == combine([hi, ha, hc])
    @test combine(Float32, [hi, ha, hc]) isa PauliSum{Float32}
    #>> Edge cases: empty collections and a single operand (fresh buffers)
    @test combine(Float64, PauliSum{Float64}[]) == 
          combine(PauliSum{Float64}[]) == PauliSum(Float64)
    @test_throws ArgumentError combine(PauliSum[])
    @test combine([ha]) == ha && combine([ha]).str[begin] !== ha.str[begin]
    #>> `simplification=false` merges all the same, retaining an exactly cancelled term
    @test combine([ha, hb, hc], false) == combine([ha, hb, hc])
    @test combine([ha, hc], false).coeff == ComplexF64[0.0] #> `1.0 - 1.0`, retained
    @test countTerms(combine([ha, hc])) == 0
    #>> Mixed chains including a `PauliStr` use the n-ary method as well
    @test ha + hc + pauli"Z" == PauliSum([pauli"Z"], [1.0])
    #>> A single `PauliStr` stored in a mixed-type container
    @test combine(Union{PauliStr, PauliSum}[pauli"X"]) == PauliSum(Int, [pauli"X"])
    @test combine(Union{PauliStr, PauliSum}[pauli"X", ha]) == PauliSum([pauli"X"], [2.0])
    #>> A callable first argument is called as `f(str => coeff)` for every element type of 
    #>> `terms`: a container of `PauliStr` only gives the same result as a mixed-type one
    sX, sZ = pauli"X", pauli"Z"
    mixed = Union{PauliStr, PauliSum}[sX, sZ]
    twice = (pair::Pair) -> 2 * pair.second
    @test combine(twice, Float64, [sX, sZ]) == combine(twice, Float64, mixed) == 
          PauliSum([sX, sZ], [2.0, 2.0])
    #>> The values returned by the callable are converted to `Complex{T}`
    @test combine(_->0x03, Int, [sX, sZ]) == combine(_->0x03, Int, mixed) == 
          PauliSum([sX, sZ], [3, 3])
    pick = pair -> (pair.first == sX ? 1 : 2.0)
    @test combine(pick, Float64, [sX, sZ]) == combine(pick, Float64, mixed) == 
          PauliSum([sX, sZ], [1.0, 2.0])
    #>> A `PauliStr` takes no part in the promotion of core data types, so the core data 
    #>> type of a chained sum does not depend on where its strings are placed; `Int` is 
    #>> only resorted to when the sum holds no `PauliSum`
    hInt8 = PauliSum([sZ], Int8[1])
    @test (sX + sX) isa PauliSum{Int} && (sX + sX + sX) isa PauliSum{Int}
    @test (hInt8 + sX) isa PauliSum{Int8} && (sX + hInt8) isa PauliSum{Int8}
    @test (hInt8 + sX + sX) isa PauliSum{Int8} && (sX + hInt8 + sX) isa PauliSum{Int8}
    @test (sX + sX + hInt8) isa PauliSum{Int8}
    @test sX + sX + hInt8 == hInt8 + sX + sX == PauliSum([sZ, sX], Int8[1, 2])
    #>> An explicitly grouped `sX + sX` is a `PauliSum{Int}` on its own, which then takes 
    #>> part in the promotion
    @test ((sX + sX) + hInt8) isa PauliSum{Int}
    #>> `combine` follows the same rule, in which `eltype(terms)` decides whenever it only 
    #>> admits `PauliSum` of one core data type
    @test combine(Union{PauliStr, PauliSum}[sX, hInt8]) isa PauliSum{Int8}
    @test combine(Union{PauliStr, PauliSum}[sX, sX]) isa PauliSum{Int}
    @test combine(Union{PauliStr, PauliSum{Int8}}[sX, hInt8]) isa PauliSum{Int8}
    @test combine(Union{PauliStr, PauliSum{Int8}}[sX, sX]) isa PauliSum{Int8}
    @test combine(Union{PauliStr, PauliSum{Int8}}[]) isa PauliSum{Int8}
    @test combine(PauliSum[hInt8, PauliSum([sX], Int16[1])]) isa PauliSum{Int16}
    #>> With `T` specified, the coefficients of equal strings are merged at the precision 
    #>> level of `extendType(T, complex(C))` and converted to `Complex{T}` last, in the 
    #>> same way as by `add(T, h1, h2)`; `C` is the promotion of the core data types of the 
    #>> `PauliSum` in `terms`
    hHalf = PauliSum([sX], [0.5])
    @test combine(Int, [hHalf, hHalf]) == add(Int, hHalf, hHalf) == PauliSum([sX], [1])
    @test combine(Int, [hHalf, hHalf]) isa PauliSum{Int}
    @test_throws InexactError combine(Int, [hHalf, hHalf, hHalf]) #> `1.5`
    hWide = PauliSum([sX], [1.0 + 2.0^-40])
    @test combine(Float32, [hWide, hc]) == add(Float32, hWide, hc) == 
          PauliSum([sX], [Float32(2.0^-40)])
    h200, hNeg150 = PauliSum([sX], [200]), PauliSum([sX], [-150])
    @test combine(Int8, [h200, hNeg150]) == add(Int8, h200, hNeg150) == 
          PauliSum([sX], Int8[50])
    #>> A fixed-width integer overflow at that precision level throws 
    #>> (`extendType(Float64, Int8) == Int8`); a wider `T` merges the same terms safely
    hMax = PauliSum([sX], Int8[100])
    @test_throws OverflowError combine(Float64, [hMax, hMax])
    @test_throws OverflowError add(Float64, hMax, hMax)
    @test combine(Int16, [hMax, hMax]) == add(Int16, hMax, hMax) == 
          PauliSum([sX], Int16[200])
    #>> A `PauliStr` among `PauliSum` takes no part in the promotion: its unit coefficient 
    #>> is merged at the precision level set by those `PauliSum`
    @test combine(Int, Union{PauliStr, PauliSum}[sX, hHalf, hHalf]) == PauliSum([sX], [2])
    hThird = PauliSum([sX], [-1//3])
    @test combine(Float64, Union{PauliStr, PauliSum}[sX, hThird]) == 
          PauliSum([sX], [Float64(2//3)]) #> Unlike `1.0 - Float64(1//3)`
    #>> `PauliStr` alone are merged at the precision level of `Complex{T}`, in the same way 
    #>> as by `PauliSum(T, strs)`
    strs200 = fill(sX, 200)
    @test combine(Float64, Union{PauliStr, PauliSum}[sX, sX]) == PauliSum(Float64, [sX, sX])
    @test_throws OverflowError PauliSum(Int8, strs200)
    @test_throws OverflowError combine(Int8, strs200)
    @test_throws OverflowError combine(Int8, Vector{Union{PauliStr, PauliSum}}(strs200))
    #>> In contrast, the coefficients returned by a callable first argument are converted 
    #>> to `Complex{T}` before the merge
    @test_throws InexactError combine(pair->pair.second, Int, [hHalf, hHalf])
end

@testset "sub" begin
    rng = Xoshiro(20261019)

    #> `sub`: named-function layer
    #>> 1. Str - Str: self-difference cancels to the empty (zero) operator; default `Int`
    resCancel = sub(pauli"X", pauli"X")
    @test resCancel == PauliSum(Int)
    @test resCancel isa PauliSum{Int}
    @test isempty(resCancel.str)

    #>> 2. Distinct strings, with and without an explicit coefficient type (value and type)
    @test sub(pauli"X", pauli"Z") == PauliSum([pauli"X", pauli"Z"], [1, -1])
    @test sub(Float64, pauli"X", pauli"Z") == PauliSum([pauli"X", pauli"Z"], [1.0, -1.0])
    @test sub(Float64, pauli"X", pauli"Z") isa PauliSum{Float64}

    #>> 3. Subtrahend phase absorbed before merging: X - (-X) == 2X, X - (im X) == (1-im)X
    @test sub(pauli"X", mul(pauli"X", negRea)) == PauliSum([pauli"X"], [2])
    @test sub(pauli"X", mul(pauli"X", posImg)) == PauliSum([pauli"X"], [1 - im])

    #>> 4. Identity-padding rebuild: `X` and `XI` become equal after site-count promotion
    @test sub(pauli"X", pauli"XI") == PauliSum(Int)
    @test sub(pauli"XI", mul(pauli"X", negRea)) == PauliSum([pauli"XI"], [2])

    #>> 5. Sum - Sum in both orders with cross-type promotion (Int ∪ Float64 -> Float64)
    h1 = PauliSum([pauli"XZ"], [2])
    h2 = PauliSum([pauli"XZ", pauli"YI"], [1.0, 0.5])
    @test sub(h1, h2) == PauliSum([pauli"XZ", pauli"YI"], [1.0, -0.5])
    @test sub(h1, h2) isa PauliSum{Float64}
    @test sub(h2, h1) == PauliSum([pauli"XZ", pauli"YI"], [-1.0, 0.5])

    #>> 6. The zero operator as minuend/subtrahend (empty-buffer `vcat` edge), and h - h
    @test sub(h2, PauliSum(Int)) == h2
    @test sub(PauliSum(Int), h2) == mul(h2, negRea)
    @test sub(h2, h2) == PauliSum(Float64)

    #>> 7. Pair methods in only one order, incl. phase absorption of the subtracted term
    hZero = PauliSum(Float64)
    @test sub(hZero, mul(pauli"X", negImg)=>2.0) == PauliSum([pauli"X"], [ 2.0im])
    @test sub(h1, pauli"X"=>0.5) == PauliSum([pauli"XZ", pauli"XI"], [ 2.0, -0.5])
    @test sub(h1, pauli"X"=>0.5) isa PauliSum{Float64}

    #>> 8. Str methods in both orders: nominal coefficient `one(T)` merged into the match
    @test sub(h2, pauli"YI") == PauliSum([pauli"XZ", pauli"YI"], [1.0, -0.5])
    @test sub(pauli"YI", h2) == PauliSum([pauli"XZ", pauli"YI"], [-1.0, 0.5])

    #>> 9. Consistency with the phase-multiplication route for every method shape
    @test sub(h1, h2) == add(h1, mul(h2, negRea))
    @test sub(h2, h1) == add(h2, mul(h1, negRea))
    @test sub(h1, pauli"X"=>0.5) == add(h1, pauli"X"=>-0.5)
    @test sub(h2, pauli"YI") == add(h2, mul(pauli"YI", negRea))
    @test sub(pauli"YI", h2) == add(mul(h2, negRea), pauli"YI")

    #>> 10. simplification=false merges equal strings all the same (canonical form), but 
    #>> retains the terms whose merged coefficient is exactly zero, per method
    resZero = sub(pauli"X", pauli"X", false)
    @test resZero.str == [pauli"X"] && resZero.coeff == Complex{Int}[0]
    @test isCanonical(resZero) && !isCanonical(resZero, true)
    @test PauliSum(resZero, true) == PauliSum(Int)
    @test countTerms(sub(h1, h1, false)) == countTerms(sub(h1, pauli"XZ"=>2, false)) == 1
    hOne = PauliSum([pauli"XZ"], [1])
    resStr = sub(hOne, pauli"XZ", false)
    @test countTerms(resStr) == 1 && resStr == sub(pauli"XZ", hOne, false) #> Zero term

    #>> 11. Round trip with indexTerm: removing a sum's own term cancels it exactly
    h3 = PauliSum([pauli"XZ"], [0.5])
    @test sub(h3, indexTerm(h3, 1)) == PauliSum(Float64)

    #>> 12. Narrow integer coefficients: promoted as in `add`, no spurious type widening
    hNarrow = PauliSum([pauli"Z"], [Int8(3)])
    @test hNarrow isa PauliSum{Int8}
    @test sub(PauliSum(Int), hNarrow) == PauliSum([pauli"Z"], [-3])
    @test sub(PauliSum(Int), hNarrow) isa PauliSum{Int}
    @test sub(hNarrow, hNarrow) == PauliSum(Int8)
    @test sub(hNarrow, pauli"Z"=>Int8(1)) == PauliSum([pauli"Z"], [Int8(2)])
    @test sub(hNarrow, pauli"Z"=>Int8(1)) isa PauliSum{Int8}
    hMin = PauliSum([pauli"Z"], [typemin(Int8)])
    @test_throws OverflowError sub(PauliSum(Int8), hMin) #> Negation overflows in `Int8`
    @test_throws OverflowError sub(pauli"X", hMin)
    @test sub(PauliSum(Int), hMin) == PauliSum([pauli"Z"], [128]) #> Negated after widening
    @test sub(hMin, PauliSum(Int8)) == hMin
    @test sub(hMin, pauli"X") == PauliSum([pauli"X", pauli"Z"], Int8[-1, typemin(Int8)])

    #>> 13. Ownership: result must not reference either input's data (buffer level)
    resOwn = sub(h1, h2)
    @test all(s -> all(t -> s.x !== t.x && s.z !== t.z, vcat(h1.str, h2.str)), resOwn.str)
    @test (resOwn.coeff !== h1.coeff) && (resOwn.coeff !== h2.coeff)

    #>> 14. Random sums with mixed site counts against the matrix reference
    bl1 = true
    for _ in 1:50
        nA = rand(rng, 1:3)
        nB = rand(rng, 1:3)
        n  = max(nA, nB)
        hA = randSum(rng, nA, 4)
        hB = randSum(rng, nB, 4)
        sB = randStr(rng, nB)
        bl1 &= (sumToMat(sub(hA, hB), n) ≈ sumToMat(hA, n) - sumToMat(hB, n))
        bl1 &= (sumToMat(sub(hA, sB), n) ≈ sumToMat(hA, n) - strToMat(sB, n))
        bl1 &= (sumToMat(sub(sB, hA), n) ≈ strToMat(sB, n) - sumToMat(hA, n))
        bl1 &= (countSites(sub(hA, hB)) == n)
    end
    @test bl1

    #> `Base.:-`: operator layer
    #>> Binary `-` on strings is equivalent to the default-`T` `sub` method
    @test pauli"X" - pauli"Z" == sub(pauli"X", pauli"Z")
    @test pauli"X" - pauli"Z" == PauliSum([pauli"X", pauli"Z"], [1, -1])
    @test pauli"X" - pauli"X" == PauliSum(Int)

    #>> Chained subtraction is left-associative (binary folds only, unlike n-ary `+`)
    @test pauli"X" - pauli"Y" - pauli"X" == PauliSum([pauli"Y"], [-1])

    #>> Sum ↔ Str in both orders are distinct; Sum - Sum routes through `sub`
    h = PauliSum([pauli"Z"], [1.0])
    @test pauli"X" - h == PauliSum([pauli"X", pauli"Z"], [1.0, -1.0])
    @test h - pauli"X" == PauliSum([pauli"X", pauli"Z"], [-1.0, 1.0])
    @test h1 - h2 == sub(h1, h2)

    #>> Mixed chain folds pairwise with correct promotion
    @test (PauliSum(Float64) - pauli"X" - pauli"Y") isa PauliSum{Float64}

    #>> Intentionally unsupported combinations stay unsupported
    @test_throws MethodError h - (pauli"Y"=>2.0)        #>> `Pair` terms are not operators
    @test_throws MethodError (pauli"Y"=>2.0) - h
    @test_throws MethodError pauli"X" - (pauli"Y"=>2.0)

    #> Type-conversion methods
    h1 = PauliSum([pauli"X"], [3])
    h2 = PauliSum([pauli"Z"], [0.5])
    @test sub(Float32, h1, h2) isa PauliSum{Float32}
    @test sub(Float32, h1, h2) == PauliSum([pauli"X", pauli"Z"], Float32[3, -0.5])
    @test sub(Float32, h2, pauli"X") == sub(PauliSum(Float32, h2), pauli"X")
    @test sub(Float32, pauli"X", h2) == pauli"X" - PauliSum(Float32, h2)
    @test sub(Float32, h2, pauli"X"=>1) == sub(PauliSum(Float32, h2), pauli"X"=>1)
    #>> A `PauliStr` takes no part in the precision level at which the terms are merged
    hThird = PauliSum([pauli"I"], [1//3])
    @test sub(Float64, pauli"I", hThird) == PauliSum([pauli"I"], [ Float64(2//3)])
    @test sub(Float64, hThird, pauli"I") == PauliSum([pauli"I"], [-Float64(2//3)])
    @test_throws MethodError sub(Bool, pauli"X", pauli"Z")
    @test_throws MethodError sub(UInt8, h1, h2)
    @test_throws MethodError sub(h1, pauli"X"=>im) #> `im` is a `Complex{Bool}`
    @test_throws ArgumentError sub(AbstractFloat, pauli"X", pauli"Z") #> Non-concrete `T`
    @test_throws ArgumentError sub(Signed, h1, h2)
    @test_throws ArgumentError sub(Rational{<:Signed}, h1, pauli"X")
    @test_throws ArgumentError sub(Rational{<:Signed}, pauli"X", h1)
end

@testset "mul" begin
    rng = Xoshiro(20260720)

    #> mul(::PhaseFactor, ::PhaseFactor)
    bl1 = true
    for l in phases, r in phases
        bl1 &= (evalPhase(mul(l, r)) == evalPhase(l) * evalPhase(r))
        bl1 &= (mul(l, r) === l * r) #> `Base.:*` overload routes to `mul`
    end
    @test bl1

    #> mul(::PauliStr, ::PauliStr)
    #>> Single-site multiplication table (spot checks with the standard results)
    @test mul(pauli"X", pauli"Y") == PauliStr([symZ], posImg) #> XY == +im*Z
    @test mul(pauli"Y", pauli"X") == PauliStr([symZ], negImg) #> YX == -im*Z
    @test mul(pauli"X", pauli"Z") == PauliStr([symY], negImg) #> XZ == -im*Y
    @test mul(pauli"Z", pauli"X") == PauliStr([symY], posImg) #> ZX == +im*Y
    @test mul(pauli"Y", pauli"Z") == PauliStr([symX], posImg) #> YZ == +im*X
    @test mul(pauli"Z", pauli"Y") == PauliStr([symX], negImg) #> ZY == -im*X

    #> Exhaustive 2-site pairs, over all input phases, against the matrix reference
    bl2 = true
    for c1 in syms, c2 in syms, p1 in phases, c3 in syms, c4 in syms, p2 in phases
        s1 = PauliStr([c1, c2], p1)
        s2 = PauliStr([c3, c4], p2)
        s3 = mul(s1, s2)
        bl2 &= (strToMat(s3) ≈ strToMat(s1) * strToMat(s2))
        bl2 &= (s3 == s1 * s2) #> `Base.:*` overload routes to `mul`
    end
    @test bl2

    #> Mixed site counts (implicit identity padding), including zero-site operands
    bl3 = true
    for n1 in 0:4, n2 in 0:4, _ in 1:4
        s1 = PauliStr(rand(rng, syms, n1), rand(rng, phases))
        s2 = PauliStr(rand(rng, syms, n2), rand(rng, phases))
        s3 = mul(s1, s2)
        n3 = max(n1, n2)
        bl3 &= (s3.n == n3)
        bl3 &= (strToMat(s3, n3) ≈ strToMat(s1, n3) * strToMat(s2, n3))
    end
    @test bl3

    #> A zero-site `PauliStr` acts as its phase times the identity
    @test mul(PauliStr(0, symI, negRea), pauli"X") == PauliStr([symX], negRea)
    @test mul(pauli"X", PauliStr(0, symI, negRea)) == PauliStr([symX], negRea)
    @test mul(PauliStr(0, symI, posImg), PauliStr(0, symI, posImg)) ==
        PauliStr(0, symI, negRea)

    #> Wide strings (multi-word buffers): self-product, order swap, and associativity
    bl4 = true
    for _ in 1:200
        n = rand(rng, 120:200)
        a = PauliStr(rand(rng, syms, n), rand(rng, phases))
        b = PauliStr(rand(rng, syms, n), rand(rng, phases))
        c = PauliStr(rand(rng, syms, n), rand(rng, phases))
        bl4 &= (mul(a, a) == PauliStr(n, symI, mul(a.phase, a.phase))) #> P*P == phase^2 * I
        swapped = checkCommute(a, b) ? mul(b, a) : scale!(mul(b, a), negRea)
        bl4 &= (mul(a, b) == swapped)                   #> ab == ±ba, sign by commutation
        bl4 &= (mul(mul(a, b), c) == mul(a, mul(b, c))) #> Associativity
    end
    @test bl4

    #> Buffer ownership: the result must not reference either input (per the docstring),
    #> including in the trivial branch taken for zero-site operands
    s1 = PauliStr(rand(rng, syms, 70), posImg)
    s2 = PauliStr(rand(rng, syms, 70), negRea)
    s3 = mul(s1, s2)
    @test (s3.x !== s1.x) && (s3.x !== s2.x) && (s3.z !== s1.z) && (s3.z !== s2.z)
    s4 = mul(PauliStr(0), s1)
    @test (s4.x !== s1.x) && (s4.z !== s1.z)

    #> mul(::PauliStr, ::PhaseFactor) and mul(::PhaseFactor, ::PauliStr)
    #>> NOTE: This block encodes the INTENDED behavior. It fails on the current `dev` due 
    #>> to the dispatch bug at Operations.jl:11 (`PauliStr(str, ::PhaseFactor)` matches no 
    #>> method); apply the fix `PauliStr(str, str.n, mul(str.phase, phase))` first.
    bl5 = true
    for p in phases, q in phases
        s = PauliStr([symX, symI, symY], p)
        r = mul(s, q)
        bl5 &= (r == PauliStr(s, s.n, mul(p, q)))
        bl5 &= (mul(s, q) == r) && (s * q == r) && (q * s == r)
        bl5 &= (r.x !== s.x) && (r.z !== s.z) #> Result owns its buffers
        bl5 &= (s.phase === p)                #> Input is not mutated
    end
    @test bl5
    @test mul(PauliStr(0, symI, posImg), posImg).phase === negRea #> im * im == -1

    #> mul(::PauliStr, ::SRealOrComplex) and the reversed order
    s = PauliStr([symX, symZ], negRea)
    h = mul(s, 2.0)
    @test h isa PauliSum{Float64}
    @test h == PauliSum([PauliStr([symX, symZ])], -2.0) #> Phase absorbed into coefficient
    @test (s * 2.0 == h) && (2.0 * s == h) #> `Base.:*` overloads route to `mul`
    @test mul(s, 1 + 2im) == PauliSum([PauliStr([symX, symZ])], -1 - 2im)
    @test mul(s, 0.0) == PauliSum(Float64)           #> Zero coefficient dropped when merged
    @test length(mul(s, 0.0, false).coeff) == 1      #> ... but kept when unsimplified

    #> mul(::PauliStr, ::PauliSum) and mul(::PauliSum, ::PauliStr)
    hY = PauliSum([pauli"Y"], 2)
    @test mul(pauli"X", hY) == PauliSum([pauli"Z"],  2im) #> X*(2Y) == +2im*Z
    @test mul(hY, pauli"X") == PauliSum([pauli"Z"], -2im) #> (2Y)*X == -2im*Z
    @test pauli"X" * hY == mul(pauli"X", hY) #> `Base.:*` overload routes to `mul`
    @test mul(PauliStr([symX], negRea), hY) == PauliSum([pauli"Z"], -2im) #> (-X)(2Y)

    #> Sidedness against the matrix reference, with mixed site counts
    bl6 = true
    for _ in 1:50
        nH = rand(rng, 1:3)
        nS = rand(rng, 1:3)
        n  = max(nH, nS)
        hStrs  = [PauliStr(rand(rng, syms, nH), rand(rng, phases)) for _ in 1:3]
        hCoeff = randn(rng, ComplexF64, 3)
        hRand  = PauliSum(hStrs, hCoeff)
        sRand  = PauliStr(rand(rng, syms, nS), rand(rng, phases))
        bl6 &= (sumToMat(mul(sRand, hRand), n) ≈ strToMat(sRand, n) * sumToMat(hRand, n))
        bl6 &= (sumToMat(mul(hRand, sRand), n) ≈ sumToMat(hRand, n) * strToMat(sRand, n))
    end
    @test bl6

    #> mul(::PauliSum, ::PauliSum)
    h1 = PauliSum([pauli"X", pauli"Y"], [1,  1])
    h2 = PauliSum([pauli"X", pauli"Y"], [1, -1])
    @test mul(h1, h2) == PauliSum([pauli"Z"], -2im) #> (X+Y)(X-Y) == -2im*Z, after merging
    @test h1 * h2 == mul(h1, h2) #> `Base.:*` overload routes to `mul`

    #> Coefficient-type promotion
    @test mul(PauliSum(Int, [pauli"X"]), PauliSum([pauli"Z"], [0.5])) isa PauliSum{Float64}

    #> `simplification=false` merges the cross terms all the same but retains the exactly 
    #> cancelled identity term (`XX - YY`); simplifying it afterwards recovers the result
    hRaw = mul(h1, h2, false)
    @test hRaw == PauliSum([pauli"I", pauli"Z"], [0, -2im], false)
    @test isCanonical(hRaw) && !isCanonical(hRaw, true)
    @test PauliSum(hRaw, true) == mul(h1, h2)

    #> The empty `PauliSum` (the zero operator) annihilates any sum
    @test mul(h1, PauliSum(Float64)) == PauliSum(Float64)
    @test mul(PauliSum(Float64), h1) == PauliSum(Float64)

    #> Random sums with mixed site counts against the matrix reference
    bl7 = true
    for _ in 1:50
        nA = rand(rng, 1:3)
        nB = rand(rng, 1:3)
        n  = max(nA, nB)
        hA = PauliSum([PauliStr(rand(rng, syms, nA), rand(rng, phases)) for _ in 1:4],
                    randn(rng, ComplexF64, 4))
        hB = PauliSum([PauliStr(rand(rng, syms, nB), rand(rng, phases)) for _ in 1:4],
                    randn(rng, ComplexF64, 4))
        bl7 &= (sumToMat(mul(hA, hB), n) ≈ sumToMat(hA, n) * sumToMat(hB, n))
        bl7 &= (countSites(mul(hA, hB)) == n)
    end
    @test bl7

    #> mul(::PauliSum, ::SRealOrComplex) and the reversed order
    hScl = PauliSum([pauli"X"], 2)
    @test mul(hScl, 2.0) == PauliSum([pauli"X"], 4.0)
    @test (hScl * 2.0 == mul(hScl, 2.0)) && (2.0 * hScl == mul(hScl, 2.0)) #> `Base.:*`

    #> Coefficient-type promotion (inexpressible via the type-preserving `scale!`)
    @test mul(hScl, 0.5) == PauliSum([pauli"X"], 1.0)
    @test mul(hScl, 0.5)      isa PauliSum{Float64}
    @test mul(hScl, 1 + 2im)  isa PauliSum{Int}     #> Complex{Int} does not widen T
    @test mul(hScl, 0.5im)    isa PauliSum{Float64}
    @test mul(hScl, 1 + 2im) == PauliSum([pauli"X"], 2 + 4im)

    #> Consistency with `scale!` where the latter is applicable (exactly convertible scalar)
    @test mul(hScl, 3) == scale!(PauliSum(hScl, true), 3)

    #> Scaling by an exact zero: terms dropped when simplified, kept when not
    @test mul(hScl, 0) == PauliSum(Int)
    @test length(mul(hScl, 0, false).coeff) == 1
    @test PauliSum(mul(hScl, 0, false), true) == PauliSum(Int)

    #> mul(::PauliSum, ::PhaseFactor) and the reversed order
    bl8 = true
    for p in phases
        r = mul(hScl, p)
        bl8 &= (r == mul(hScl, evalPhase(p)))     #> Definition via `evalPhase`
        bl8 &= (hScl * p == r) && (p * hScl == r) #> `Base.:*` overloads route to `mul`
        bl8 &= (r isa PauliSum{Int})              #> No spurious type widening
        bl8 &= (r !== hScl)                       #> New object even for the trivial phase
        bl8 &= (r.coeff !== hScl.coeff) && (r.str !== hScl.str)
    end
    @test bl8
    @test mul(hScl, posImg) == PauliSum([pauli"X"], 2im)
    @test mul(hScl, negImg) == PauliSum([pauli"X"], -2im)
    @test mul(hScl, negRea) == PauliSum([pauli"X"], -2)
    @test_throws MethodError hScl * im #> `im` is a `Complex{Bool}`, not a `Complex{Int}`
    @test hScl * 1im == mul(hScl, posImg)

    #> Non-mutation and deep buffer ownership: the input sum is untouched, and the result
    #> shares neither its coefficient storage nor any string buffer with the input
    hMul = PauliSum([PauliStr(rand(rng, syms, 5), rand(rng, phases)) for _ in 1:3],
                    randn(rng, ComplexF64, 3))
    hRef = PauliSum(hMul, true)
    rMul = mul(hMul, 2.0)
    @test hMul == hRef                            #> Input unchanged
    @test (rMul.coeff !== hMul.coeff) && (rMul.str !== hMul.str)
    @test (rMul.str[begin].x !== hMul.str[begin].x) &&
        (rMul.str[begin].z !== hMul.str[begin].z)

    #> Random multi-term sums against the matrix reference
    bl9 = true
    for _ in 1:50
        n = rand(rng, 1:3)
        hR = PauliSum([PauliStr(rand(rng, syms, n), rand(rng, phases)) for _ in 1:4],
                    randn(rng, ComplexF64, 4))
        c = randn(rng, ComplexF64)
        p = rand(rng, phases)
        bl9 &= (sumToMat(mul(hR, c), n) ≈ c .* sumToMat(hR, n))
        bl9 &= (sumToMat(mul(hR, p), n) ≈ evalPhase(p) .* sumToMat(hR, n))
    end
    @test bl9

    #> Family consistency: string-scalar and sum-scalar products agree
    @test mul(pauli"X", 2.0) == mul(PauliSum([pauli"X"], 1), 2.0)

    #> Type-conversion methods
    h1 = PauliSum([pauli"X"], [4])
    h2 = PauliSum([pauli"Z"], [0.6])
    @test mul(Float32, h1, h2)         isa PauliSum{Float32}
    @test mul(Float32, pauli"X", h2)   isa PauliSum{Float32}
    @test mul(Float32, h2, pauli"X")   isa PauliSum{Float32}
    @test mul(Float32, h2, posImg)     isa PauliSum{Float32}
    @test_throws InexactError mul(Int, h2, 2) #> 0.6 not representable in Int
    @test mul(h1, h2)                  isa PauliSum{Float64}
    h3 = PauliSum([pauli"X"], [5.952419006512908])
    @test mul(Float32, h3, 3).coeff[] == 17.857258f0 #> Less-accurate type imposes lastly
    h4 = PauliSum([pauli"X"], [2^40])
    @test mul(PauliSum(Float64, h4), h4) == PauliSum([PauliStr(1)], [2.0^80])
    @test mul(h4, 2.0^40) == PauliSum([pauli"X"],    [2.0^80])
    @test_throws InexactError mul(Int32, PauliSum([pauli"X"], [2^20]), 
                                         PauliSum([pauli"X"], [2^20]))
    #> An exact evaluation type is kept exact: integer overflow throws instead of wrapping 
    #> around (even when the result type is a float, as the products are evaluated in `Int`)
    @test_throws OverflowError mul(Float64, h4, h4)
    @test_throws OverflowError mul(Float64, h4, 2^40)
    @test_throws OverflowError mul(h4, h4)
    @test_throws OverflowError mul(PauliSum([pauli"X"], Int8[100]), Int8(2))
    @test mul(PauliSum([pauli"X"], Int8[100]), 2) == PauliSum([pauli"X"], [200])
    #> A phase is applied in the coefficient type of `h`, so negating `typemin` overflows
    @test_throws OverflowError mul(PauliSum([pauli"X"], [typemin(Int)]), negRea)
    @test_throws OverflowError mul(PauliSum([pauli"X"], [typemin(Int8)]), negRea)
    @test mul(Int, PauliSum([pauli"X"], [typemin(Int8)]), negRea) == 
          PauliSum([pauli"X"], [128])
end

@testset "scale!(::PauliSum)" begin
    h = PauliSum([pauli"X", pauli"Z"], [2, -3])
    @test scale!(h, 2) === h && h == PauliSum([pauli"X", pauli"Z"], [4, -6])
    @test scale!(h, 1im) === h && h == PauliSum([pauli"X", pauli"Z"], [4im, -6im])
    @test scale!(h, negImg) === h && h == PauliSum([pauli"X", pauli"Z"], [4, -6])
    @test scale!(h, posRea) === h && h == PauliSum([pauli"X", pauli"Z"], [4, -6])
    #> Each product is converted back to the coefficient type of `h`
    hF = PauliSum([pauli"X"], Float32[1.5])
    @test scale!(hF, 2.0) isa PauliSum{Float32} && hF.coeff == [3f0]
    @test_throws InexactError scale!(PauliSum([pauli"X"], [3]), 0.5)
    @test scale!(PauliSum([pauli"X"], [1//2]), 3) == PauliSum([pauli"X"], [3//2])
    #> Exact coefficient types: an overflow (`OverflowError`) or a narrowing failure 
    #> (`InexactError`) is detected before `h` is mutated, so `h` is left unchanged. The 
    #> failing term is the second one in the canonical order (`Z` precedes `X`), so that a 
    #> term-by-term check would have already scaled the first term when it throws
    hMax = PauliSum([pauli"X", pauli"Z"], [typemax(Int), 1])
    @test_throws OverflowError scale!(hMax, 2)
    @test hMax.coeff == [1, typemax(Int)]
    @test hMax == PauliSum([pauli"X", pauli"Z"], [typemax(Int), 1])
    @test_throws OverflowError scale!(PauliSum([pauli"X"], [typemin(Int)]), negRea)
    h8 = PauliSum([pauli"X", pauli"Z"], Int8[100, 1])
    @test_throws InexactError scale!(h8, 2)
    @test h8.coeff == Int8[1, 100]
    @test h8 == PauliSum([pauli"X", pauli"Z"], Int8[100, 1])
    @test_throws OverflowError scale!(PauliSum([pauli"X"], [typemin(Int8)]), negRea)
end

#> `checkCommute` and `checkAntiCom`
@test checkCommute(pauli"Y", pauli"Y")
@test checkCommute(pauli"IY", pauli"IY")
bl1 = true
for P in (pauli"X", pauli"Y", pauli"Z", pauli"XY", pauli"YZ", pauli"XYZ")
    bl1 *=  checkCommute(P, P)   #>> an operator always commutes with itself
    bl1 *= !checkAntiCom(P, P)
end
@test bl1

#>> Two anticommuting clashes cancel: XX and ZZ commute
#>>> Sites 1 and 2 each contribute an X/Z clash
@test checkCommute(pauli"XX", pauli"ZZ")      # was false (BUG)
@test checkCommute(pauli"XX", pauli"YY")      # was false (BUG): count == 2
@test checkCommute(pauli"YY", pauli"ZZ")      # was false (BUG): count == 2

#>> A single clash still (correctly) anticommutes
@test checkAntiCom(pauli"X", pauli"Z")
@test checkAntiCom(pauli"X", pauli"Y")
@test checkAntiCom(pauli"XXIZ", pauli"XXIY") # one Z/Y clash on site 4

#>> Exhaustive cross-check on all 2-qubit Pauli pairs against the
#>>> Symplectic parity rule (the ground truth for Pauli commutation)
twoSite = [pauli"II", pauli"IX", pauli"IY", pauli"IZ",
           pauli"XI", pauli"XX", pauli"XY", pauli"XZ",
           pauli"YI", pauli"YX", pauli"YY", pauli"YZ",
           pauli"ZI", pauli"ZX", pauli"ZY", pauli"ZZ"]

# Reference: parity of the symplectic inner product computed independently
sympl(a, b) = sum(count_ones.(a.z .& b.x) .+ count_ones.(a.x .& b.z))
bl2 = true
for a in twoSite, b in twoSite
    bl2 *= checkCommute(a, b) == iseven(sympl(a, b))
    bl2 *= checkAntiCom(a, b) == isodd(sympl(a, b))
end
@test bl2

#>> For `PauliSum` operands, both judgments are made at the value level: they depend on 
#>> the two products only, even when the difference or the sum of them is not representable
hLow = PauliSum([pauli"Z"], [typemin(Int)])
hTop = PauliSum([pauli"Z"], [typemax(Int)])
@test  checkCommute(hLow, pauli"Z") && !checkAntiCom(hLow, pauli"Z")
@test  checkCommute(pauli"Z", hTop) && !checkAntiCom(pauli"Z", hTop)
@test !checkCommute(hTop, pauli"X") &&  checkAntiCom(hTop, pauli"X")
#>> The evaluations can still throw where the judgments are made: the sum of the two 
#>> products overflows in `evalAntiCom`, and so does the negation of one in `evalCommute`
@test_throws OverflowError evalAntiCom(hLow, pauli"Z")
@test_throws OverflowError evalCommute(hLow, pauli"Z")

@testset "evalCommute/evalAntiCom" begin
    @test evalCommute(pauli"X", pauli"Z") == PauliSum([pauli"Y"], -2im)
    @test evalCommute(pauli"X", pauli"Z") isa PauliSum{Int}
    @test evalCommute(pauli"X", pauli"X") == PauliSum(Int)           #> zero commutator
    @test evalAntiCom(pauli"X", pauli"X") == PauliSum([pauli"I"], 2) #> {X,X} = 2I, merged
    @test evalAntiCom(pauli"X", pauli"Z") == PauliSum(Int)           #> zero anticommutator
    #> `onlyFinite` is accepted by every method, and has no effect on two `PauliStr`
    pX, pZ = pauli"X", pauli"Z"
    for flag in (true, false)
        @test evalCommute(pX, pZ, flag) == PauliSum([pauli"Y"], -2im)
        @test evalAntiCom(pX, pX, flag) == PauliSum([pauli"I"], 2)
        @test  checkCommute(pX, pX, flag) && !checkCommute(pX, pZ, flag)
        @test  checkAntiCom(pX, pZ, flag) && !checkAntiCom(pX, pX, flag)
    end

    #> Fixed exact values (matrix-validated): implicit identity-padding across site counts
    hX  = PauliSum(Int, [pauli"X"])  #> 1 explicit site, padded against 2-site operand
    hZZ = PauliSum(Int, [pauli"ZZ"])
    @test evalAntiCom(hX, hZZ) == PauliSum(Int) #> Anticommuting strings: {XI, ZZ} == 0
    @test evalCommute(hX, hZZ) == PauliSum([pauli"YZ"], Complex{Int}(0, -2))

    #> Multi-term exact values (matrix-validated): 
    #>> h1 = XI + 2 ZZ, h2 = YI - XZ
    #>> {h1, h2} == -2 IZ,    [h1, h2] == 2im ZI - 4im YI - 4im XZ
    h1 = PauliSum([pauli"XI", pauli"ZZ"], Complex{Int}[1, 2])
    h2 = PauliSum([pauli"YI", pauli"XZ"], Complex{Int}[1, -1])
    @test evalAntiCom(h1, h2) == PauliSum([pauli"IZ"], Complex{Int}(-2))
    @test evalCommute(h1, h2) == 
          PauliSum([pauli"ZI", pauli"YI", pauli"XZ"], Complex{Int}[2im, -4im, -4im])
    @test evalCommute(h1, h2, false) == evalCommute(h1, h2)
    @test evalAntiCom(h1, h2, false) == evalAntiCom(h1, h2)
    @test evalCommute(pauli"XI", h2, false) == evalCommute(pauli"XI", h2)
    @test evalAntiCom(h1, pauli"YI", false) == evalAntiCom(h1, pauli"YI")
    @test !checkCommute(h1, h2, false) && !checkAntiCom(h1, h2, false)

    #> Zero-site phased unit is central: commutes with everything, {u, h} == 2h
    u = PauliSum(Int, [PauliStr()])
    @test isempty(evalCommute(u, h1).str)
    @test evalAntiCom(u, h1) == mul(h1, 2)

    #> Zero-operand behavior and coefficient-type promotion
    @test evalAntiCom(h1, PauliSum(Int)) == PauliSum(Int)
    @test evalCommute(h1, PauliSum(Int)) == PauliSum(Int)
    hRat = PauliSum([pauli"XI"], Complex{Rational{Int}}(1//2, 0))
    @test evalAntiCom(hRat, mul(h2, 1.0)) isa PauliSum{Float64}

    #> A commutator that is not representable throws instead of wrapping around, whereas 
    #> whether the operands commute can still be judged
    hIZ = PauliSum([pauli"I", pauli"Z"], Int8[1, 1])
    hXY = PauliSum([pauli"X", pauli"Y"], Complex{Int8}[-64im, -64])
    @test isempty(mul(hIZ, hXY).str)
    @test mul(hXY, hIZ) == PauliSum([pauli"X", pauli"Y"], Complex{Int8}[-128im, -128])
    @test_throws OverflowError evalCommute(hIZ, hXY)
    @test !checkCommute(hIZ, hXY) && !checkAntiCom(hIZ, hXY)
    #>> A product that is not representable throws in the judgments as well, whichever 
    #>> value `onlyFinite` takes
    hOver = PauliSum([pauli"Z"], Int8[100])
    @test_throws OverflowError checkCommute(hOver, PauliSum([pauli"Z"], Int8[2]))
    @test_throws OverflowError checkAntiCom(hOver, hOver, false)
    @test checkCommute(hOver, hOver) #> Special case: equal terms passing to `checkCommute`
    @test checkCommute(hOver, PauliSum(hOver), false)
    #>> In this case, `evalCommute` returns the zero operator, which has the same core type 
    #>> as any other result of the same method
    @test evalCommute(hOver, hOver) == evalCommute(PauliSum(hOver), hOver) == PauliSum(Int8)
    @test evalCommute(hOver, hOver) isa PauliSum{Int8}
    hZf, hZi = PauliSum([pauli"Z"], [2.0]), PauliSum([pauli"Z"], [2])
    @test checkCommute(hZf, hZi) && checkCommute(pauli"Z", PauliSum([pauli"Z"], Int8[1]))
    @test evalCommute(hZf, hZi) isa PauliSum{Float64}
    @test evalCommute(hZi, hZf) isa PauliSum{Float64}
    @test evalCommute(PauliSum([pauli"Z"], [1.0]), pauli"Z") isa PauliSum{Float64}
    @test evalCommute(pauli"Z", PauliSum([pauli"Z"], Int8[1])) isa PauliSum{Int8}

    #> Property fuzz: string-level consistency, (anti)symmetry, adjoint interplay, 
    #> the decomposition {h1,h2} + [h1,h2] == 2 h1 h2, the Jacobi identity, and 
    #> matrix ground truth; all comparisons exact except the matrix references
    rng = Xoshiro(44)
    bl = true
    for _ in 1:64
        n = rand(rng, 1:3)
        s1, s2 = randStr(rng, n), randStr(rng, n)
        bl &= (evalCommute(PauliSum(Int, [s1]), PauliSum(Int, [s2])) == evalCommute(s1, s2))
        bl &= (evalAntiCom(PauliSum(Int, [s1]), PauliSum(Int, [s2])) == evalAntiCom(s1, s2))
        hS1, hS2 = PauliSum(Int, [s1]), PauliSum(Int, [s2])
        bl &= (checkCommute(hS1, hS2) == checkCommute(s1, hS2) == checkCommute(s1, s2))
        bl &= (checkAntiCom(hS1, hS2) == checkAntiCom(hS1, s2) == checkAntiCom(s1, s2))

        hA = randSum(rng, n, rand(rng, 0:4))
        hB = randSum(rng, n, rand(rng, 0:4))
        hC = randSum(rng, n, rand(rng, 0:4))
        cmAB = evalCommute(hA, hB)
        acAB = evalAntiCom(hA, hB)
        bl &= (checkCommute(hA, hB) == isempty(cmAB.str))
        bl &= (checkAntiCom(hA, hB) == isempty(acAB.str))
        bl &= (acAB == evalAntiCom(hB, hA))
        bl &= (cmAB == mul(evalCommute(hB, hA), -1))
        bl &= (isempty(evalCommute(hA, hA).str))
        bl &= (evalAntiCom(hA, hA) == mul(mul(hA, hA), 2))
        bl &= (toAdjoint(acAB) == evalAntiCom(hA', hB'))
        bl &= (toAdjoint(cmAB) == mul(evalCommute(hA', hB'), -1))
        bl &= (PauliSum(vcat(acAB.str, cmAB.str), vcat(acAB.coeff, cmAB.coeff)) == 
               mul(mul(hA, hB), 2))

        #> `onlyFinite` makes no difference to finite operands whose products are finite
        gA, gB = PauliSum(Float64, hA), PauliSum(Float64, hB)
        bl &= (evalCommute(gA, gB) == evalCommute(gA, gB, false) == PauliSum(Float64, cmAB))
        bl &= (evalAntiCom(gA, gB) == evalAntiCom(gA, gB, false) == PauliSum(Float64, acAB))
        bl &= (checkCommute(gA, gB) == checkCommute(gA, gB, false) == checkCommute(hA, hB))
        bl &= (checkAntiCom(gA, gB) == checkAntiCom(gA, gB, false) == checkAntiCom(hA, hB))
        #> Once every product of two nonzero coefficients overflows, the judgments under 
        #> `onlyFinite=false` are `false` unless either operand is the zero operator
        fA, fB = mul(hA, 2.0^600), mul(hB, 2.0^500)
        anyZero = isempty(hA.str) || isempty(hB.str)
        bl &= (checkCommute(fA, fB, false) == checkAntiCom(fA, fB, false) == anyZero)
        #> A finite operator commutes with itself whether or not its square overflows
        bl &= (checkCommute(fA, fA) && evalCommute(fA, fA) == PauliSum(Float64))

        bl &= (sumToMat(cmAB, n) ≈ 
               sumToMat(hA, n) * sumToMat(hB, n) - sumToMat(hB, n) * sumToMat(hA, n))
        bl &= (sumToMat(acAB, n) ≈ 
               sumToMat(hA, n) * sumToMat(hB, n) + sumToMat(hB, n) * sumToMat(hA, n))

        #> Jacobi identity: [[A,B],C] + [[B,C],A] + [[C,A],B] == 0
        j1 = evalCommute(evalCommute(hA, hB), hC)
        j2 = evalCommute(evalCommute(hB, hC), hA)
        j3 = evalCommute(evalCommute(hC, hA), hB)
        bl &= isempty(PauliSum(vcat(j1.str, j2.str, j3.str), 
                               vcat(j1.coeff, j2.coeff, j3.coeff)).str)
    end
    @test bl

    #> By default (`onlyFinite=true`), an operand that holds a non-finite coefficient is 
    #> rejected by the judgments and the evaluations alike. With `onlyFinite=false`, the 
    #> judgments return `false`, whereas the evaluations let the non-finite coefficients 
    #> propagate
    hTwo = PauliSum([pauli"Z"], [2.0])
    for hBad in (PauliSum([pauli"Z"], [Inf]), PauliSum([pauli"Z"], [NaN]))
        @test_throws ArgumentError checkCommute(hBad, hTwo)
        @test_throws ArgumentError checkCommute(hTwo, hBad)
        @test_throws ArgumentError checkAntiCom(hBad, pauli"X")
        @test_throws ArgumentError evalCommute(pauli"Z", hBad)
        @test_throws ArgumentError evalAntiCom(hBad, hTwo)
        @test !checkCommute(hBad, hTwo, false) && !checkCommute(hTwo, hBad, false)
        @test !checkCommute(hBad, hBad, false) && !checkAntiCom(hBad, pauli"X", false)
        @test isnan(evalCommute(hBad, hTwo, false).coeff[])
        @test !isfinite(evalAntiCom(hBad, hTwo, false).coeff[])
        #>> Nor does such an operand take the special case of equal operands
        @test isnan(evalCommute(hBad, hBad, false).coeff[])
        @test isnan(evalCommute(hBad, PauliSum(hBad), false).coeff[])
    end
    #>> Hence, an unvalidated evaluation can return the zero operator where the judgment is 
    #>> `false`
    hInf = PauliSum([pauli"Z"], [Inf])
    @test isempty(evalCommute(hInf, PauliSum(Float64), false).str)
    @test !checkCommute(hInf, PauliSum(Float64), false)

    #> Finite operands whose products overflow are handled in the same way
    hXs = PauliSum([pauli"XI", pauli"XZ"], [1e154, 1e154])
    hZs = PauliSum([pauli"ZI", pauli"ZZ"], [1e154, 1e154])
    @test mul(hXs, hZs) == PauliSum([pauli"YI", pauli"YZ"], fill(Complex(0.0, -Inf), 2))
    @test mul(hZs, hXs) == PauliSum([pauli"YI", pauli"YZ"], fill(Complex(0.0, Inf), 2))
    @test_throws ArgumentError checkAntiCom(hXs, hZs)
    @test_throws ArgumentError checkCommute(hXs, hZs)
    @test_throws ArgumentError evalAntiCom(hXs, hZs)
    @test_throws ArgumentError evalCommute(hXs, hZs)
    @test !checkAntiCom(hXs, hZs, false) && !checkCommute(hXs, hZs, false)
    acXZ = evalAntiCom(hXs, hZs, false) #>> `Inf + (-Inf)` leaves a `NaN` in each term
    @test countTerms(acXZ) == 2 && all(isnan, acXZ.coeff)
    @test evalCommute(hXs, hZs, false) == mul(hXs, hZs)
    #>> The self-commutator is valid as long as no non-finite coefficients exist initially
    hBig = PauliSum([pauli"X", pauli"Z"], [1e200, 1e200])
    @test checkCommute(hBig, hBig)
    @test checkCommute(hBig, hBig, false)
    @test evalCommute(hBig, hBig) == evalCommute(hBig, hBig, false) == PauliSum(Float64)
    @test_throws ArgumentError checkAntiCom(hBig, hBig)
    @test_throws ArgumentError evalAntiCom(hBig, hBig)
    #>> The same applies to other floating-point types and to mixed coefficient types
    hBig32 = PauliSum([pauli"X", pauli"Z"], Float32[1f30, 1f30])
    @test checkCommute(hBig32, hBig32)
    @test checkCommute(hBig32, hBig32, false)
    @test checkCommute(hBig32, PauliSum(Float64, hBig32))
    #>> Different operators will throw by default
    hHuge = PauliSum([pauli"Z"], [1e300])
    @test_throws ArgumentError checkCommute(hTop, hHuge)
    @test !checkCommute(hTop, hHuge, false) && !checkAntiCom(hHuge, hTop, false)
    #>> Two objects holding the same operator count as equal operands, whether or not they 
    #>> store it in the canonical form
    hPerm = PauliSum(hBig)
    @test checkCommute(hBig, hPerm) && checkCommute(hPerm, hBig, false)
    #>> Both coefficients are equal, so swapping the strings only changes the term order
    hPerm.str[begin], hPerm.str[end] = hPerm.str[end], hPerm.str[begin]
    @test !isCanonical(hPerm)
    @test checkCommute(hBig, hPerm) && checkCommute(hPerm, hBig, false)
    @test evalCommute(hPerm, hBig) == evalCommute(hBig, hPerm, false) == PauliSum(Float64)
    #>> For floating-point core data types, a canonical form that merges terms is not an 
    #>> exact image of its operand, since the merge can round (`hRnd`) or overflow (`hDup1` 
    #>> and `hDup2`). Operands that are equal only in such forms are left to their 
    #>> products, unless they are the same object
    hRnd = PauliSum([pauli"Z", pauli"X", pauli"Y"], [1.0, 1.0, 2.0^-60])
    stamp!(indexTerm(hRnd, 3, false).first, 1, symX) #> The third stored term is `Y`
    hMerged = PauliSum([pauli"Z", pauli"X"], [1.0, 1.0])
    @test PauliSum(hRnd, false) == hMerged
    @test !checkCommute(hRnd, hMerged) && !checkCommute(hMerged, hRnd, false)
    @test  checkCommute(hRnd, hRnd) && evalCommute(hRnd, hRnd) == PauliSum(Float64)
    @test evalCommute(hRnd, hMerged) == PauliSum([pauli"Y"], [Complex(0.0, -2.0^-59)])
    hDup1 = PauliSum([pauli"X", pauli"Y", pauli"Z"], [1e308, 1.0, 1e308])
    hDup2 = PauliSum([pauli"X", pauli"Y", pauli"Z"], [1e308, 1.0, 1.5e308])
    for hDup in (hDup1, hDup2)
        stamp!(indexTerm(hDup, 1, false).first, 1, symX) #> The first stored term is `Z`
    end
    @test PauliSum(hDup1, false) == PauliSum(hDup2, false) == 
          PauliSum([pauli"X", pauli"Y"], [Inf, 1.0])
    @test_throws ArgumentError checkCommute(hDup1, hDup2)
    @test_throws ArgumentError evalCommute(hDup1, hDup2)
    @test !checkCommute(hDup1, hDup2, false)
    @test  checkCommute(hDup1, hDup1) && evalCommute(hDup1, hDup1) == PauliSum(Float64)
    #>> For exact core data types, the merge is exact, so operands that are equal in their 
    #>> canonical forms count as equal operands: `hHalves` stores `Z` twice, and its 
    #>> product with `hOver` is not representable
    hHalves = PauliSum([pauli"Z", pauli"X"], Int8[50, 50])
    stamp!(indexTerm(hHalves, 2, false).first, 1, symZ) #> The second stored term is `X`
    @test PauliSum(hHalves) == hOver && !isCanonical(hHalves)
    @test checkCommute(hHalves, hOver) && checkCommute(hOver, hHalves, false)
    @test evalCommute(hOver, hHalves) == PauliSum(Int8)
    @test evalCommute(hHalves, hOver) isa PauliSum{Int8}

    #> The zero operator (anti)commutes with every finite operator, whichever value 
    #> `onlyFinite` takes; a floating-point operand counts as one when every coefficient it 
    #> stores is zero
    hZeroTerm = PauliSum([pauli"X"], [0.0], false)
    for hZero in (PauliSum(Float64), hZeroTerm), flag in (true, false)
        @test checkCommute(hZero, hBig, flag) && checkCommute(hBig, hZero, flag)
        @test checkAntiCom(hZero, hBig, flag) && checkAntiCom(hBig, hZero, flag)
        @test evalCommute(hZero, hBig, flag) == evalAntiCom(hBig, hZero, flag) == 
              PauliSum(Float64)
    end
    @test_throws ArgumentError checkAntiCom(PauliSum(Float64), hInf)
    @test !checkAntiCom(PauliSum(Float64), hInf, false)
    #>> Neither the judgments nor the evaluations depend on the order of the operands, also 
    #>> when the terms of one of them cannot be merged at its core data type: `hOverDup` 
    #>> stores `X` twice, and so does `hZeroDup` (with zero coefficients)
    hOverDup = PauliSum([pauli"X", pauli"Z"], Int8[100, 100])
    hZeroDup = PauliSum([pauli"X", pauli"Z"], [0.0, 0.0], false)
    for hDup in (hOverDup, hZeroDup)
        stamp!(indexTerm(hDup, 1, false).first, 1, symX) #> The first stored term is `Z`
    end
    #>> Neither `checkAntiCom` nor `evalAntiCom` makes a canonical-form copy of an operand: 
    #>> the two agree on `hOverDup` paired with the zero operator, or with an operand of a 
    #>> wider core data type that represents both of their products
    for hOther in (PauliSum(Int8), PauliSum([pauli"Z"], Int16[1]))
        @test checkAntiCom(hOther, hOverDup) && checkAntiCom(hOverDup, hOther)
        @test countTerms(evalAntiCom(hOther, hOverDup)) == 0
        @test countTerms(evalAntiCom(hOverDup, hOther)) == 0
    end
    #>> `checkCommute` and `evalCommute` first compare the canonical forms of both 
    #>> operands, which are formed at the promoted core data type. With a partner of its 
    #>> own core data type, `hOverDup` cannot form one (`100 + 100` is not an `Int8`), so 
    #>> they reject it wherever it is placed
    hNone8 = PauliSum(Int8)
    for (opL, opR) in ((hNone8, hOverDup), (hOverDup, hNone8))
        @test_throws OverflowError checkCommute(opL, opR)
        @test_throws OverflowError evalCommute(opL, opR)
    end
    #>> With an `Int16` partner, the canonical form of `hOverDup` is `200 X`, and the 
    #>> products decide when the two forms differ: [200 X, Z] == -400im Y
    hWideZ = PauliSum([pauli"Z"], Int16[1])
    @test !checkCommute(hOverDup, hWideZ) && !checkCommute(hWideZ, hOverDup)
    @test evalCommute(hOverDup, hWideZ) == PauliSum([pauli"Y"], Complex{Int16}[-400im])
    @test evalCommute(hWideZ, hOverDup) == PauliSum([pauli"Y"], Complex{Int16}[400im])
    #>> `hWideX` has the same canonical form, so the two count as equal operands, although 
    #>> their product is not representable
    hWideX = PauliSum([pauli"X"], Int16[200])
    @test_throws OverflowError mul(hOverDup, hWideX)
    @test checkCommute(hOverDup, hWideX) && checkCommute(hWideX, hOverDup)
    @test evalCommute(hOverDup, hWideX) == evalCommute(hWideX, hOverDup) == PauliSum(Int16)
    #>> With a floating-point partner, no terms are merged for the comparison: `hOverDup` 
    #>> is left to the products, like a floating-point operand that stores `X` twice
    for (opL, opR) in ((hZeroDup, hOverDup), (hOverDup, hZeroDup))
        @test checkCommute(opL, opR) && countTerms(evalCommute(opL, opR)) == 0
    end
    hHalfZ = PauliSum([pauli"Z"], [0.5])
    @test !checkCommute(hOverDup, hHalfZ) && !checkCommute(hHalfZ, hOverDup)
    @test evalCommute(hOverDup, hHalfZ) == PauliSum([pauli"Y"], [Complex(0.0, -200.0)])
    @test evalCommute(hHalfZ, hOverDup) == PauliSum([pauli"Y"], [Complex(0.0, 200.0)])
    #>> Coefficients that only cancel out once equal strings are merged do not make a 
    #>> floating-point operand the zero operator: such a merge is subject to rounding, and 
    #>> the evaluations work on the stored terms. `hNull` stores `X` three times, and the 
    #>> judgments on it stay consistent with the evaluations
    hNull = PauliSum([pauli"X", pauli"Y", pauli"Z"], [1.0, 2.0, -3.0])
    fill!(hNull.str, pauli"X")
    @test PauliSum(hNull) == PauliSum(Float64)
    for hTenth in (PauliSum([pauli"Z"], [0.1]), PauliSum([pauli"X"], [0.1]))
        @test checkCommute(hNull, hTenth) == (countTerms(evalCommute(hNull, hTenth)) == 0)
        @test checkAntiCom(hNull, hTenth) == (countTerms(evalAntiCom(hNull, hTenth)) == 0)
    end
    @test !checkCommute(hNull, PauliSum([pauli"Z"], [0.1]))
    @test !checkAntiCom(hNull, PauliSum([pauli"X"], [0.1]))
    #>> `hTinyX` stores `XI` six times with coefficients that add up to `2.0^-60`, which 
    #>> its merge rounds to zero
    hTinyX = PauliSum([pauli"XI", pauli"YI", pauli"ZI", pauli"XX", pauli"XY", pauli"XZ"], 
                      [2.0^-60, -2.0^52 - 2, -2.0^53, -3 * 2.0^52, 3 * 2.0^52, 
                       3 * 2.0^52 + 2])
    fill!(hTinyX.str, pauli"XI")
    hOneZ = PauliSum([pauli"ZI"], [1.0])
    @test PauliSum(hTinyX) == PauliSum(Float64)
    @test !checkCommute(hTinyX, hOneZ) && !checkCommute(hOneZ, hTinyX, false)
    @test checkCommute(hTinyX, hOneZ) == (countTerms(evalCommute(hTinyX, hOneZ)) == 0)
    @test checkAntiCom(hTinyX, hOneZ) == (countTerms(evalAntiCom(hTinyX, hOneZ)) == 0)
    #>> For exact core data types, the same layout is the zero operator
    hNullInt = PauliSum([pauli"X", pauli"Y", pauli"Z"], [1, 2, -3])
    fill!(hNullInt.str, pauli"X")
    hOneInt = PauliSum([pauli"Z"], [1])
    @test checkCommute(hNullInt, hOneInt) && checkAntiCom(hOneInt, hNullInt)
    @test evalCommute(hNullInt, hOneInt) == evalAntiCom(hOneInt, hNullInt) == PauliSum(Int)

    #> Equal (or opposite) infinite coefficients in the two products are never taken as 
    #> equal (or opposite) products: [hEq1, hEq2] == 1.35e308 (Z - X), and {hOpp1, hOpp2} 
    #> is about 1.6e308 (XI + XZ + ZI + ZZ)
    hEq1 = PauliSum([pauli"I", pauli"Y"], [1.35e154, 0.5e154im])
    hEq2 = PauliSum([pauli"X", pauli"Z"], [1.35e154, 1.35e154])
    @test mul(hEq1, hEq2) == mul(hEq2, hEq1) == PauliSum([pauli"X", pauli"Z"], [Inf, Inf])
    @test_throws ArgumentError checkCommute(hEq1, hEq2)
    @test !checkCommute(hEq1, hEq2, false)
    hOpp1 = PauliSum([pauli"II", pauli"YI", pauli"YZ"], [4.70588e153, 1e154im, 1e154im])
    hOpp2 = PauliSum([pauli"XI", pauli"XZ", pauli"ZI", pauli"ZZ"], fill(1.7e154, 4))
    @test all(isinf, mul(hOpp1, hOpp2).coeff)
    @test mul(hOpp1, hOpp2).coeff == map(-, mul(hOpp2, hOpp1).coeff)
    @test_throws ArgumentError checkAntiCom(hOpp1, hOpp2)
    @test !checkAntiCom(hOpp1, hOpp2, false)

    #> A product can overflow in one order only (`hP * hQ` is the zero operator), which is 
    #> caught for either order of the operands
    hP = PauliSum([pauli"I", pauli"Y"], [1e154, -1e154])
    hQ = PauliSum([pauli"X", pauli"Z"], [1e154im, 1e154])
    @test isempty(mul(hP, hQ).str)
    @test mul(hQ, hP) == 
          PauliSum([pauli"X", pauli"Z"], [Complex(0.0, Inf), Complex(Inf, 0.0)])
    @test_throws ArgumentError checkCommute(hP, hQ)
    @test_throws ArgumentError evalCommute(hQ, hP)
    @test_throws ArgumentError evalAntiCom(hP, hQ)
    @test !checkCommute(hP, hQ, false) && !checkAntiCom(hQ, hP, false)

    #> Only the operands and the two products are validated: the difference of two finite 
    #> products can still overflow in the returned commutator
    hRootX, hRootZ = PauliSum([pauli"X"], [1e154]), PauliSum([pauli"Z"], [1e154])
    @test evalCommute(hRootX, hRootZ) == PauliSum([pauli"Y"], [Complex(0.0, -Inf)])
    @test !checkCommute(hRootX, hRootZ)
    @test  checkAntiCom(hRootX, hRootZ) && isempty(evalAntiCom(hRootX, hRootZ).str)
end


@testset "checkAdjoint" begin
    #> Two `PauliStr`: equal single-site operators and site counts, conjugate phases
    for (phase, phaseConj) in ((posRea, posRea), (posImg, negImg), 
                               (negRea, negRea), (negImg, posImg))
        s = PauliStr(2, symX, phase)
        @test  checkAdjoint(s, PauliStr(2, symX, phaseConj))
        @test !checkAdjoint(s, PauliStr(2, symY, phaseConj))
        @test  checkAdjoint(s, s) == isHermitian(s) == (phase == phaseConj)
    end

    #> No implicit identity padding, unlike `checkCommute`
    @test !checkAdjoint(pauli"X", pauli"XI") && checkCommute(pauli"X", pauli"XI")
    @test !checkAdjoint(pauli"X", PauliSum([pauli"XI"], [1]))
    @test !checkAdjoint(PauliSum([pauli"X"], [1]), PauliSum([pauli"XI"], [1]))

    #> Agreement with `toAdjoint` on random strings, for either order of the operands
    rng = Xoshiro(44)
    bl = true
    for _ in 1:128
        s1, s2 = randStr(rng, rand(rng, 0:3)), randStr(rng, rand(rng, 0:3))
        bl &= checkAdjoint(s1, s1') && checkAdjoint(s1', s1)
        bl &= (checkAdjoint(s1, s2) == (s1' == s2) == checkAdjoint(s2, s1))
        #>> A `PauliStr` is compared with a `PauliSum` as a single term
        bl &= checkAdjoint(s1, PauliSum([s1'], 1)) && checkAdjoint(PauliSum([s1'], 1), s1)
        bl &= (checkAdjoint(s1, PauliSum([s2], 1)) == (s1' == s2))
    end
    @test bl

    #> The same agreement on random sums (in the simplified form); identical objects give 
    #> the same answer as `isHermitian`
    rng = Xoshiro(45)
    bl = true
    for _ in 1:64
        n = rand(rng, 1:3)
        hA, hB = randSum(rng, n, rand(rng, 0:4)), randSum(rng, n, rand(rng, 0:4))
        bl &= checkAdjoint(hA, hA') && checkAdjoint(hA', hA)
        bl &= (checkAdjoint(hA, hA) == isHermitian(hA) == (hA == hA'))
        bl &= (checkAdjoint(hA, hB) == (hA' == hB) == checkAdjoint(hB, hA))
    end
    @test bl

    #> The coefficients are compared by value across different core data types, and the 
    #> terms with zero coefficients are disregarded
    hInt = PauliSum([pauli"X", pauli"Y"], [2, 3im])
    hRat = PauliSum([pauli"X", pauli"Y"], Complex{Rational{Int}}[2, -3im])
    @test  checkAdjoint(hInt, hRat) && checkAdjoint(hRat, hInt)
    @test  checkAdjoint(hInt, PauliSum([pauli"X", pauli"Y"], [2.0, -3.0im]))
    @test !checkAdjoint(hInt, PauliSum([pauli"X", pauli"Y"], [2.0, -3.5im]))
    hZero = PauliSum([pauli"XI", pauli"ZZ"], [0.0, 1.0im], false)
    @test  checkAdjoint(hZero, PauliSum([pauli"ZZ"], [-1im]))
    @test  checkAdjoint(hZero, toAdjoint(hZero, false)) && checkAdjoint(hZero, hZero')
    @test  checkAdjoint(PauliSum(Int), PauliSum([pauli"XX"], [0.0], false))

    #> Mergeable terms are merged for an exact core data type and never for a 
    #> floating-point one, in which case only identical objects can still pass (through 
    #> `isHermitian`)
    for T in (Rational{Int}, Float64)
        isExact = !(T <: AbstractFloat)
        hT = PauliSum([pauli"X", pauli"Y"], Complex{T}[1//2, Complex(0, 1//2)])
        #>> `hSplit` holds the operator of `hT`, storing `X` twice
        hSplit = PauliSum([pauli"X", pauli"Y", pauli"Z"], 
                          Complex{T}[1//4, Complex(0, 1//2), 1//4])
        stamp!(indexTerm(hSplit, 1, false).first, 1, symX) #> The first stored term is `Z`
        @test !isCanonical(hSplit) && PauliSum(hSplit) == hT
        @test checkAdjoint(hSplit, hT') == checkAdjoint(hT', hSplit) == isExact
        @test checkAdjoint(PauliSum(hSplit), hT')
        #>> `hTwice` is Hermitian whether or not its terms are merged
        hTwice = PauliSum([pauli"X", pauli"Z"], Complex{T}[1//4, 1//2])
        stamp!(indexTerm(hTwice, 1, false).first, 1, symX) #> `X` is stored twice
        @test isHermitian(hTwice) && checkAdjoint(hTwice, hTwice)
        @test checkAdjoint(hTwice, deepcopy(hTwice)) == isExact
    end

    #> No conjugate is constructed, so the judgment is also made where the conjugate of a 
    #> coefficient is not representable by its core data type
    hMin = PauliSum([pauli"X"], [Complex(Int8(0), typemin(Int8))])
    hPos = PauliSum([pauli"X"], [Complex(Int16(0), Int16(128))])
    @test_throws OverflowError toAdjoint(hMin)
    @test  checkAdjoint(hMin, hPos) && checkAdjoint(hPos, hMin)
    @test !checkAdjoint(hMin, hMin) && !checkAdjoint(hMin, deepcopy(hMin))

    #> Operands of different core data types are compared at the promoted one, where a 
    #> canonical-form copy follows the overflow checking policy of `PauliSum`
    hOver = PauliSum([pauli"X", pauli"Z"], Int8[100, 100])
    stamp!(indexTerm(hOver, 1, false).first, 1, symX) #> `X` is stored twice
    hWide = PauliSum([pauli"X"], Int16[200])
    @test checkAdjoint(hOver, hWide) && checkAdjoint(hWide, hOver) #> `200 X` as `Int16`
    hNarrow = PauliSum([pauli"X"], Int8[100])
    @test_throws OverflowError checkAdjoint(hOver, hNarrow)
    @test_throws OverflowError checkAdjoint(hNarrow, hOver)
    @test checkAdjoint(hOver, hOver) #> Judged by `isHermitian`, which needs no copy here
    #>> At a floating-point promoted type, no terms are merged: an operand with mergeable 
    #>> terms settles the result for either order of the operands, whichever core data type 
    #>> it has
    hQuarters = PauliSum([pauli"X", pauli"Z"], [0.25, 0.25])
    stamp!(indexTerm(hQuarters, 1, false).first, 1, symX) #> `X` is stored twice
    @test !checkAdjoint(hQuarters, hOver) && !checkAdjoint(hOver, hQuarters)
    hFlat = PauliSum([pauli"X"], [200.0])
    @test !checkAdjoint(hOver, hFlat) && !checkAdjoint(hFlat, hOver)
    @test  checkAdjoint(hWide, hFlat) &&  checkAdjoint(hFlat, hWide)
    #>> A coefficient that the promotion rounds does not pass for the rounded value
    pow53 = Int64(2)^53 #> `pow53 + 1` is rounded to `2.0^53` as a `Float64`
    hOdd = PauliSum([pauli"X"], [pow53 + 1])
    @test Float64(pow53 + 1) === 2.0^53
    @test !checkAdjoint(hOdd, PauliSum([pauli"X"], [2.0^53]))
    @test !checkAdjoint(PauliSum([pauli"X"], [2.0^53]), hOdd)

    #> Infinite coefficients are compared by value as well: no product is evaluated, so no 
    #> argument mirrors `onlyFinite` of `checkCommute`
    hInf = PauliSum([pauli"X", pauli"Y"], [Complex(Inf, 0.0), Complex(0.0, Inf)])
    @test checkAdjoint(hInf, hInf') && !checkAdjoint(hInf, hInf)
end


@testset "toAdjoint(::PauliStr)" begin
    #> Phase conjugation table: conj(im^k) == im^((4-k) mod 4)
    for (phase, phaseConj) in ((posRea, posRea), (posImg, negImg), 
                               (negRea, negRea), (negImg, posImg))
        s = PauliStr(2, symX, phase)
        @test (s').phase == phaseConj
        @test s' == PauliStr(2, symX, phaseConj)
        @test evalPhase((s').phase) == conj(evalPhase(s.phase))
    end

    #> Real-phase strings are Hermitian (self-adjoint)
    @test (pauli"YZ")' == pauli"YZ"
    @test mul(pauli"XY", negRea)' == mul(pauli"XY", negRea)

    #> Zero-site phased multiplicative unit
    @test PauliStr(0, symI, posImg)' == PauliStr(0, symI, negImg)

    #> Mechanism pin: zero-length `Memory` is a runtime-interned per-type singleton, so 
    #> buffer-identity (`===`/`!==`) assertions are only meaningful for nonempty buffers
    @test Memory{UInt}(undef, 0) === Memory{UInt}(undef, 0)
    @test PauliStr(0, symI, posImg)'.x === PauliStr(0).x

    #> Content preservation, involution, anti-distribution over `mul`, and no aliasing
    rng = Xoshiro(42)
    bl = true
    for _ in 1:128
        n1, n2 = rand(rng, 0:3), rand(rng, 0:3) #> Mixed site counts exercise padding
        s1, s2 = randStr(rng, n1), randStr(rng, n2)
        bl &= (countSites(s1') == countSites(s1))
        bl &= (countWeight(s1') == countWeight(s1))
        bl &= (s1'.x == s1.x && s1'.z == s1.z)
        #> Fresh buffers, no aliasing — except zero-length buffers, which are exempt
        bl &= (isempty(s1.x) || (s1'.x !== s1.x && s1'.z !== s1.z))
        bl &= ((s1')' == s1)
        bl &= (mul(s1, s2)' == mul(s2', s1'))
        bl &= (strToMat(s1', max(n1, n2)) ≈ adjoint(strToMat(s1, max(n1, n2))))
    end
    @test bl
end

@testset "toAdjoint(::PauliSum)" begin
    #> Empty (zero-operator) sums and coefficient-type preservation
    @test toAdjoint(PauliSum(Int)) == PauliSum(Int)
    @test toAdjoint(PauliSum(Float64)) isa PauliSum{Float64}
    @test toAdjoint(PauliSum([pauli"X"], Complex{Rational{Int}}(1//2, 1//3))) isa 
          PauliSum{Rational{Int}}

    #> Exact coefficient conjugation with unchanged (canonical, phase-free) strings
    h = PauliSum([pauli"XI", pauli"ZZ"], [Complex{Rational{Int}}(1//2, 1//3), 
                                          Complex{Rational{Int}}(-2//1, 5//7)])
    @test (h').str == h.str
    @test (h').coeff == conj.(h.coeff)
    @test (h').str !== h.str && (h').coeff !== h.coeff #> Fresh Memory, no aliasing
    @test (h').str[begin] !== h.str[begin]             #> Strings rebuilt, not shared

    #> Nontrivial input phases: absorption and conjugation must compose consistently
    @test toAdjoint(PauliSum([PauliStr(1, symY, posImg)], 1)) == 
          PauliSum([PauliStr(1, symY, negImg)], 1)

    #> Conjugating exact coefficients is checked against overflow
    @test_throws OverflowError toAdjoint(PauliSum([pauli"X"], [Complex(0, typemin(Int))]))
    @test_throws OverflowError toAdjoint(PauliSum([pauli"X"], 
                                                  [Complex(Int8(0), typemin(Int8))]))

    #> `simplification=false` retains zero-coefficient terms, so the adjoint is an 
    #> involution on every canonical sum, not only on the simplified ones
    hZero = PauliSum([pauli"XI", pauli"ZZ"], [0.0, 1.0im], false)
    @test toAdjoint(hZero) == PauliSum([pauli"ZZ"], [-1.0im])
    @test toAdjoint(hZero, false) == PauliSum([pauli"XI", pauli"ZZ"], [0.0, -1.0im], false)
    @test toAdjoint(toAdjoint(hZero, false), false) == hZero != (hZero')'

    #> Hermitian (real-coefficient) and anti-Hermitian (imaginary-coefficient) sums
    hReal = PauliSum([pauli"XI", pauli"ZZ"], Complex{Int}[2, -3])
    hImag = PauliSum([pauli"XI", pauli"ZZ"], Complex{Int}[2im, -3im])
    @test hReal' == hReal
    @test hImag' == mul(hImag, -1)

    #> Involution, matrix ground truth, and anti-distribution over `mul`
    rng = Xoshiro(43)
    bl = true
    for _ in 1:64
        n = rand(rng, 1:3)
        hA = randSum(rng, n, rand(rng, 0:4))
        hB = randSum(rng, n, rand(rng, 0:4))
        bl &= (toAdjoint(toAdjoint(hA)) == hA)
        bl &= (sumToMat(toAdjoint(hA), n) ≈ adjoint(sumToMat(hA, n)))
        bl &= (mul(hA, hB)' == mul(hB', hA'))
    end
    @test bl
end

end
