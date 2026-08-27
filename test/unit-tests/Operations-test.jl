using Test
using Random: Xoshiro
using LinearAlgebra: kron
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg

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

function sumToMat(ham::PauliSum, nSite::Int=countSites(ham))
    res = zeros(ComplexF64, 2^nSite, 2^nSite)
    for i in eachindex(ham.str)
        res .+= ham.coeff[begin+i-1] .* strToMat(ham.str[begin+i-1], nSite)
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

    #>> 10. simplification=false retains duplicates (still canonically sorted), per method
    resDup = add(pauli"X", pauli"X", false)
    @test resDup.coeff == Complex{Int}[1, 1]
    @test PauliSum(resDup, true) == PauliSum([pauli"X"], [2])
    @test length(add(h1, h1, false).str) == 2
    @test length(add(h1, pauli"XZ", false).str) == 2
    @test length(add(h1, pauli"XZ"=>1, false).str) == 2

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

    #>> Sum ↔ Str and Sum ↔ Pair in both orders; Sum + Sum routes through `add`
    h = PauliSum([pauli"Z"], [1.0])
    @test pauli"X" + h == PauliSum([pauli"X", pauli"Z"], [1.0, 1.0])
    @test h + pauli"X" == pauli"X" + h
    @test h + (pauli"X"=>2.0im) == (pauli"X"=>2.0im) + h
    @test h1 + h2 == add(h1, h2)

    #>> Mixed chain falls back to Base's pairwise fold with correct promotion
    @test (PauliSum(Float64) + pauli"X" + pauli"Y") isa PauliSum{Float64}

    #>> Intentionally unsupported combinations stay unsupported
    @test_throws MethodError +(pauli"X")            #>> No unary `+`
    @test_throws MethodError pauli"X" + (pauli"Y"=>2.0) #>> Pair terms attach only to a sum
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

    #> mul(::PauliStr, ::Union{Real, Complex}) and the reversed order
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

    #> `simplification=false` preserves the raw cross-term count; simplifying it afterwards
    #> recovers the merged result
    hRaw = mul(h1, h2, false)
    @test length(hRaw.coeff) == 4
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

    #> mul(::PauliSum, ::Union{Real, Complex}) and the reversed order
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

@testset "evalCommute/evalAntiCom" begin
    @test evalCommute(pauli"X", pauli"Z") == PauliSum([pauli"Y"], -2im)
    @test evalCommute(pauli"X", pauli"Z") isa PauliSum{Int}
    @test evalCommute(pauli"X", pauli"X") == PauliSum(Int)           #> zero commutator
    @test evalAntiCom(pauli"X", pauli"X") == PauliSum([pauli"I"], 2) #> {X,X} = 2I, merged
    @test evalAntiCom(pauli"X", pauli"Z") == PauliSum(Int)           #> zero anticommutator

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

    #> Zero-site phased unit is central: commutes with everything, {u, h} == 2h
    u = PauliSum(Int, [PauliStr()])
    @test isempty(evalCommute(u, h1).str)
    @test evalAntiCom(u, h1) == mul(h1, 2)

    #> Zero-operand behavior and coefficient-type promotion
    @test evalAntiCom(h1, PauliSum(Int)) == PauliSum(Int)
    @test evalCommute(h1, PauliSum(Int)) == PauliSum(Int)
    hRat = PauliSum([pauli"XI"], Complex{Rational{Int}}(1//2, 0))
    @test evalAntiCom(hRat, mul(h2, 1.0)) isa PauliSum{Float64}

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

        hA = randSum(rng, n, rand(rng, 0:4))
        hB = randSum(rng, n, rand(rng, 0:4))
        hC = randSum(rng, n, rand(rng, 0:4))
        cmAB = evalCommute(hA, hB)
        acAB = evalAntiCom(hA, hB)
        bl &= (acAB == evalAntiCom(hB, hA))
        bl &= (cmAB == mul(evalCommute(hB, hA), -1))
        bl &= (isempty(evalCommute(hA, hA).str))
        bl &= (evalAntiCom(hA, hA) == mul(mul(hA, hA), 2))
        bl &= (toAdjoint(acAB) == evalAntiCom(hA', hB'))
        bl &= (toAdjoint(cmAB) == mul(evalCommute(hA', hB'), -1))
        bl &= (PauliSum(vcat(acAB.str, cmAB.str), vcat(acAB.coeff, cmAB.coeff)) == 
               mul(mul(hA, hB), 2))
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
