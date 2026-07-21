using Test
using Random: Xoshiro
using LinearAlgebra: kron
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg

@testset "Operations.jl" begin

@test string(pauli"") == "I"
@test string(pauli"I") == "I"
@test string(pauli"II") == "I"
@test string(pauli"IIX") == "+X₃"

@testset "mul" begin
    #> Ground-truth reference: dense matrix representation
    #>> The same `kron` factor order is applied to both sides of every comparison, so the
    #>> checks below are independent of the package's site-ordering convention.
    matI = toMatrix(symI)
    matX = toMatrix(symX) 
    matY = toMatrix(symY) 
    matZ = toMatrix(symZ)

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

    syms   = [symI, symX, symY, symZ]
    phases = [posRea, posImg, negRea, negImg]
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
        bl5 &= (mul(q, s) == r) && (s * q == r) && (q * s == r)
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
    @test mul(2.0, s) == h
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
    @test mul(2.0, hScl) == mul(hScl, 2.0)
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
        bl8 &= (r == mul(p, hScl))
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

#> `evalCommute` and `evalAntiCom`
@test evalCommute(pauli"X", pauli"Z") == PauliSum([pauli"Y"], -2im)  #> [X,Z] = −2iY
@test evalCommute(pauli"X", pauli"Z") isa PauliSum{Int}
@test evalCommute(pauli"X", pauli"X") == PauliSum(Int)               #> zero commutator
@test evalAntiCom(pauli"X", pauli"X") == PauliSum([pauli"I"], 2)     #> {X,X} = 2I, merged
@test evalAntiCom(pauli"X", pauli"Z") == PauliSum(Int)               #> zero anticommutator

end