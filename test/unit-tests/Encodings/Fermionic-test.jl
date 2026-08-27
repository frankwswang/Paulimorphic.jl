using Test
using Paulimorphic
using Paulimorphic: posImg, negRea, negImg

@testset "Fermionic.jl" begin

@testset "genJordanWignerEnc" begin
    enc1 = genJordanWignerEnc(1)
    @test enc1.first  == [pauli"X"]
    @test enc1.second == [pauli"Y"]

    enc3 = genJordanWignerEnc(3)
    @test enc3.first  == [pauli"XII", pauli"ZXI", pauli"ZZX"]
    @test enc3.second == [pauli"YII", pauli"ZYI", pauli"ZZY"]

    #> Site padding only appends identities
    encPad = genJordanWignerEnc(2, 4)
    @test encPad.first  == [pauli"XIII", pauli"ZXII"]
    @test encPad.second == [pauli"YIII", pauli"ZYII"]

    @test_throws DomainError genJordanWignerEnc(0)
    @test_throws DomainError genJordanWignerEnc(-2)
    @test_throws DomainError genJordanWignerEnc(3, 2)
end

@testset "genParityEnc" begin
    enc3 = genParityEnc(3)
    @test enc3.first  == [pauli"XXX", pauli"ZXX", pauli"IZX"]
    @test enc3.second == [pauli"YXX", pauli"IYX", pauli"IIY"]

    encPad = genParityEnc(2, 4)
    @test encPad.first  == [pauli"XXII", pauli"ZXII"]
    @test encPad.second == [pauli"YXII", pauli"IYII"]

    for n in 1:6
        @test checkMajoranaEnc(genParityEnc(n))
    end

    @test_throws DomainError genParityEnc(0)
    @test_throws DomainError genParityEnc(2, 1)
end

@testset "genBravyiKitaevEnc" begin
    #> `nMode == 4`: the Seeley-Richard-Love table (mode `p` <-> their qubit `p - 1`)
    enc4 = genBravyiKitaevEnc(4)
    @test enc4.first  == [pauli"XXIX", pauli"ZXIX", pauli"IZXX", pauli"IZZX"]
    @test enc4.second == [pauli"YXIX", pauli"IYIX", pauli"IZYX", pauli"IIIY"]

    #> `nMode == 3`: the truncation (forest) convention, pinned to OpenFermion's 
    #>> `bravyi_kitaev`; the Havlicek-style single-rooted tree variant would differ here
    enc3 = genBravyiKitaevEnc(3)
    @test enc3.first  == [pauli"XXI", pauli"ZXI", pauli"IZX"]
    @test enc3.second == [pauli"YXI", pauli"IYI", pauli"IZY"]

    #> Site padding only appends identities (the update sets stay bounded by `nMode`)
    encPad = genBravyiKitaevEnc(3, 5)
    @test encPad.first  == [pauli"XXIII", pauli"ZXIII", pauli"IZXII"]
    @test encPad.second == [pauli"YXIII", pauli"IYIII", pauli"IZYII"]

    #> Validity across sizes, including non-powers of 2 and padded widths
    for n in 1:8
        @test checkMajoranaEnc(genBravyiKitaevEnc(n))
        @test checkMajoranaEnc(genBravyiKitaevEnc(n, n + 2))
    end

    @test_throws DomainError genBravyiKitaevEnc(0)
    @test_throws DomainError genBravyiKitaevEnc(2, 1)
end

@testset "`PauliStr`-based `checkMajoranaEnc`" begin
    #> Hardcoded 3-mode Jordan-Wigner encoding (generator-independent), plus view input
    oddJW3 = [pauli"XII", pauli"ZXI", pauli"ZZX"]
    evnJW3 = [pauli"YII", pauli"ZYI", pauli"ZZY"]
    @test checkMajoranaEnc(oddJW3 => evnJW3)
    @test checkMajoranaEnc(view(oddJW3, :) => view(evnJW3, :))

    #> A valid non-Jordan-Wigner pairing, and acceptance of the (Hermitian) `negRea` phase
    @test checkMajoranaEnc([pauli"XI", pauli"ZX"] => [pauli"YI", pauli"ZZ"])
    @test checkMajoranaEnc([mul(pauli"X", negRea)] => [pauli"Y"])

    #> Sector-count violations: empty sectors and unequal sector lengths
    @test !checkMajoranaEnc(PauliStr[] => PauliStr[])
    @test !checkMajoranaEnc(oddJW3 => evnJW3[1:2])
    @test_throws ArgumentError checkMajoranaEnc(PauliStr[] => PauliStr[], true)
    @test_throws ArgumentError checkMajoranaEnc(oddJW3 => evnJW3[1:2], true)

    #> Site-count uniformity violation (term 2 of the second sector)
    badSite = [pauli"XI", pauli"ZX"] => [pauli"YI", pauli"ZYI"]
    @test !checkMajoranaEnc(badSite)
    @test_throws ArgumentError checkMajoranaEnc(badSite, true)

    #> Hermiticity violation: imaginary phase in the second sector
    badPhase = [pauli"X"] => [mul(pauli"Y", posImg)]
    @test !checkMajoranaEnc(badPhase)
    @test_throws ArgumentError checkMajoranaEnc(badPhase, true)

    #> Same-sector commuting pair (`XI` and `XZ` commute within the first sector)
    badSame = [pauli"XI", pauli"XZ"] => [pauli"YI", pauli"ZY"]
    @test !checkMajoranaEnc(badSame)
    @test_throws ArgumentError checkMajoranaEnc(badSame, true)

    #> Cross-sector commuting pair at equal mode index (γ_1 duplicated as γ_2): the only 
    #>> violated pair sits at `(i, j) == (1, 1)` across the two sectors
    badCross = [pauli"XI", pauli"ZX"] => [pauli"XI", pauli"ZY"]
    @test !checkMajoranaEnc(badCross)
    @test_throws ArgumentError checkMajoranaEnc(badCross, true)
end

@testset "`PauliSum`-based `checkMajoranaEnc`" begin
    oneTemSum(strs) = [PauliSum(Int, [s]) for s in strs]

    #> One-string `PauliSum` encodings are valid, including with `negRea` phases
    for gen in (genJordanWignerEnc, genParityEnc, genBravyiKitaevEnc), n in 1:3
        mEnc = gen(n)
        @test checkMajoranaEnc(oneTemSum(mEnc.first) => oneTemSum(mEnc.second))
    end
    mEnc2 = genJordanWignerEnc(2)
    negged = [PauliSum(Int, [mul(mEnc2.first[1], negRea)]), 
              PauliSum(Int, [mEnc2.first[2]])] => oneTemSum(mEnc2.second)
    @test checkMajoranaEnc(negged)

    #> A genuinely `PauliSum`-based (class-3) example: exact rational conjugated frame 
    #> with commuting support strings (matrix-validated); valid despite 8 > 2n strings
    c, s = 3//5, 4//5
    conjEnc = [PauliSum([pauli"XI", pauli"YZ"], [c,  s]), 
               PauliSum([pauli"ZX", pauli"IY"], [c,  s])] => 
              [PauliSum([pauli"YI", pauli"XZ"], [c, -s]), 
               PauliSum([pauli"ZY", pauli"IX"], [c, -s])]
    @test checkMajoranaEnc(conjEnc)

    #> Disjointness between Dirac encoding and Majorana encoding
    dEnc = toDiracEnc(genJordanWignerEnc(2))
    @test !checkMajoranaEnc(dEnc)                    #> Annihilation ops are non-Hermitian
    @test !checkDiracEnc(conjEnc)                    #> Majorana ops are non-nilpotent
    @test_throws "Hermitian" checkMajoranaEnc(dEnc, true)

    #> Condition-isolated encoding violations
    g1 = PauliSum(Int, [pauli"XI"])
    g2 = PauliSum(Int, [pauli"YI"])
    @test_throws "involution" checkMajoranaEnc(
        [mul(g1, 1//2)] => [g2], true)               #> Hermitian but (γ/2)^2 != I
    @test_throws "Hermitian" checkMajoranaEnc(
        [PauliSum([pauli"XI"], Complex{Int}(0, 1))] => [g2], true)

    #> Complex-orthogonal (not unitary) frame, `M Mᵀ == I` over ℂ with `c^2 - s^2 == 1`: 
    #> passes the involution and every anticommutation condition exactly, and is rejected 
    #> ONLY by the Hermiticity (real-coefficient) check
    gC1 = PauliSum([pauli"X", pauli"Y"], Complex{Rational{Int}}[5//4, 3im//4])
    gC2 = PauliSum([pauli"X", pauli"Y"], Complex{Rational{Int}}[-3im//4, 5//4])
    @test mul(gC1, gC1) == PauliSum(Int, [PauliStr(1)])
    @test mul(gC2, gC2) == PauliSum(Int, [PauliStr(1)])
    @test isempty(evalAntiCom(gC1, gC2).str)
    @test !checkMajoranaEnc([gC1] => [gC2])
    @test_throws "Hermitian" checkMajoranaEnc([gC1] => [gC2], true)
    @test_throws "anticommute" checkMajoranaEnc(
        [g1] => [PauliSum(Int, [pauli"XZ"])], true)  #> XI and XZ commute
    @test_throws "positive integer" checkMajoranaEnc(
        PauliSum{Int}[] => PauliSum{Int}[], true)
    @test_throws "must equal" checkMajoranaEnc([g1, g2] => [g2], true)
    @test_throws "same number of sites" checkMajoranaEnc(
        [g1] => [PauliSum(Int, [pauli"Y"])], true)
end

@testset "toDiracEnc" begin
    enc = genJordanWignerEnc(2)  #> γ1=XI, γ2=YI, γ3=ZX, γ4=ZY
    ann, cre = toDiracEnc(enc)

    #> Default T and shape
    @test ann isa Vector{PauliSum{Rational{Int}}}
    @test length(ann) == length(cre) == 2

    #> Exact mode-1 content
    xI = PauliStr([symX, symI])
    yI = PauliStr([symY, symI])
    @test ann[1] == PauliSum([xI, yI], [Complex{Rational{Int}}(1//2, 0), 
                                        Complex{Rational{Int}}(0,  1//2)])
    @test cre[1] == PauliSum([xI, yI], [Complex{Rational{Int}}(1//2, 0), 
                                        Complex{Rational{Int}}(0, -1//2)])

    #> Checked canonical fermionic commutation relation
    idSum = PauliSum([PauliStr(2)], Complex{Rational{Int}}(1))
    for p in 1:2, q in 1:2
        resPQ = ann[p] * cre[q] + cre[q] * ann[p]
        @test p == q ? (resPQ == idSum) : isempty(resPQ.str)  #> {a_p, a_q'} == δ_pq I
        @test isempty((ann[p]*ann[q] + ann[q]*ann[p]).str)    #> {a_p, a_q}  == 0
    end

    #> No aliasing into `enc` (string objects and bit buffers both fresh)
    @test ann[1].str[1] !== enc.first[1] && ann[1].str[1].x !== enc.first[1].x

    #> Unrepresentable `T` raises ArgumentError
    @test_throws ArgumentError toDiracEnc(Int, enc)

    #> Fail-fast ordering: invalid `T` is reported first even when `enc` is also invalid
    @test_throws "one half" toDiracEnc(Int, PauliStr[] => PauliStr[])

    #> Invalid encoding is rejected up front
    @test_throws ArgumentError toDiracEnc(PauliStr[] => PauliStr[])

    #> Pin current abstract-`T` behavior: respect abstract `T`
    resAbs = toDiracEnc(Rational, genJordanWignerEnc(1))
    @test resAbs.first isa Vector{PauliSum{Rational}}
end

@testset "checkDiracEnc" begin
    #> Check (Majorana) fermionic-encoding generators
    for gen in (genJordanWignerEnc, genParityEnc, genBravyiKitaevEnc), n in 1:3
        mEnc = gen(n)
        @test checkDiracEnc(toDiracEnc(mEnc))
        @test checkDiracEnc(toDiracEnc(Float64, mEnc))
    end

    ann3, cre3 = toDiracEnc(genBravyiKitaevEnc(3))

    #> Role swap passes: the relations are invariant under `a <-> a'`, so the 
    #> annihilation-versus-creation assignment is a positional convention
    @test checkDiracEnc(cre3 => ann3)

    #> View (SubArray) input; a mode subset of a valid encoding remains valid
    @test checkDiracEnc(view(ann3, 1:2) => view(cre3, 1:2))

    #> Sector-count violations
    enc11 = PauliSum{Rational{Int}}[] => PauliSum{Rational{Int}}[]
    enc12 = ann3 => cre3[1:2]
    @test !checkDiracEnc(enc11)
    @test !checkDiracEnc(enc12)
    @test_throws "positive integer" checkDiracEnc(enc11, true)
    @test_throws "must equal" checkDiracEnc(enc12, true)

    #> Site-count uniformity violation
    ann2, cre2 = toDiracEnc(genJordanWignerEnc(2))
    enc21 = [ann3[1], ann2[1]] => [cre3[1], cre2[1]]
    @test !checkDiracEnc(enc21)
    @test_throws "same number of sites" checkDiracEnc(enc21, true)

    a1 = toDiracEnc(genJordanWignerEnc(1)).first[1] #> a == (X + im Y) / 2

    #> Adjoint-condition violation that every anticommutator condition misses: 
    #>> `cre` = Z - im*Y, which satisfies {a_1, cre} == I and cre^2 == 0 yet `cre != (a_1)'`
    creFake = PauliSum([pauli"Z", pauli"Y"], 
                       [Complex{Rational{Int}}(1, 0), Complex{Rational{Int}}(0, -1)])
    enc31 = [a1] => [creFake]
    @test isempty(evalAntiCom(a1, a1).str)
    @test evalAntiCom(a1, creFake) == PauliSum(Int, [PauliStr(1)])
    @test !checkDiracEnc(enc31)
    @test_throws "adjoint" checkDiracEnc(enc31, true)

    #> Self-anticommutation violation with the adjoint condition intact: Hermitian operator
    hermOp = PauliSum([pauli"X"], Complex{Rational{Int}}(1//2, 0))
    enc41 = [hermOp] => [hermOp]
    @test hermOp' == hermOp
    @test !checkDiracEnc(enc41)
    @test_throws "annihilation operators" checkDiracEnc(enc41, true)

    #> Cross-mode anticommutation violation: modes that are mutual adjoints
    enc51 = [a1, adjoint(a1)] => [adjoint(a1), a1]
    @test !checkDiracEnc(enc51)
    @test_throws "annihilation operators" checkDiracEnc(enc51, true)

    #> Mode-independence violation: a duplicated mode passes the adjoint and 
    #> annihilation-sector conditions and is caught only by the cross-mode {a_p, (a_q)'}
    enc61 = [a1, a1] => [adjoint(a1), adjoint(a1)]
    @test isempty(evalAntiCom(a1, a1).str)
    @test !checkDiracEnc(enc61)
    @test_throws "anticommute to zero" checkDiracEnc(enc61, true)

    #> Normalization violation: adjoint pairing intact, {2a, (2a)'} == 4I != I
    badAnn = mul(a1, 2)
    enc71 = [badAnn] => [adjoint(badAnn)]
    @test !checkDiracEnc(enc71)
    @test_throws "identity operator" checkDiracEnc(enc71, true)

    #> Zero-operator encoding: rejected by the normalization condition, not structurally
    enc81 = [PauliSum(Rational{Int})] => [PauliSum(Rational{Int})]
    @test !checkDiracEnc(enc81)
    @test_throws "identity operator" checkDiracEnc(enc81, true)

    #> Mixed coefficient types across modes: comparisons and products promote exactly
    annMix = PauliSum[ann2[1], mul(ann2[2], 1.0)]
    creMix = PauliSum[cre2[1], adjoint(mul(ann2[2], 1.0))]
    @test checkDiracEnc(annMix => creMix)

    #> Conservative rejection of a broken canonical-form invariant: an in-place phase 
    #> mutation (only reachable through non-exported access) must not silently validate
    cpy = PauliSum(collect(a1.str), collect(a1.coeff))
    cpy.str[begin].phase = negImg
    @test !checkDiracEnc([cpy] => [adjoint(a1)])
end

@testset "toMajoranaEnc" begin
    #> `PauliStr` method (default)
    for gen in (genJordanWignerEnc, genParityEnc, genBravyiKitaevEnc), n in 1:3
        mEnc = gen(n)
        dEnc = toDiracEnc(mEnc)
        @test toMajoranaEnc(dEnc) == toMajoranaEnc(PauliStr, dEnc) == mEnc
        @test toDiracEnc(toMajoranaEnc(dEnc)) == dEnc
    end

    #> `PauliSum` method
    ann2, _ = toDiracEnc(genJordanWignerEnc(2))
    b1 = add(mul(ann2[1], 3//5), mul(ann2[2],  4//5))
    b2 = add(mul(ann2[1], 4//5), mul(ann2[2], -3//5))
    rotEnc = [b1, b2] => [toAdjoint(b1), toAdjoint(b2)]
    mSum = toMajoranaEnc(PauliSum, rotEnc)
    T2 = Complex{Rational{Int}}
    @test mSum.first[1]  == PauliSum([pauli"XI", pauli"ZX"], T2[3//5,  4//5])
    @test mSum.first[2]  == PauliSum([pauli"XI", pauli"ZX"], T2[4//5, -3//5])
    @test mSum.second[1] == PauliSum([pauli"YI", pauli"ZY"], T2[3//5,  4//5])
    @test mSum.second[2] == PauliSum([pauli"YI", pauli"ZY"], T2[4//5, -3//5])
    @test checkMajoranaEnc(mSum)
    @test mSum.first[1] isa PauliSum{Rational{Int}} #> Uniform input type preserved

    #> On the image of `toDiracEnc`, the two representations agree term by term
    dEnc3 = toDiracEnc(genBravyiKitaevEnc(3))
    mStr3 = toMajoranaEnc(PauliStr, dEnc3)
    mSum3 = toMajoranaEnc(PauliSum, dEnc3)
    @test map(toPauliStr, mSum3.first)  == mStr3.first
    @test map(toPauliStr, mSum3.second) == mStr3.second

    #> Majorana-string-unconvertible input: `PauliStr` method throws, `PauliSum` succeeds
    @test_throws "single Pauli string" toMajoranaEnc(PauliStr, rotEnc)
    @test checkDiracEnc(rotEnc) && !checkDiracEnc(rotEnc; strRestricted=true)

    #> Invalid input rejected up front by both methods
    a1 = toDiracEnc(genJordanWignerEnc(1)).first[1]
    @test_throws "adjoint" toMajoranaEnc(PauliStr, [a1] => [a1])
    @test_throws "adjoint" toMajoranaEnc(PauliSum, [a1] => [a1])
end

@testset "buildMajoranaFrame" begin
    ann2, _ = toDiracEnc(genJordanWignerEnc(2))
    b1 = add(mul(ann2[1], 3//5), mul(ann2[2],  4//5))
    b2 = add(mul(ann2[1], 4//5), mul(ann2[2], -3//5))
    mSum_can = toMajoranaEnc(PauliSum, [b1, b2] => [toAdjoint(b1), toAdjoint(b2)])
    function flipPhase(pSum::PauliSum)
        res = PauliSum(pSum)
        for (str, i) in zip(res.str, eachindex(res.coeff))
            str.phase = posImg
            res.coeff[i] *= evalPhase(negImg)
        end
        res
    end
    mSum_nca = map(flipPhase, mSum_can.first) => map(flipPhase, mSum_can.second)

    #> Class-1 input: exact frame and matrices, orthogonality, and reconstruction
    for mSum in (mSum_can, mSum_nca)
        mats, frame = buildMajoranaFrame(mSum)
        matA, matC = mats
        @test frame == [pauli"XI", pauli"YI", pauli"ZX", pauli"ZY"] #> Ascending order
        @test matA == Rational{Int}[3//5 4//5; 0 0; 4//5 -3//5; 0 0]
        @test matC == Rational{Int}[0 0; 3//5 4//5; 0 0; 4//5 -3//5]
        matR = hcat(matA, matC)
        @test matR * transpose(matR) == Rational{Int}[i == j for i in 1:4, j in 1:4]
        for i in 1:2
            @test PauliSum(mSum.first[ i]) == PauliSum(frame, matA[:, i])
            @test PauliSum(mSum.second[i]) == PauliSum(frame, matC[:, i])
        end
        @test frame[begin].x !== mSum.first[begin].str[begin].x #> Fresh bit buffers
    end

    #> Class-2 (single-string) input: the recovered `hcat(matA, matC)` is a signed 
    #> permutation, with signs recording the absorbed `negRea` phases
    mEnc2 = genJordanWignerEnc(2)
    wrapped = [PauliSum(Int, [mul(mEnc2.first[1], negRea)]), 
               PauliSum(Int, [mEnc2.first[2]])] => 
              [PauliSum(Int, [s]) for s in mEnc2.second]
    wMats, wFrame = buildMajoranaFrame(wrapped)
    wA, wC = wMats
    @test length(wFrame) == 4 && issorted(wFrame)
    wR = hcat(wA, wC)
    @test all(count(!iszero, wR[i, :]) == 1 for i in 1:4)      #> One entry per row
    @test all(x -> x == 0 || x == 1 || x == -1, wR)
    @test any(==(-1), wA)                                      #> The negRea sign survives

    #> Class-3 inputs: valid encoding but wrong frame basis count
    skew = [PauliSum([pauli"X", pauli"Y", pauli"Z"], 
                     Complex{Rational{Int}}[2//3, 2//3, 1//3])] => 
           [PauliSum([pauli"X", pauli"Y", pauli"Z"], 
                     Complex{Rational{Int}}[2//3, -1//3, -2//3])]
    @test checkMajoranaEnc(skew)
    sMats, sFrame = buildMajoranaFrame(skew)
    sA, sC = sMats
    @test size(sA) == size(sC) == (0, 0) && isempty(sFrame)

    #>> Sub-case: valid class-3 encoding whose 8 support strings include commuting 
    #>> pairs (e.g., XI and XZ) — rejected by the frame basis count check (8 != 2n == 4) 
    #>> before any anticommutativity question arises
    c, s = 3//5, 4//5
    conjEnc = [PauliSum([pauli"XI", pauli"YZ"], [c,  s]), 
               PauliSum([pauli"ZX", pauli"IY"], [c,  s])] => 
              [PauliSum([pauli"YI", pauli"XZ"], [c, -s]), 
               PauliSum([pauli"ZY", pauli"IX"], [c, -s])]
    @test checkMajoranaEnc(conjEnc)
    @test !checkAntiCom(pauli"XI", pauli"XZ")
    cMats, cFrame = buildMajoranaFrame(conjEnc)
    cA, cC = cMats
    @test size(cA) == (0, 0) && isempty(cFrame)

    #> Invalid input propagates the validation error
    @test_throws "Hermitian" buildMajoranaFrame(toDiracEnc(genJordanWignerEnc(2)))

    #> Phase modification on a `PauliStr` (copied from `pSum`) that matches with `strTarget`
    function changePhase(pSum::PauliSum, strTarget::PauliStr, phase::PhaseFactor)
        res = PauliSum(pSum)
        for (str, i) in zip(res.str, eachindex(res.coeff))
            if str == strTarget
                str.phase = phase
                res.coeff[i] *= evalPhase(PhaseFactor((0x4 - UInt8(phase)) & 0x3))
            end
        end
        res
    end

    @testset "Handling encodings with negative-phase `PauliStr`" begin
        ann2, _ = toDiracEnc(genJordanWignerEnc(2))
        b1 = add(mul(ann2[1], 3//5), mul(ann2[2],  4//5))
        b2 = add(mul(ann2[1], 4//5), mul(ann2[2], -3//5))
        mSum = toMajoranaEnc(PauliSum, [b1, b2] => [toAdjoint(b1), toAdjoint(b2)])

        negged = [changePhase(mSum.first[1], pauli"XI", negRea), mSum.first[2]] => 
                collect(mSum.second)
        @test checkMajoranaEnc(negged) #> Sanity: value unchanged, still valid

        mats, frame = buildMajoranaFrame(negged)
        matA, _ = mats
        @test frame == [pauli"XI", pauli"YI", pauli"ZX", pauli"ZY"]
        @test iszero(matA[2, 1])              #> CURRENT: the `YI` row holds a stray `-3//5`
        @test PauliSum(negged.first[1]) == PauliSum(frame, matA[:, 1])
    end

    @testset "Handling encodings with non-canonical (mergeable) `PauliSum`" begin
        #> This encoding's is framed by string pair `(XI, YI)`, but the raw storage (built 
        #> with `simplification=false`) contains a canceling `YZ` pair that pushes the 
        #> distinct-string count to 3.
        op1a = PauliSum([pauli"XI", pauli"YZ", pauli"YZ"], 
                        Complex{Rational{Int}}[1, im, -im], false)
        op2 = PauliSum(Int, [pauli"YI"])
        @test checkMajoranaEnc([op1a] => [op2]) #> Value-level validation passes
        matsA, frameA = buildMajoranaFrame([op1a] => [op2])
        @test frameA == [pauli"XI", pauli"YI"]
        @test PauliSum(op1a) == PauliSum(frameA, first(matsA)[:, 1])

        #> When the phantom pair coincides with a genuine frame string
        op1b = PauliSum([pauli"XI", pauli"YI", pauli"YI"], 
                        Complex{Rational{Int}}[1, im, -im], false)
        @test checkMajoranaEnc([op1b] => [op2])
        matsB, frameB = buildMajoranaFrame([op1b] => [op2])
        @test frameB == [pauli"XI", pauli"YI"]
        @test PauliSum(op1b) == PauliSum(frameB, first(matsB)[:, 1])
    end
end

end