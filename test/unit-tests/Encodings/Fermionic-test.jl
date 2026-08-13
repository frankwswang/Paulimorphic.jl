using Test
using Paulimorphic

@testset "Fermionic.jl" begin

@testset "checkMajoranaEnc" begin
    #> Regression guard: `nMode >= 3` exposes same-sector pair-enumeration errors 
    #>> (self-pairs must never be tested for anticommutation)
    for n in 1:4
        @test checkMajoranaEnc(genJordanWignerEnc(n))
    end

    #> Hardcoded 3-mode Jordan-Wigner encoding (generator-independent), plus view input
    oddJW3 = [pauli"XII", pauli"ZXI", pauli"ZZX"]
    evnJW3 = [pauli"YII", pauli"ZYI", pauli"ZZY"]
    @test checkMajoranaEnc(oddJW3 => evnJW3)
    @test checkMajoranaEnc(view(oddJW3, :) => view(evnJW3, :))

    #> A valid non-Jordan-Wigner pairing, and acceptance of the (Hermitian) `negRea` phase
    @test checkMajoranaEnc([pauli"XI", pauli"ZX"] => [pauli"YI", pauli"ZZ"])
    @test checkMajoranaEnc([mul(pauli"X", Paulimorphic.negRea)] => [pauli"Y"])

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
    badPhase = [pauli"X"] => [mul(pauli"Y", Paulimorphic.posImg)]
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

end


# #> ===== Section 2: additions to test/unit-tests/Encodings/Fermionic-test.jl ===== <#

# @testset "checkDiracEnc" begin
#     #> Happy path: every fermionic-encoding generator, both default (exact `Rational`) 
#     #> and `Float64` coefficient types (all stored amplitudes are dyadic, hence exact)
#     for gen in (genJordanWignerEnc, genParityEnc, genBravyiKitaevEnc), n in 1:3
#         mEnc = gen(n)
#         @test checkDiracEnc(toDiracEnc(mEnc))
#         @test checkDiracEnc(toDiracEnc(Float64, mEnc))
#     end

#     ann3, cre3 = toDiracEnc(genBravyiKitaevEnc(3))

#     #> `adjoint` ties the two sectors of a valid encoding together
#     @test all(adjoint(ann3[begin+p-1]) == cre3[begin+p-1] for p in 1:3)

#     #> Role swap passes: the relations are invariant under `a <-> a'`, so the 
#     #> annihilation-versus-creation assignment is a positional convention
#     @test checkDiracEnc(cre3 => ann3)

#     #> View (SubArray) input; a mode subset of a valid encoding remains valid
#     @test checkDiracEnc(view(ann3, 1:2) => view(cre3, 1:2))

#     #> Sector-count violations
#     @test !checkDiracEnc(PauliSum{Rational{Int}}[] => PauliSum{Rational{Int}}[])
#     @test !checkDiracEnc(ann3 => cre3[1:2])
#     @test_throws "positive integer" checkDiracEnc(
#         PauliSum{Rational{Int}}[] => PauliSum{Rational{Int}}[], true)
#     @test_throws "must equal" checkDiracEnc(ann3 => cre3[1:2], true)

#     #> Site-count uniformity violation
#     ann2, cre2 = toDiracEnc(genJordanWignerEnc(2))
#     @test !checkDiracEnc([ann3[1], ann2[1]] => [cre3[1], cre2[1]])
#     @test_throws "same number of sites" checkDiracEnc(
#         [ann3[1], ann2[1]] => [cre3[1], cre2[1]], true)

#     a1 = toDiracEnc(genJordanWignerEnc(1)).first[1] #> a == (X + im Y) / 2

#     #> Adjoint-condition violation that every anticommutator condition misses: 
#     #>> `a == 1` member of the family cre == [[a, -a^2], [1, -a]] == Z - im Y, which 
#     #>> satisfies {a_1, cre} == I and cre^2 == 0 yet is not the adjoint of `a_1`
#     creFake = PauliSum([pauli"Z", pauli"Y"], 
#                        [Complex{Rational{Int}}(1, 0), Complex{Rational{Int}}(0, -1)])
#     @test isempty(evalAntiCom(a1, a1).str)                        #> C4 holds
#     @test evalAntiCom(a1, creFake) == PauliSum(Int, [PauliStr(1)]) #> C5 holds
#     @test !checkDiracEnc([a1] => [creFake])                       #> Only C3 catches it
#     @test_throws "adjoint" checkDiracEnc([a1] => [creFake], true)

#     #> Nilpotency (self-anticommutation) violation with the adjoint condition intact: 
#     #> a Hermitian operator as `ann`
#     hermOp = PauliSum([pauli"X"], Complex{Rational{Int}}(1//2, 0))
#     @test hermOp' == hermOp
#     @test !checkDiracEnc([hermOp] => [hermOp])
#     @test_throws "annihilation operators" checkDiracEnc([hermOp] => [hermOp], true)

#     #> Cross-mode anticommutation violation: modes that are mutual adjoints
#     @test !checkDiracEnc([a1, adjoint(a1)] => [adjoint(a1), a1])
#     @test_throws "annihilation operators" checkDiracEnc(
#         [a1, adjoint(a1)] => [adjoint(a1), a1], true)

#     #> Mode-independence violation: a duplicated mode passes the adjoint and 
#     #> annihilation-sector conditions and is caught only by the cross-mode {a_p, (a_q)'}
#     dupAnn = [a1, a1]
#     dupCre = [adjoint(a1), adjoint(a1)]
#     @test isempty(evalAntiCom(a1, a1).str)
#     @test !checkDiracEnc(dupAnn => dupCre)
#     @test_throws "anticommute to zero" checkDiracEnc(dupAnn => dupCre, true)

#     #> Normalization violation: adjoint pairing intact, {2a, (2a)'} == 4I != I
#     badAnn = mul(a1, 2)
#     @test !checkDiracEnc([badAnn] => [adjoint(badAnn)])
#     @test_throws "identity operator" checkDiracEnc([badAnn] => [adjoint(badAnn)], true)

#     #> Zero-operator encoding: rejected by the normalization condition, not structurally
#     @test !checkDiracEnc([PauliSum(Rational{Int})] => [PauliSum(Rational{Int})])
#     @test_throws "identity operator" checkDiracEnc(
#         [PauliSum(Rational{Int})] => [PauliSum(Rational{Int})], true)

#     #> Mixed coefficient types across modes: comparisons and products promote exactly
#     annMix = PauliSum[ann2[1], mul(ann2[2], 1.0)]
#     creMix = PauliSum[cre2[1], adjoint(mul(ann2[2], 1.0))]
#     @test checkDiracEnc(annMix => creMix)

#     #> Conservative rejection of a broken canonical-form invariant: an in-place phase 
#     #> mutation (only reachable through non-exported access) must not silently validate
#     cpy = PauliSum(collect(a1.str), collect(a1.coeff))
#     cpy.str[begin].phase = Paulimorphic.negImg
#     @test !checkDiracEnc([cpy] => [adjoint(a1)])
# end