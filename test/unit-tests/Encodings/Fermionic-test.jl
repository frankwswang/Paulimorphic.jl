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

end