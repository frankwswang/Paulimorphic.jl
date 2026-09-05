using Test
using Paulimorphic
using Paulimorphic: formatMolecularInteData

@testset "Molecular.jl" begin

#> Shared fixtures
v = [1.0 0.25; 0.25 0.5]
g1 = [v[i, j] * v[m, n] for i in 1:2, j in 1:2, m in 1:2, n in 1:2] #> 8-fold symmetric
g2 = zeros(2, 2, 2, 2) #> Hermitian and pair transposable, but not exchange symmetric
g2[1, 2, 1, 1] = 1.0
g2[2, 1, 1, 1] = 1.0

enc2 = toDiracEnc(genJordanWignerEnc(2))
enc4 = toDiracEnc(genJordanWignerEnc(4))
enc8 = toDiracEnc(genJordanWignerEnc(8))
secEnc = formatSpinSectoredEnc(enc4, (2, 2))
secEnc8 = formatSpinSectoredEnc(enc8, (4, 4))

@testset "formatSpinSectoredEnc" begin
    @test length(first(secEnc).first) == 2
    #> The pair method now trims each sector to the requested mode count; currently it 
    #> raises UndefVarError, since the do-block references `annOps`/`creOps` from the 
    #> single-encoding method instead of `sec.first`/`sec.second`
    trimmed = formatSpinSectoredEnc(secEnc8, (2, 3))
    @test map(sec->length(sec.first), trimmed) == (2, 3)
    @test map(sec->length(sec.second), trimmed) == (2, 3)
    @test_throws DomainError formatSpinSectoredEnc(enc4, (3, -1))
    @test_throws ArgumentError formatSpinSectoredEnc(enc4, (3, 2))
end

@testset "genNBodyOperatorSum" begin
    @test genNBodyOperatorSum(NormalOrder(), secEnc, zeros(2, 2, 2, 2), (false, false); 
                              checkInput=false) == PauliSum(Float64)
    @test genNBodyOperatorSum(NormalOrder(), secEnc, zeros(2, 2, 2, 2), 
                              (false, false)) == PauliSum(Float64)

    #> Window disjointness under `particleExch=true` (now enforced unconditionally)
    @test_throws ArgumentError genNBodyOperatorSum(NormalOrder(), secEnc8, 
        zeros(2, 2, 2, 2), (false, false), (1, 2); checkInput=false)
    @test genNBodyOperatorSum(NormalOrder(), secEnc8, zeros(2, 2, 2, 2), (false, false), 
                              (1, 3)) == PauliSum(Float64)

    #> Literal-weight convention (`particleExch=false`)
    gN = genNBodyOperatorSum(NormalOrder(), secEnc, g1, (false, false); checkInput=false)
    @test genNBodyOperatorSum(NormalOrder(), secEnc, g1, (false, false); 
                              checkInput=false, particleExch=false) == gN + gN
    gLit = zeros(2, 2, 2, 2)
    gLit[1, 1, 1, 2] = 1.0
    gLit[2, 2, 1, 1] = 1.0
    litSum = secEnc[1].second[1] * secEnc[1].first[1] * 
             secEnc[1].second[1] * secEnc[1].first[2] + 
             secEnc[1].second[2] * secEnc[1].first[2] * 
             secEnc[1].second[1] * secEnc[1].first[1]
    @test genNBodyOperatorSum(PairedOrder(), secEnc, gLit, (false, false); 
                              checkInput=false, particleExch=false) == litSum
    @test genNBodyOperatorSum(PairedOrder(), secEnc, g2, (false, false); 
                              particleExch=false) isa PauliSum
end

@testset "gen1BodyOperatorSum" begin
    @test gen1BodyOperatorSum(first(secEnc), fill(2.0, 1, 1)) == 
          PauliSum([pauli"IIII", pauli"ZIII"], [1.0, -1.0])
    @test gen1BodyOperatorSum(first(secEnc), fill(2.0, 1, 1); checkInput=false) == 
          PauliSum([pauli"IIII", pauli"ZIII"], [1.0, -1.0])
    #> `iModeStart` is now the optional positional argument
    @test gen1BodyOperatorSum(first(secEnc), fill(2.0, 1, 1), 2; checkInput=false) == 
          PauliSum([pauli"IIII", pauli"IZII"], [1.0, -1.0])
end

@testset "formatMolecularInteData" begin
    h2 = [-1.0 0.5; 0.5 -0.5]
    @test formatMolecularInteData(NormalOrder(), (h2, g1)) == (h2, g1)
    @test formatMolecularInteData(PairedOrder(), (h2, g1)) == 
          ([-1.53125 0.3125; 0.3125 -0.65625], g1)
    @test formatMolecularInteData(PairedOrder(), (fill(1, 1, 1), fill(1, 1, 1, 1, 1))) == 
          (fill(0.5, 1, 1), fill(1, 1, 1, 1, 1))
end

@testset "encodeElecHam (golden acceptance)" begin
    #> Single-encoding path currently raises DomainError: the `h1Spin1` call still 
    #> passes the removed `isSpin2Sec` Boolean, which now lands in `iModeStart` 
    #> (`false` -> mode start `0`); the `h1Spin2` call survives only because 
    #> `true == 1`. The stale third positional should be deleted from both calls.
    @test encodeElecHam(NormalOrder(), enc2, (fill(1.0, 1, 1), zeros(1, 1, 1, 1))) == 
          PauliSum([pauli"II", pauli"ZI", pauli"IZ"], [1.0, -0.5, -0.5])

    h2 = [-1.0 0.5; 0.5 -0.5]
    golden = PauliSum(
        [pauli"IIII", pauli"ZIII", pauli"IZII", pauli"IIZI", pauli"IIIZ", pauli"XXII", 
         pauli"YYII", pauli"ZZII", pauli"ZIZI", pauli"ZIIZ", pauli"IZZI", pauli"IZIZ", 
         pauli"IIXX", pauli"IIYY", pauli"IIZZ", pauli"XXZI", pauli"XXIZ", pauli"YYZI", 
         pauli"YYIZ", pauli"ZIXX", pauli"ZIYY", pauli"IZXX", pauli"IZYY", pauli"XXXX", 
         pauli"XXYY", pauli"YYXX", pauli"YYYY"], 
        [-0.71875, 0.015625, -0.046875, 0.015625, -0.046875, 0.34375, 0.34375, 0.109375, 
         0.25, 0.125, 0.125, 0.0625, 0.34375, 0.34375, 0.109375, -0.0625, -0.03125, 
         -0.0625, -0.03125, -0.0625, -0.0625, -0.03125, -0.03125, 0.015625, 0.015625, 
         0.015625, 0.015625])
    @test encodeElecHam(NormalOrder(), enc4, (h2, g1)) == golden
    @test encodeElecHam(PairedOrder(), enc4, (h2, g1)) == golden

    #> Pair-encoding path currently raises UndefVarError inside the pair method of 
    #> `formatSpinSectoredEnc` (see above)
    @test encodeElecHam(NormalOrder(), secEnc, (h2, g1)) == golden

    #> Note (no runtime test): the same-spin two-body calls pass the per-spin pair 
    #> `idxPairSymm` as per-particle flags; they should pass `(sec1Symm, sec1Symm)` and 
    #> `(sec2Symm, sec2Symm)` respectively. Inert while `checkInput=false`, but mixed 
    #> flags would trip the checker's constancy rule if validation is ever enabled.
end

end
