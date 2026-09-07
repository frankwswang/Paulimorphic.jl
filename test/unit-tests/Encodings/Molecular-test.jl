using Test
using Paulimorphic
using Paulimorphic: formatMolecularInteData, genNBodyOperator

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
    trimmed = formatSpinSectoredEnc(secEnc8, (2, 3))
    @test map(sec->length(sec.first), trimmed) == (2, 3)
    @test map(sec->length(sec.second), trimmed) == (2, 3)
    @test_throws DomainError formatSpinSectoredEnc(enc4, (3, -1))
    @test_throws ArgumentError formatSpinSectoredEnc(enc4, (3, 2))
end

@testset "genNBodyOperator" begin
    @test genNBodyOperator(PairedOrder(), secEnc, (false,), ((1, 2),)) == 
          mul(first(secEnc).second[1], first(secEnc).first[2])
    @test genNBodyOperator(NormalOrder(), secEnc, (false,), ((2, 1),)) == 
          mul(first(secEnc).second[2], first(secEnc).first[1])
end

@testset "genNBodyOperatorSum" begin
    @test genNBodyOperatorSum(NormalOrder(), secEnc, zeros(2, 2, 2, 2), (false, false); 
                              checkInput=false) == PauliSum(Float64)
    @test genNBodyOperatorSum(NormalOrder(), secEnc, zeros(2, 2, 2, 2), 
                              (false, false)) == PauliSum(Float64)

    @test_throws ArgumentError begin
        genNBodyOperatorSum(NormalOrder(), secEnc8, zeros(2, 2, 2, 2), (false, false), 
                            (1, 2); checkInput=false)
    end

    #> Odd particle count exercises the center branch of `genNBodyOperatorCore!`
    s1, s2 = secEnc8
    litOp = s1.second[1] * s1.second[4] * s2.second[1] * s2.first[2] * s1.first[3] * 
            s1.first[2]
    t6 = zeros(2, 2, 2, 2, 2, 2)
    t6[1, 2, 2, 1, 1, 2] = 1.0
    @test genNBodyOperatorSum(NormalOrder(), secEnc8, t6, (false, false, true), 
                              (1, 3, 1); hermiticity=false) == PauliSum(Float64, litOp)

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

    t1 = ones(2, 2, 3, 3) #> particle 1 window 1:2, particle 2 window 1:3
    @test_throws ArgumentError genNBodyOperatorSum(NormalOrder(), secEnc8, t1, 
        (false, false), (1, 1); checkInput=false) #> Overlapped windows disallowed
    @test genNBodyOperatorSum(NormalOrder(), secEnc8,  ones(2, 2, 2, 2), 
        (false, false), (1, 1); checkInput=false) isa PauliSum #> Identical windows allowed
    @test genNBodyOperatorSum(NormalOrder(), secEnc8, zeros(2, 2, 2, 2), 
        (false, false), (1, 3)) == PauliSum(Float64) #> Disjoint windows allowed
    @test genNBodyOperatorSum(NormalOrder(), secEnc8, zeros(0, 0, 2, 2), 
        (false, false), (4, 3)) == PauliSum(Float64) #> Zero axis extent allowed
    @test genNBodyOperatorSum(NormalOrder(), secEnc8, zeros(2, 2, 2, 2), 
        (false, false), (Int8(1), Int8(3))) == PauliSum(Float64) #> Non-`Int` index allowed
    @test_throws ArgumentError genNBodyOperatorSum(NormalOrder(), secEnc8, 
        zeros(0, 0, 2, 2), (false, false), (5, 3)) #> Out-of-bound mode index rejected
    @test_throws ArgumentError genNBodyOperatorSum(NormalOrder(), secEnc8, 
        ones(2, 2, 2, 2), (false, false), (UInt(1), UInt(6))) #> Out-of-bound `UInt` index
end

@testset "gen1BodyOperatorSum" begin
    @test gen1BodyOperatorSum(first(secEnc), fill(2.0, 1, 1)) == 
          PauliSum([pauli"IIII", pauli"ZIII"], [1.0, -1.0])
    @test gen1BodyOperatorSum(first(secEnc), fill(2.0, 1, 1); checkInput=false) == 
          PauliSum([pauli"IIII", pauli"ZIII"], [1.0, -1.0])
    @test gen1BodyOperatorSum(first(secEnc), fill(2.0, 1, 1), 2; checkInput=false) == 
          PauliSum([pauli"IIII", pauli"IZII"], [1.0, -1.0])

    #> Coefficient summation accuracy check
    oneSec = toDiracEnc(genJordanWignerEnc(3))
    h = [2e16 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 -2e16]
    res = gen1BodyOperatorSum(oneSec, h)
    @test countTerms(res) == 4
    @test res == PauliSum([pauli"III", pauli"ZII", pauli"IZI", pauli"IIZ"], 
                        [1.0, -1e16, -1.0, 1e16])
end

@testset "formatMolecularInteData" begin
    h2 = [-1.0 0.5; 0.5 -0.5]
    @test formatMolecularInteData(NormalOrder(), (h2, g1)) == (h2, g1)
    @test formatMolecularInteData(PairedOrder(), (h2, g1)) == 
          ([-1.53125 0.3125; 0.3125 -0.65625], g1)
    @test formatMolecularInteData(PairedOrder(), (fill(1, 1, 1), fill(1, 1, 1, 1, 1))) == 
          (fill(0.5, 1, 1), fill(1, 1, 1, 1, 1))

    #> Coefficient summation accuracy check
    g = zeros(3, 3, 3, 3)
    g[1, 1, 1, 1] = 1e16
    for idx in ((1, 2, 2, 1), (2, 1, 2, 1), (1, 2, 1, 2), (2, 1, 1, 2)); g[idx...] = 1.0 end
    for idx in ((1, 3, 3, 1), (3, 1, 3, 1), (1, 3, 1, 3), (3, 1, 1, 3)); g[idx...] = -1e16 end
    new1B, _ = formatMolecularInteData(PairedOrder(), (zeros(3, 3), g))
    @test new1B[1, 1] == -0.5
    @test new1B[2, 2] == -0.5 #> Control entry: independent of the accumulation order
end

@testset "encodeElecHam" begin
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

    @test encodeElecHam(NormalOrder(), secEnc, (h2, g1)) == golden

    @testset "Cross-tensor validation" begin
        g3 = zeros(2, 2, 2, 2) #> Hermitian and exchange symmetric; not pair transposable
        g3[1, 2, 1, 2] = 1.0
        g3[2, 1, 2, 1] = 1.0
        h0 = zeros(2, 2)

        #> An aliased or value-equal cross tensor must still be validated as a cross tensor
        @test_throws ArgumentError encodeElecHam(NormalOrder(), enc4, (h0, g3), (h0, g1), 
                                                 g3; idxPairSymm=(false, true))
        @test_throws ArgumentError encodeElecHam(NormalOrder(), enc4, (h0, g3), (h0, g1), 
                                                 copy(g3); idxPairSymm=(false, true))
        @test_throws ArgumentError encodeElecHam(NormalOrder(), enc4, (h0, g1), (h0, g3), 
                                                 g3; idxPairSymm=(true, false))

        #> Outputs must be consistent across different call patterns that are equivalent
        @test encodeElecHam(NormalOrder(), enc4, (h2, g1), (h2, g1), copy(g1)) == 
              encodeElecHam(NormalOrder(), enc4, (h2, g1))
        @test encodeElecHam(PairedOrder(), enc4, (h2, g1), (h2, g1), copy(g1)) == 
              encodeElecHam(PairedOrder(), enc4, (h2, g1))
    end
end

end
