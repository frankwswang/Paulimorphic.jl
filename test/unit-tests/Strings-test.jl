using Test
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg

@testset "Strings.jl" begin

m = "X"
@test (@pauli_str "$m") == pauli"X" == pauli"X"
@test (@pauli_str [2,3,2,0,1]) == pauli"XYXIZ" #> 0->I, 1->Z, 2->X, 3->Y

@test string(pauli"")   == "+I"
@test string(pauli"I")  == "+I"
@test string(pauli"II") == "+I"
@test toString(pauli"IIX") == "+X₃"
@test toString(pauli"IIX", true) == "IIX"
@test toString(pauli"XXIX", true) == "XXIX" #> Exact macro round-trip
@test toString(pauli"XXIX", true, omitPlusSign=false) == "+XXIX"
@test toString(pauli"X", omitPlusSign=true) == "X₁"
@test toString(PauliStr(2, symX, posImg)) == "+im*X₁X₂"
@test toString(PauliStr(2, symX, posImg), true) == "im*XX"
@test toString(PauliStr(3, symI, negRea)) == "-I"
@test toString(PauliStr(3, symI, negRea), true) == "-III"
@test toString(PauliStr(0, symI, negImg)) ==
      toString(PauliStr(0, symI, negImg), true) == "-im*I"

@testset "`PauliStr` ordering" begin
    #> Build a symbol list of `nSite` sites carrying `sym` at each site index in `locs`
    function genSparseList(nSite::Int, locs::Pair{Int, PauliSym}...)
        list = fill(symI, nSite)
        for (site, sym) in locs
            list[begin+site-1] = sym
        end
        list
    end

    #>≡≡≡ Attribute 1: site count ≡≡≡<#
    #> Pitfall: drafts that graded by weight first compared weights across different ambient 
    #> spaces, e.g. ranking a 2-site identity before a 1-site X.
    @test PauliStr(0) < pauli"I"   #> Zero-site string precedes any explicit one
    @test pauli"X"  < pauli"II"    #> Site count dominates weight (1 vs 0)
    @test pauli"YY" < pauli"III"   #> ... even at maximal weight difference
    @test PauliStr(0, symI, posRea) < PauliStr(0, symI, posImg) #> Zero-site pair: phase key

    #>≡≡≡ Attribute 2: weight grading within a fixed site count ≡≡≡<#
    #> Pitfall: the original word-value comparison anti-correlated with weight, ranking 
    #> X₁⋯X₆₃ before X₆₄.
    @test pauli"IIII" < pauli"IIIZ"
    @test PauliStr(genSparseList(64, 64 => symX)) <
          PauliStr([fill(symX, 63); symI])       #> Weight 1 before weight 63, same word
    @test pauli"YI" < pauli"ZZ"                  #> Weight grades before symbol content
    @test PauliStr(genSparseList(65, 65 => symX)) <
          PauliStr(genSparseList(65, 1 => symZ, 2 => symZ))

    #>≡≡≡ Attribute 3: highest differing site, compared by `PauliSym` value ≡≡≡<#
    #> Pitfall: earlier drafts had non-monotone site significance across words, 
    #> word-granular X-before-Z comparison, and a rank collapse that deferred the X<->Y 
    #> distinction behind every Z-vs-X distinction in the same word.
    @test sort([pauli"Y", pauli"X", pauli"I", pauli"Z"]) ==
               [pauli"I", pauli"Z", pauli"X", pauli"Y"]  #> Matches `PauliSym` value order
    @test issorted([PauliStr([sym]) for sym in (symI, symZ, symX, symY)])
    @test sort([symY, symX, symZ, symI]) == [symI, symZ, symX, symY]

    #> Monotone site significance across the word boundary
    @test PauliStr(genSparseList(65, 64 => symX)) < PauliStr(genSparseList(65, 65 => symX))
    #> Site position beats X-before-Z buffer order
    @test PauliStr(genSparseList(64, 32 => symX)) < PauliStr(genSparseList(64, 33 => symZ))
    #> X<->Y distinction at the higher site beats Z-vs-X at the lower one
    @test pauli"XX" < pauli"ZY"
    #> The later word dominates the earlier word entirely
    @test PauliStr(genSparseList(128, 2 => symX, 127 => symZ)) <
          PauliStr(genSparseList(128, 1 => symX, 128 => symZ))
    @test PauliStr(genSparseList(65, 1 => symY, 65 => symZ)) <
          PauliStr(genSparseList(65, 1 => symZ, 65 => symX))
    #> Mask edge cases: deciding site at the top bit (no shift overflow) and the bottom bit
    @test PauliStr(genSparseList(64, 1 => symZ, 64 => symX)) <
          PauliStr(genSparseList(64, 1 => symZ, 64 => symY))
    @test PauliStr(genSparseList(64, 1 => symZ, 64 => symX)) <
          PauliStr(genSparseList(64, 1 => symX, 64 => symX))
    #> Full weight-1 chain on two sites: site 2 outranks site 1, `I < Z < X < Y` at each
    @test pauli"II" < pauli"ZI" < pauli"XI" < pauli"YI" < pauli"IZ" < pauli"IX" < 
          pauli"IY" < pauli"ZZ"

    #>≡≡≡ Attribute 4: phase as the final tie breaker, consistent with `==` ≡≡≡<#
    phasePool = [posRea, posImg, negRea, negImg]
    #> Phase tie breaker follows the `PhaseFactor` value order: +1 < +im < -1 < -im
    phasedStrs = [PauliStr([symX, symY], p) for p in (posRea, posImg, negRea, negImg)]
    @test issorted(phasedStrs)
    @test allunique(phasedStrs)
    @test !(phasedStrs[begin] < PauliStr([symX, symY])) &&
          !(PauliStr([symX, symY]) < phasedStrs[begin]) #> Equal content and phase: no order
end

@testset "indexSite" begin
    str = pauli"XZIY"
    #> Strict mode (default): unchanged behavior
    @test indexSite(str, 2) === symZ
    @test_throws DomainError indexSite(str, 0)
    @test_throws DomainError indexSite(str, 5)

    #> Tolerant mode: in-range reads identical to strict mode
    @test all(indexSite(str, i, true) === indexSite(str, i) for i in 1:4)

    #> Tolerant mode: beyond the explicit site count reads as implicit identity
    @test indexSite(str, 5, true) === symI
    @test indexSite(str, 64, true) === symI #> Within word capacity (padding region)
    @test indexSite(str, 65, true) === symI #> Beyond word capacity (would be OOB word read)
    @test indexSite(str, typemax(Int), true) === symI

    #> Tolerant mode still rejects non-positive input
    @test_throws DomainError indexSite(str, 0, true)
    @test_throws DomainError indexSite(str, -1, true)

    #> Zero-site identity composes with overflow mode
    @test indexSite(PauliStr(0), 1, true) === symI
    @test_throws DomainError indexSite(PauliStr(0), 1)
end

#> countWeight
@test countWeight(PauliStr(0)) == countWeight(pauli"III") == 0
@test countWeight(pauli"XIZ") == 2
@test countWeight(PauliStr(pauli"XIZ", 100)) == 2  #> Padding does not change the weight

end