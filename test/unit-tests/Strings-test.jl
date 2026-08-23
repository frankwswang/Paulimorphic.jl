using Test
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg, setCoeff!

@testset "Strings.jl" begin

m = "X"
@test (@pauli_str "$m") == pauli"X" == pauli"X"
@test (@pauli_str [2,3,2,0,1]) == pauli"XYXIZ" #> 0->I, 1->Z, 2->X, 3->Y

@testset "toString" begin
    @testset "full-string method" begin
        #> Sparse (default) format: identity sites omitted, subscripted site indices
        @test toString(pauli"IIX") == "+X₃"
        @test toString(pauli"XXIX") == "+X₁X₂X₄"
        @test toString(pauli"X", omitPlusSign=true) == "X₁"

        #> Dense format: one character per site; `omitPlusSign` defaults to `denseString`
        @test toString(pauli"IIX", denseString=true) == "IIX"
        @test toString(pauli"XXIX", denseString=true) == "XXIX" #> Exact macro round-trip
        @test toString(pauli"XXIX", denseString=true, omitPlusSign=false) == "+XXIX"

        #> Phase prefixes in both formats
        @test toString(PauliStr(2, symX, posImg)) == "+im*X₁X₂"
        @test toString(PauliStr(2, symX, posImg), denseString=true) == "im*XX"

        #> All-identity strings: sparse body collapses to "I"; dense body emits every site
        @test toString(PauliStr(3, symI, negRea)) == "-I"
        @test toString(PauliStr(3, symI, negRea), denseString=true) == "-III"
        @test toString(PauliStr(0, symI, negImg)) == 
            toString(PauliStr(0, symI, negImg), denseString=true) == "-im*I"
    end

    @testset "per-site method" begin
        p = pauli"XZIY"

        @test toString(p, 1) == "X₁" && toString(p, 2) == "Z₂" && toString(p, 4) == "Y₄"
        @test toString(p, 3) == "" && toString(p, 3, denseString=true) == "I"
        @test toString(p, 4, denseString=true) == "Y" #> Dense factors carry no subscript

        #> The phase of the parent string does not leak into per-site output
        @test toString(PauliStr(2, symX, negRea), 1) == "X₁"

        #> Range violations are delegated to `indexSite`
        @test_throws DomainError toString(p, 0)
        @test_throws DomainError toString(p, 5)

        #> Site-wise concatenation reproduces the corresponding full-string body
        @test toString(p; omitPlusSign=true) == 
            join(toString(p, i) for i in 1:countSites(p))
        @test toString(p; denseString=true) == 
            join(toString(p, i; denseString=true) for i in 1:countSites(p))

        #> Other ´Integer´ subtypes are accepted as the site index
        @test toString(p, UInt8(2)) == "Z₂"
    end
end

@testset "PauliStr printing" begin
    ctx = (:limit=>true)

    #> Non-limited IO stays faithful regardless of length (`repr`/`string`/`print` path)
    q = PauliStr(30, symX)
    fullQ = "+X₁X₂X₃X₄X₅X₆X₇X₈X₉X₁₀X₁₁X₁₂X₁₃X₁₄X₁₅X₁₆X₁₇X₁₈X₁₉X₂₀" * 
            "X₂₁X₂₂X₂₃X₂₄X₂₅X₂₆X₂₇X₂₈X₂₉X₃₀"
    @test repr(q) == string(q) == toString(q) == fullQ

    #> Limited IO elides when `countWeight(q) > 20`: first 9 factors, `" … "`, last 9
    limQ = "+X₁X₂X₃X₄X₅X₆X₇X₈X₉ … X₂₂X₂₃X₂₄X₂₅X₂₆X₂₇X₂₈X₂₉X₃₀"
    @test repr(q; context=ctx) == limQ

    @testset "site-count annotation (rich display)" begin
        #> The REPL's rich display (3-arg fallback) inherits the truncation from the context
        @test sprint(show, MIME"text/plain"(), q; context=ctx) == limQ * "  (30 sites)"

        str1 = @pauli_str vcat(fill(0, 32), 1, 0) #> 34 sites, weight 1

        #> Rich scalar display appends the site count; faithful outputs never do
        @test sprint(show, MIME"text/plain"(), str1) == "+Z₃₃  (34 sites)"
        @test repr(str1) == string(str1) == toString(str1) == "+Z₃₃"

        #> Rich `Vector` display annotates each single-line element (as it does `Function`s)
        @test sprint(show, MIME"text/plain"(), [str1]; context=(:limit=>true)) == 
            "1-element Vector{PauliStr}:\n +Z₃₃  (34 sites)"

        #> Containers rendered through the `MIME`-less `show` stay bare
        @test sprint(show, (str1,); context=(:limit=>true)) == "(+Z₃₃,)"
        @test sprint(show, MIME"text/plain"(), (str1,); context=(:limit=>true)) == "(+Z₃₃,)"
        @test sprint(print, [str1]) == "PauliStr[+Z₃₃]"
        @test sprint(print, (str1,)) == "(+Z₃₃,)"

        #> `:compact` rich display falls back to the plain form (as `Function`'s does)
        @test sprint(show, MIME"text/plain"(), str1; context=(:compact=>true)) == "+Z₃₃"

        #> No annotation for site counts below 2
        @test sprint(show, MIME"text/plain"(), pauli"X") == "+X₁"
        @test sprint(show, MIME"text/plain"(), PauliStr(0)) == "+I"

        #> Truncation and annotation compose in rich display
        q = PauliStr(30, symX)
        @test sprint(show, MIME"text/plain"(), q; context=(:limit=>true)) == 
              "+X₁X₂X₃X₄X₅X₆X₇X₈X₉ … X₂₂X₂₃X₂₄X₂₅X₂₆X₂₇X₂₈X₂₉X₃₀  (30 sites)"
    end

    #> Boundary: exactly 20 factors is never elided
    r = PauliStr(20, symZ)
    @test repr(r; context=ctx) == repr(r)

    #> 21 factors: sites 10–12 are hidden; phase prefix is preserved
    s = PauliStr(21, symY, negRea)
    @test repr(s; context=ctx) == "-Y₁Y₂Y₃Y₄Y₅Y₆Y₇Y₈Y₉ … Y₁₃Y₁₄Y₁₅Y₁₆Y₁₇Y₁₈Y₁₉Y₂₀Y₂₁"

    #> Identity sites do not count toward the cutoff, and short strings are unaffected
    p = pauli"XZIY"
    @test repr(p; context=ctx) == repr(p) == "+X₁Z₂Y₄"

    #> Sparse factors (not raw sites) are the elided units: 50 sites but weight 25
    t = @pauli_str "IX"^25
    @test repr(t; context=ctx) == 
          "+X₂X₄X₆X₈X₁₀X₁₂X₁₄X₁₆X₁₈ … X₃₄X₃₆X₃₈X₄₀X₄₂X₄₄X₄₆X₄₈X₅₀"
end

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

    #> Implicit-site mode: in-range reads identical to strict mode
    @test all(indexSite(str, i, true) === indexSite(str, i) for i in 1:4)

    #> Implicit-site mode: beyond the explicit site count reads as implicit identity
    @test indexSite(str, 5, true) === symI
    @test indexSite(str, 64, true) === symI #> Within word capacity (padding region)
    @test indexSite(str, 65, true) === symI #> Beyond word capacity (would be OOB word read)
    @test indexSite(str, typemax(Int), true) === symI

    #> Implicit-site mode still rejects non-positive input
    @test_throws DomainError indexSite(str, 0, true)
    @test_throws DomainError indexSite(str, -1, true)

    #> Zero-site identity composes with implicit-site mode
    @test indexSite(PauliStr(0), 1, true) === symI
    @test_throws DomainError indexSite(PauliStr(0), 1)
end

@testset "indexTerm" begin
    ham = PauliSum([pauli"XX", pauli"IZ", pauli"ZI"], [1.0, 2.0, 3.0])
    #> Canonical order (weight, then highest differing site): ZI, IZ, XX
    @test indexTerm(ham, 1) == (pauli"ZI" => 3.0)
    @test indexTerm(ham, 2) == (pauli"IZ" => 2.0)
    @test indexTerm(ham, 3) == (pauli"XX" => 1.0)

    #> `copyStr=true` (default): equal content, but a fresh copy holding no reference
    term = indexTerm(ham, 1)
    @test term.first == ham.str[begin] && term.first !== ham.str[begin]

    #> `copyStr=false`: aliases the stored string
    @test indexTerm(ham, 1, false).first === ham.str[begin]

    #> Out-of-range `i` throws `DomainError` (not `BoundsError`)
    @test_throws DomainError indexTerm(ham, 0)
    @test_throws DomainError indexTerm(ham, 4)
    @test_throws DomainError indexTerm(PauliSum(Float64), 1) #> Empty sum: domain is `1:0`
end

@testset "setCoeff!" begin
    ham = PauliSum([pauli"XX", pauli"IZ", pauli"ZI"], [1.0, 2.0, 3.0])
    #> Canonical order (weight, then highest differing site): ZI, IZ, XX

    #> Returns the updated term with the coefficient converted to `Complex{T}`
    @test setCoeff!(ham, -0.5im, 2) == (pauli"IZ" => -0.5im)
    @test ham.coeff == ComplexF64[3.0, -0.5im, 1.0] #> Only the target term is modified

    #> `copyStr` pass-through mirrors `indexTerm`
    @test setCoeff!(ham, 2.0, 2, false).first === ham.str[begin+1]

    #> Documented non-merging: a zeroed coefficient leaves the term in place
    setCoeff!(ham, 0, 3) #> `Real` input: converted, not rejected
    @test length(ham.str) == 3 && iszero(ham.coeff[end])

    #> Out-of-range `i` throws `DomainError` before any mutation occurs
    @test_throws DomainError setCoeff!(ham, 99.0, 0)
    @test_throws DomainError setCoeff!(ham, 99.0, 4)
    @test ham.coeff == ComplexF64[3.0, 2.0, 0.0] #> Failed calls left `ham` untouched
end

#> `countWeight`
@test countWeight(PauliStr(0)) == countWeight(pauli"III") == 0
@test countWeight(pauli"XIZ") == 2
@test countWeight(PauliStr(pauli"XIZ", 100)) == 2  #> Padding does not change the weight

#> `isHermitian`
@test  isHermitian(PauliSum([PauliStr(1, symY, Paulimorphic.negImg)], Complex{Int}(0, 2)))
@test !isHermitian(PauliSum([pauli"Y"], Complex{Int}(0, 2)))
@test !isHermitian(PauliSum([pauli"Y"], Complex{Int}(1, 1)))
@test  isHermitian(PauliSum(Int))

#> `isIdentity`
@test !isIdentity(PauliSum(Int))
@test  isIdentity(PauliSum(Int, [PauliStr(2)]))
@test  isIdentity(mul(PauliSum([mul(pauli"II", Paulimorphic.negRea)], Complex{Int}(-1)), 1))

@testset "toPauliStr" begin
    #> Single term with a coefficient expressible as a phase
    for (c, phase) in ((1, posRea), (-1, negRea), (im, posImg), (-im, negImg))
        op = PauliSum([pauli"XZ"], Complex{Rational{Int}}(c))
        @test toPauliStr(op) == mul(pauli"XZ", phase)
        @test PauliSum(Int, [toPauliStr(op)]) == op #> Exact round trip as operators
    end

    #> Fallback-path checking
    badCoeff = PauliSum([pauli"X"], Complex{Rational{Int}}(2, 0))
    @test toPauliStr(badCoeff, pauli"Z") == pauli"Z"
    twoTerm = PauliSum([pauli"X", pauli"Y"], Complex{Int}[1, 1])
    @test toPauliStr(twoTerm, pauli"Z") == pauli"Z"
    @test toPauliStr(PauliSum(Int), pauli"Z") == pauli"Z" #> Zero-term operator

    #> Throwing path and per-specialization type stability
    @test_throws "exactly one term" toPauliStr(twoTerm)
    @test_throws "must carry a coefficient" toPauliStr(badCoeff)
    @test (@inferred toPauliStr(PauliSum(Int, [pauli"X"]))) isa PauliStr
    @test (@inferred toPauliStr(badCoeff, pauli"Z")) isa PauliStr
end

end