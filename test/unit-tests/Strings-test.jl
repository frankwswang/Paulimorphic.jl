using Test
using Paulimorphic
using Paulimorphic: posRea, posImg, negRea, negImg, mul, sub, compareBits, 
                    isExplicitZeroOp, sortStrings!, checkPauliTermOrder, isPlainPauliStr, 
                    genPlainPauliStrKey, directOrderPauliStr, checkSiteCountMatch
using Random

@testset "Strings.jl" begin

m = "X"
@test (@pauli_str "$m") == pauli"X" == pauli"X"
@test (@pauli_str [2,3,2,0,1]) == pauli"XYXIZ" #> 0->I, 1->Z, 2->X, 3->Y

@testset "`@pauli_str` phase marker" begin
    #> Every marker `toString` can emit is accepted, with `"+"` optional
    @test pauli"+XXIX" == pauli"XXIX"
    @test pauli"iXXIX" == pauli"+iXXIX" == mul(pauli"XXIX", posImg)
    @test pauli"-XXIX" == mul(pauli"XXIX", negRea)
    @test pauli"-iXXIX" == mul(pauli"XXIX", negImg)

    #> Dense-format round trip for every phase, with either value of `omitPlusSign`
    for p in (posRea, posImg, negRea, negImg)
        s = mul(pauli"XZIY", p)
        @test (@pauli_str toString(s, denseString=true)) == s
        @test (@pauli_str toString(s, denseString=true, omitPlusSign=false)) == s
    end

    #> A marker alone yields the zero-site unit carrying that phase
    @test pauli"" == PauliStr()
    @test pauli"-" == PauliStr(0, symI, negRea)
    @test pauli"i" == PauliStr(0, symI, posImg)

    #> A runtime `String` goes through the same parser as the literal
    @test (@pauli_str "-" * "i" * "ZZ") == mul(pauli"ZZ", negImg)

    #> Case-sensitive: a lowercase `i` is the imaginary unit, never an identity site
    @test pauli"iIXZ" == mul(pauli"IXZ", posImg) && countSites(pauli"iIXZ") == 3
    @test pauli"-iXZ" == mul(pauli"XZ", negImg) && countSites(pauli"-iXZ") == 2
    @test_throws "phase marker" pauli"xXZ" #> Lowercase body letters are not Pauli letters
    @test_throws "character" pauli"Xx"

    #> The marker is the prefix preceding the first Pauli letter: anything after that
    #> point is a Pauli letter, or else fails as one
    @test_throws "character" pauli"XXIX-"
    @test_throws "character" pauli"X-X"
    @test_throws "character" pauli"XXIXi"

    #> Prefixes that are not one of the six markers fail as a phase marker
    @test_throws "phase marker" pauli"-1XX"
    @test_throws "phase marker" pauli"imXX"
    @test_throws "phase marker" pauli"im*XX"
    @test_throws "phase marker" pauli"i*XX"
    @test_throws "phase marker" pauli"+-XX"
    @test_throws "phase marker" pauli"--XX"
    @test_throws "phase marker" pauli"ii XX"
    @test_throws "phase marker" pauli"*XX"
    @test_throws "phase marker" pauli"i-XX"
    @test_throws "phase marker" pauli" -XX"
    @test_throws "phase marker" pauli"garbage"

    #> The integer-vector form carries the unit phase only
    @test (@pauli_str [2, 2, 0, 2]) == pauli"+XXIX"
end

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
        @test toString(PauliStr(2, symX, posImg)) == "+iX₁X₂"
        @test toString(PauliStr(2, symX, posImg), denseString=true) == "iXX"

        #> All-identity strings: sparse body collapses to "I"; dense body emits every site
        @test toString(PauliStr(3, symI, negRea)) == "-I"
        @test toString(PauliStr(3, symI, negRea), denseString=true) == "-III"
        @test toString(PauliStr(0, symI, negImg)) == 
            toString(PauliStr(0, symI, negImg), denseString=true) == "-iI"
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

@testset "`PauliStr` printing" begin
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
    #> Pitfall: drafts that graded by weight first compared weights across different 
    #> ambient spaces, e.g. ranking a 2-site identity before a 1-site X.
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

    #>≡≡≡ Consistency with the cached sort keys of `sortStrings!` ≡≡≡<#
    #> `isless` compares the words of two strings directly, whereas `sortStrings!` sorts 
    #> bit-plain strings by their keys from `genPlainPauliStrKey`: the two must order every 
    #> pair of bit-plain strings alike (`filter` keeps the test word-size independent)
    plainPool = filter(isPlainPauliStr, 
                       [PauliStr(0), PauliStr(0, symI, negImg), pauli"I", pauli"X", 
                        pauli"II", pauli"YY", pauli"IIII", pauli"IIIZ", pauli"YI", 
                        pauli"ZZ", pauli"XX", pauli"ZY", pauli"-XZ", pauli"iZY", 
                        PauliStr(genSparseList(64, 64 => symX)), 
                        PauliStr([fill(symX, 63); symI]), 
                        PauliStr(genSparseList(64, 32 => symX)), 
                        PauliStr(genSparseList(64, 33 => symZ)), 
                        PauliStr(genSparseList(64, 1 => symZ, 64 => symX)), 
                        PauliStr(genSparseList(64, 1 => symZ, 64 => symY)), 
                        PauliStr(genSparseList(64, 1 => symX, 64 => symX)), 
                        PauliStr(genSparseList(64, 1 => symX, 64 => symX), negRea), 
                        phasedStrs...])
    @test length(plainPool) > length(phasedStrs)
    @test all(isless(a, b) == isless(genPlainPauliStrKey(a), genPlainPauliStrKey(b)) 
              for a in plainPool, b in plainPool)

    #>≡≡≡ Site-count guard of `directOrderPauliStr` ≡≡≡<#
    #> `directOrderPauliStr` reads the words of both strings without bounds checking, so 
    #> its guard must hold on every call, including one that is inlined under `@inbounds`
    @test checkSiteCountMatch(pauli"XZ", pauli"XY") === nothing
    @test_throws ArgumentError checkSiteCountMatch(pauli"X", pauli"XX")
    @test directOrderPauliStr(pauli"XZ", pauli"XY") #> `Z < Y` at the highest site
    @test !directOrderPauliStr(pauli"XY", pauli"XZ")
    @test_throws ArgumentError directOrderPauliStr(pauli"X", pauli"XX")
    function directOrderInbounds(a::PauliStr, b::PauliStr)
        @inbounds directOrderPauliStr(a, b)
    end
    @test directOrderInbounds(pauli"XZ", pauli"XY")
    @test_throws ArgumentError directOrderInbounds(pauli"X", pauli"XX")
    #> Strings whose buffers differ in length never reach the word comparison of `isless`
    str64, str65 = PauliStr(genSparseList(64)), PauliStr(genSparseList(65))
    @test str64 < str65 && !(str65 < str64)
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

@testset "PauliSum" begin
    h = PauliSum([pauli"X", PauliStr([symZ], posImg)], [1e-50, 2.0])
    r = PauliSum(Float32, h)

    @test r isa PauliSum{Float32}
    @test 1 == countTerms(r)        #> `1e-50` rounds to `0f0` and is dropped
    @test r.coeff == [0f0 + 2f0im]
    @test r == PauliSum(r, true)    #> Canonicalization
    @test r.str[1] !== h.str[2]     #> Strings are rebuilt
    @test PauliSum(Float64, h) == h #> Equal result for the same `T`
    @test_throws InexactError PauliSum(Int, PauliSum([pauli"X"], [0.5]))
    @test_throws MethodError PauliSum(Bool, h)
    @test_throws MethodError PauliSum(UInt8, h)
    @test_throws MethodError PauliSum(Rational{UInt8}, h)
    @test_throws ArgumentError PauliSum(AbstractFloat, [pauli""])
    @test_throws ArgumentError PauliSum(Rational{<:Signed}, [pauli""])
    #> Sign-less coefficient inputs are rejected at dispatch (`im` is a `Complex{Bool}`)
    @test_throws MethodError PauliSum(Float64, [pauli"X"], im)
    @test_throws MethodError PauliSum([pauli"X"], [true])

    #> Coefficient summation accuracy check
    res1 = PauliSum(fill(pauli"X", 3), [1e16, 1.0, -1e16])
    @test countTerms(res1) == 1
    @test res1.coeff == Complex{Float64}[1.0]

    res2 = PauliSum(fill(pauli"X", 4), [1e16, 1.0, -1e16, -1.0])
    @test countTerms(res2) == 0

    #> `simplification=false`: equal strings are merged all the same (the canonical form 
    #> is unconditional); only the terms whose merged coefficient is exactly zero are 
    #> retained, which leaves the sum outside the simplified form
    res3 = PauliSum(fill(pauli"X", 3), [1e16, 1.0, -1e16], false)
    @test res3 == res1
    res4 = PauliSum(fill(pauli"X", 4), [1e16, 1.0, -1e16, -1.0], false)
    @test countTerms(res4) == 1 && res4.coeff == Complex{Float64}[0.0]
    @test isCanonical(res4) && !isCanonical(res4, true)
    @test res4 == PauliSum(res4, false) && PauliSum(res4) == res2

    #> A merged coefficient that is not finite is the one that `+` gives, instead of a 
    #> `NaN` made by a residue that is not finite either
    @test PauliSum(fill(pauli"X", 2), [1e308, 1e308]).coeff == [Inf]
    @test PauliSum(fill(pauli"X", 2), [1.0, -Inf]).coeff == [-Inf]
    @test PauliSum(fill(pauli"X", 3), [1e308, 2.0im, 1e308]).coeff == [Inf + 2.0im]
    @test isnan(first(PauliSum(fill(pauli"X", 2), [Inf, -Inf]).coeff))
    #> Merging two coefficients is the same as adding them, down to the sign of a zero
    for (c1, c2) in ((1e308, 1e308), (Inf, 1.0), (Inf, -Inf), (0.1, 0.2), (1e16, 1.0), 
                     (Complex(-0.0, 0.1), Complex(-0.0, 0.2)), (-0.0, -0.0), (1.0, -1.0))
        merged = first(PauliSum(fill(pauli"X", 2), [c1, c2], false).coeff)
        @test isequal(merged, Complex{Float64}(c1) + Complex{Float64}(c2))
    end
    #> The accumulator of Pauli terms concludes its sums in the same way
    termDict = Paulimorphic.PauliTermDict{Float64}()
    foreach(c->Paulimorphic.updateNeuAccumulator!(termDict, pauli"X"=>c), (1e308, 1e308))
    @test Paulimorphic.combinePauliTerms!(termDict).coeff == [Inf]

    #> Sub-`T` cancellation survives the extended-precision merge (accuracy-contract pin)
    @test PauliSum(Float32, fill(pauli"X", 2), [1.0 + 2.0^-26, -1.0]).coeff == 
          Complex{Float32}[1.4901161f-8]
    #> Extended-precision merge defers conversion failures to the single final rounding
    @test PauliSum(Rational{Int16}, fill(pauli"X", 2), [0.1, 0.4]).coeff == 
          Complex{Rational{Int16}}[1//2]
    #> Terms whose `Complex{T}`-converted coefficients are zero are dropped
    @test countTerms(PauliSum(Float32, fill(pauli"X", 2), [1e-50, 1e-50])) == 0

    #> Exact coefficient types: a fixed-width integer overflow throws instead of wrapping 
    #> around, in the merge of equal strings as well as in the phase absorption, whereas 
    #> a wider core type `T` provides the room to evaluate the same inputs
    @test_throws OverflowError PauliSum(fill(pauli"X", 2), [typemax(Int), 1])
    @test_throws OverflowError PauliSum(fill(pauli"X", 2), Int8[100, 100])
    @test PauliSum(Int, fill(pauli"X", 2), Int8[100, 100]) == PauliSum([pauli"X"], [200])
    @test_throws OverflowError PauliSum(fill(pauli"X", 2), [typemax(Int)//1, 1//1])
    @test PauliSum(fill(pauli"X", 2), [big(typemax(Int)), big(1)]).coeff == 
          [big(typemax(Int)) + 1]
    @test_throws OverflowError PauliSum([pauli"-X"], [typemin(Int)])
    @test_throws OverflowError PauliSum([pauli"-X"], [typemin(Int8)])
    @test PauliSum(Int, [pauli"-X"], [typemin(Int8)]) == PauliSum([pauli"X"], [128])

    #> Input-order independence: the same multiset of terms gives an identical sum
    strs = [pauli"X", pauli"Z", pauli"X", pauli"Y", pauli"X", pauli"Z"]
    cs = [0.1, -0.7, 0.2, 1e-3, -0.3, 0.7]
    ref = PauliSum(strs, cs)
    @test countTerms(ref) == 2 && ref.coeff == ComplexF64[2.0^-55, 1e-3] #> `X`: 2⁻⁵⁵
    for p in ([2, 1, 3, 6, 5, 4], [6, 5, 4, 3, 2, 1], [3, 1, 5, 2, 6, 4], 
              [5, 3, 1, 4, 2, 6])
        @test PauliSum(strs[p], cs[p]) == ref
    end
    #> Terms that are already in the canonical order skip the sort: merging them in that 
    #> order yields the same result (bit for bit) as sorting them first
    rng = Xoshiro(47)
    syms = (symI, symX, symY, symZ)
    bl = true
    for _ in 1:200
        strs = [PauliStr([syms[rand(rng, 1:4)] for _ in 1:2]) for _ in 1:rand(rng, 1:12)]
        cs = [complex(randn(rng), randn(rng)) * 2.0^rand(rng, -30:30) for _ in strs]
        orderKey = i->(strs[i], Paulimorphic.evalMagnitude(cs[i]), real(cs[i]), imag(cs[i]))
        perm = sort!(collect(eachindex(strs)); by=orderKey, alg=Base.Sort.DEFAULT_STABLE)
        hUnsorted, hSorted = PauliSum(strs, cs), PauliSum(strs[perm], cs[perm])
        bl &= hUnsorted.str == hSorted.str && isequal(hUnsorted.coeff, hSorted.coeff)
    end
    @test bl

    @testset "Term-filtering construction" begin
        #> Stored order: ZI, IZ, XX
        h = PauliSum([pauli"XX", pauli"IZ", pauli"ZI"], [1.0, 2.0, 3.0])
        #> `selector` receives each term as a `Pair`; retained strings are shared with `h`
        hf = PauliSum(term->countWeight(term.first) == 1 && real(term.second) > 2.0, h)
        @test hf == PauliSum([pauli"ZI"], [3.0]) && hf.str[begin] === h.str[begin]
        @test PauliSum(term->true, h) == h && PauliSum(term->false, h) == PauliSum(Float64)
        @test PauliSum(term->true, PauliSum(Int)) == PauliSum(Int)
        #> `checkForm` (`true` by default): a non-canonical `h` is rejected unless the 
        #> check is lifted, in which case the retained terms simply keep the order of `h`
        hd = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
        stamp!(indexTerm(hd, 1, false).first, 1, symX) #> `Z` -> `X`: equal strings
        @test_throws ArgumentError PauliSum(term->true, hd)
        @test PauliSum(term->true, hd, false).coeff == ComplexF64[2.0, 1.0]
        @test PauliSum(term->real(term.second) < 1.5, hd, false).coeff == ComplexF64[1.0]
    end
end

@testset "indexTerm" begin
    h = PauliSum([pauli"XX", pauli"IZ", pauli"ZI"], [1.0, 2.0, 3.0])
    #> Canonical order (weight, then highest differing site): ZI, IZ, XX
    @test indexTerm(h, 1) == (pauli"ZI" => 3.0)
    @test indexTerm(h, 2) == (pauli"IZ" => 2.0)
    @test indexTerm(h, 3) == (pauli"XX" => 1.0)

    #> `copyStr=true` (default): equal content, but a fresh copy holding no reference
    term = indexTerm(h, 1)
    @test term.first == h.str[begin] && term.first !== h.str[begin]

    #> `copyStr=false`: aliases the stored string
    @test indexTerm(h, 1, false).first === h.str[begin]

    #> Out-of-range `i` throws `DomainError` (not `BoundsError`)
    @test_throws DomainError indexTerm(h, 0)
    @test_throws DomainError indexTerm(h, 4)
    @test_throws DomainError indexTerm(PauliSum(Float64), 1) #> Empty sum: domain is `1:0`
end

@testset "listTerms" begin
    h = PauliSum([pauli"XX", pauli"IZ", pauli"ZI"], [1.0, 2.0im, 3.0])
    #> Canonical order (weight, then highest differing site): ZI, IZ, XX
    terms = listTerms(h)
    @test terms isa Vector{Pair{PauliStr, ComplexF64}}
    @test terms == [pauli"ZI" => 3.0, pauli"IZ" => 2.0im, pauli"XX" => 1.0]
    @test length(terms) == countTerms(h)

    #> Element-wise agreement with `indexTerm`
    @test all(terms[i] == indexTerm(h, i) for i in eachindex(terms))

    #> Rebuilding from the listed terms recovers an equal sum
    @test PauliSum(first.(terms), last.(terms)) == h

    #> `copyStr=true` (default): equal content, but fresh copies holding no reference
    @test all(t.first == s && t.first !== s for (t, s) in zip(terms, h.str))

    #> `copyStr=false`: aliases the stored strings
    @test all(t.first === s for (t, s) in zip(listTerms(h, false), h.str))

    #> Empty sum: an empty `Vector` of the documented element type
    @test listTerms(PauliSum(Float64)) == Pair{PauliStr, ComplexF64}[]
end

@testset "setCoeff!" begin
    h = PauliSum([pauli"XX", pauli"IZ", pauli"ZI"], [1.0, 2.0, 3.0])
    #> Canonical order (weight, then highest differing site): ZI, IZ, XX

    #> Returns the updated term with the coefficient converted to `Complex{T}`
    @test setCoeff!(h, -0.5im, 2) == (pauli"IZ" => -0.5im)
    @test h.coeff == ComplexF64[3.0, -0.5im, 1.0] #> Only the target term is modified

    #> `copyStr` pass-through mirrors `indexTerm`
    @test setCoeff!(h, 2.0, 2, false).first === h.str[begin+1]

    #> Documented non-removal: a zeroed coefficient leaves the term in place, which keeps 
    #> the canonical form (independent of the coefficients) but not the simplified form
    setCoeff!(h, 0, 3) #> `Real` input: converted, not rejected
    @test length(h.str) == 3 && iszero(h.coeff[end])
    @test isCanonical(h) && !isCanonical(h, true)
    @test h == PauliSum(h, false) && countTerms(PauliSum(h)) == 2

    #> Out-of-range `i` throws `DomainError` before any mutation occurs
    @test_throws DomainError setCoeff!(h, 99.0, 0)
    @test_throws DomainError setCoeff!(h, 99.0, 4)
    @test h.coeff == ComplexF64[3.0, 2.0, 0.0] #> Failed calls left `h` untouched

    #> `coeff` is bound like every other coefficient input: `Bool`, `Unsigned`, and their 
    #> complex forms (e.g., `im`) are rejected; a value that `Complex{T}` cannot represent 
    #> throws `InexactError`
    @test_throws MethodError setCoeff!(h, im, 1) #> `im` is a `Complex{Bool}`
    @test_throws MethodError setCoeff!(h, true, 1)
    @test_throws MethodError setCoeff!(h, 0x01, 1)
    @test setCoeff!(h, 1im, 1) == (pauli"ZI" => 1.0im) #> `Complex{Int}` is accepted
    hi = PauliSum([pauli"X"], [1])
    @test_throws InexactError setCoeff!(hi, 0.5, 1)
    @test_throws InexactError setCoeff!(hi, 1.5im, 1) #> Converted componentwise
    @test hi.coeff == Complex{Int}[1] #> Failed conversions left `hi` untouched
end

@testset "isCanonical" begin
    #> Constructor outputs are canonical, and simplified when `simplification=true`
    h = PauliSum([pauli"XX", pauli"IZ", pauli"ZI"], [1.0, 2.0, 3.0])
    @test isCanonical(h) && isCanonical(h, true)
    @test isCanonical(PauliSum(Int)) && isCanonical(PauliSum(Int), true)
    @test h == PauliSum(h, false) == PauliSum(h)

    #> Retained zero-coefficient terms: canonical but not simplified
    strs0 = [pauli"XX", pauli"IZ", pauli"ZI", pauli"ZI"]
    h0 = PauliSum(strs0, [1.0, 0.0, 3.0, -3.0], false)
    @test h0.str == [pauli"ZI", pauli"IZ", pauli"XX"] && h0.coeff == ComplexF64[0, 0, 1]
    @test isCanonical(h0) && !isCanonical(h0, true)
    @test h0 == PauliSum(h0, false) && h0 != PauliSum(h0)
    @test PauliSum(h0) == PauliSum([pauli"XX"], [1.0])

    #> Violations reachable only through the aliasing form of `indexTerm`: a phase left 
    #> inside a stored string, strings out of order, and equal strings (the canonical order 
    #> of `X` and `Z` on one site is `Z`, `X`)
    hp = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    scale!(indexTerm(hp, 1, false).first, negRea)
    @test !isCanonical(hp)
    ho = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    stamp!(indexTerm(ho, 1, false).first, 1, symY) #> `Z` -> `Y`, which sorts after `X`
    @test !isCanonical(ho) && allunique(ho.str)
    hd = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    stamp!(indexTerm(hd, 1, false).first, 1, symX) #> `Z` -> `X`
    @test !isCanonical(hd) && issorted(hd.str)

    #> Inconsistent site counts through direct assignment to `.str` (the strings below 
    #> are still in ascending order, so only the site-count check can detect it)
    hn = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    hn.str[begin+1] = pauli"XI"
    @test !isCanonical(hn) && issorted(hn.str)
    @test hn != PauliSum(hn, false)
end

@testset "canonicalize!" begin
    #> A canonical sum is left as is, with the same term count
    h = PauliSum([pauli"XX", pauli"IZ", pauli"ZI"], [1.0, 2.0, 3.0])
    hc = deepcopy(h)
    @test canonicalize!(hc) === hc && hc == h
    @test canonicalize!(PauliSum(Float64)) == PauliSum(Float64)

    #> In-place restoration: a phase left inside a stored string, and strings out of order 
    #> (both through the aliasing form of `indexTerm`; the stored order of `X` and `Z` on 
    #> one site is `Z`, `X`)
    hp = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    scale!(indexTerm(hp, 1, false).first, negImg)
    @test !isCanonical(hp)
    @test canonicalize!(hp) === hp && isCanonical(hp)
    @test hp == PauliSum([pauli"X", pauli"Z"], [1.0, -2.0im])
    ho = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    stamp!(indexTerm(ho, 1, false).first, 1, symY) #> `Z` -> `Y`: order becomes `Y`, `X`
    @test !isCanonical(ho)
    @test isCanonical(canonicalize!(ho)) && ho == PauliSum([pauli"Y", pauli"X"], [2.0, 1.0])

    #> Conditions that cannot be restored in place throw before any mutation: strings 
    #> equal up to their phases, and strings with different site counts
    hd = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    stamp!(indexTerm(hd, 1, false).first, 1, symX) #> `Z` -> `X`
    scale!(indexTerm(hd, 1, false).first, negRea)  #> ..., equal to the other up to phase
    hdRef = deepcopy(hd)
    @test_throws ArgumentError canonicalize!(hd)
    @test hd == hdRef
    @test PauliSum(hd, false) == PauliSum([pauli"X"], [-1.0]) #> The rebuild merges them
    hn = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    hn.str[begin+1] = pauli"XI"
    hnRef = deepcopy(hn)
    @test_throws ArgumentError canonicalize!(hn)
    @test hn == hnRef
    hn.str[begin] = pauli"XI" #> Equal strings again, now sharing a site count of two
    @test_throws ArgumentError canonicalize!(hn)

    #> Phase absorption is overflow-checked for exact coefficient types; a failure leaves 
    #> the offending term untouched, so `h` still represents the same operator
    hi = PauliSum([pauli"X", pauli"Z"], [typemin(Int), 1]) #> Stored: `Z` => 1, `X` => min
    indexTerm(hi, 2, false).first.phase = negRea
    @test_throws OverflowError canonicalize!(hi)
    @test hi.str[begin+1].phase == negRea && hi.coeff[begin+1] == Complex(typemin(Int))
    @test PauliSum(Int128, hi) == 
          PauliSum([pauli"X", pauli"Z"], [-Int128(typemin(Int)), Int128(1)])
end

@testset "sortStrings!" begin
    #> Real coefficients are accepted as well as complex ones. Among equal strings, ties 
    #> are broken by `(evalMagnitude(c), real(c), imag(c))` (`Z` precedes `X` on one site)
    strs = [pauli"X", pauli"X", pauli"Z"]
    coeffs = [2.0, -1.0, 3.0]
    @test sortStrings!(strs, coeffs) === nothing
    @test checkPauliTermOrder(strs, coeffs)
    @test strs == [pauli"Z", pauli"X", pauli"X"] && coeffs == [3.0, -1.0, 2.0]
    #> `considerCoeff=false`: the sort is stable, so equal strings keep their input order
    strs = [pauli"X", pauli"X", pauli"Z"]
    coeffs = [2.0, -1.0, 3.0]
    sortStrings!(strs, coeffs, false)
    @test !checkPauliTermOrder(strs, coeffs)
    @test checkPauliTermOrder(strs, [1.0, 1.0, 1.0])
    @test strs == [pauli"Z", pauli"X", pauli"X"] && coeffs == [3.0, 2.0, -1.0]
    #> Real and complex coefficients of equal values are ordered alike
    strs = [pauli"X", pauli"X", pauli"Z"]
    cplx = ComplexF64[2.0, -1.0, 3.0]
    sortStrings!(strs, cplx)
    @test checkPauliTermOrder(strs, cplx)
    @test cplx == [3.0, -1.0, 2.0]
    #> The sort keys are evaluated once per term (packed into isbits keys for strings on at 
    #> most 64 sites), yet the permutation stays that of a stable sort by the tuples 
    #> `(str, evalMagnitude(c), real(c), imag(c))`: the sorted strings are the same objects 
    #> in the same order
    rng = Xoshiro(46)
    syms = (symI, symX, symY, symZ)
    vals = (0.0, -0.0, 1.0, -1.0, 0.5, NaN)
    bl = true
    for _ in 1:200, nMax in (3, 64, 130)
        pool = [PauliStr([syms[rand(rng, 1:4)] for _ in 1:rand(rng, 0:nMax)], 
                         PhaseFactor(rand(rng, 0:3))) for _ in 1:rand(rng, 1:5)]
        nTerm = rand(rng, 1:30)
        strs = [PauliStr(rand(rng, pool)) for _ in 1:nTerm] #> Distinct equal objects
        coeffs = [complex(vals[rand(rng, 1:6)], vals[rand(rng, 1:6)]) for _ in 1:nTerm]
        for considerCoeff in (true, false)
            orderKey = if considerCoeff
                i -> let coeff=coeffs[i]
                    (strs[i], Paulimorphic.evalMagnitude(coeff), real(coeff), imag(coeff))
                end
            else
                i -> strs[i]
            end
            perm = sort!(collect(1:nTerm); by=orderKey, alg=Base.Sort.DEFAULT_STABLE)
            strsSorted, coeffsSorted = copy(strs), copy(coeffs)
            sortStrings!(strsSorted, coeffsSorted, considerCoeff)
            bl &= (!considerCoeff || checkPauliTermOrder(strsSorted, coeffsSorted))
            bl &= all(strsSorted .=== strs[perm]) && isequal(coeffsSorted, coeffs[perm])
        end
    end
    @test bl
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
#>> A term whose string still carries a phase is judged by the value it represents
hStr = PauliSum([pauli"Z"], [typemin(Int)])
indexTerm(hStr, 1, false).first.phase = posImg #> `hStr` represents `typemin(Int)*im*Z`
@test !isHermitian(hStr)
setCoeff!(hStr, 2im, 1)                        #> `hStr` represents `-2*Z`
@test  isHermitian(hStr)

#> `isIdentity`
@test  isIdentity(PauliStr())
@test  isIdentity(PauliSum(Int, [PauliStr(2)]))
@test  isIdentity(mul(PauliSum([mul(pauli"II", Paulimorphic.negRea)], Complex{Int}(-1)), 1))
@test !isIdentity(PauliSum(Int))
@test !isIdentity(PauliStr(0, symI, posImg))
#>> Several identity strings are judged through the canonical rebuild, whose merge of 
#>> their coefficients is overflow-checked (so a sum that wraps around to 1 does not count)
hIds = PauliSum([pauli"X", pauli"Y", pauli"Z"], [typemax(Int), typemax(Int), 3])
foreach(i->stamp!(indexTerm(hIds, i, false).first, 1, symI), 1:3)
@test sum(hIds.coeff) == 1 #> `sum` wraps around: the coefficients are 3 and two `typemax`
@test_throws OverflowError isIdentity(hIds)
setCoeff!(hIds, 2, 2)
setCoeff!(hIds, -4, 3)
@test  isIdentity(hIds)    #> 3 + 2 - 4 == 1
setCoeff!(hIds, -3, 3)
@test !isIdentity(hIds)    #> 3 + 2 - 3 == 2

#> `isHermitian`, `isIdentity` only rely on the canonical form of a `PauliSum` when it 
#> represents the operator exactly. The terms that the constructor would merge are merged 
#> for the judgment when the core data type is an exact numeric type, whereas a 
#> floating-point `PauliSum` that stores such terms is judged from its stored terms alone.
for T in (Int, Rational{Int}, Float64)
    isExact = !(T <: AbstractFloat)
    #>> `hSelf` stores `X` three times: `-im*X + X + im*X == X`
    hSelf = PauliSum([pauli"X", pauli"Y", pauli"Z"], Complex{T}[1, im, -im])
    fill!(hSelf.str, pauli"X")
    @test isHermitian(hSelf) == isExact && isHermitian(PauliSum(hSelf))
    #>> `hUnit` stores `I` twice: `2*I - I == I`
    hUnit = PauliSum([pauli"I", pauli"Z"], T[2, -1])
    stamp!(indexTerm(hUnit, 2, false).first, 1, symI)
    @test isIdentity(hUnit) == isExact && isIdentity(PauliSum(hUnit))
    #>> `hNone` stores `X` three times: `-3*X + X + 2*X == 0`
    hNone = PauliSum([pauli"X", pauli"Y", pauli"Z"], T[1, 2, -3])
    fill!(hNone.str, pauli"X")
    @test isHermitian(hNone) && !isIdentity(hNone)
    #>> `isExplicitZeroOp` does not rely on canonicalization at all
    @test !isExplicitZeroOp(hNone) && isExplicitZeroOp(PauliSum(hNone))
end
#>> A floating-point `PauliSum` that restores its canonical form without merging terms is 
#>> still judged by the operator it represents, and so is one that retains zero-coefficient 
#>> terms
hTurn = PauliSum([pauli"Z", pauli"X"], [2.0im, 1.0])
indexTerm(hTurn, 1, false).first.phase = posImg #> `hTurn` represents `-2*Z + X`
@test !isCanonical(hTurn) && isHermitian(hTurn)
hSpare = PauliSum([pauli"I", pauli"X"], [1.0, 0.0], false)
@test  isIdentity(hSpare) && countTerms(hSpare) == 2
setCoeff!(hSpare, 1e-300, 2)
@test !isIdentity(hSpare)
#>> The stored terms settle the Hermiticity of `hReal` (real coefficients of Hermitian 
#>> strings), although it stores `X` twice
hReal = PauliSum([pauli"X", pauli"Z"], [0.1, 0.2])
stamp!(indexTerm(hReal, 1, false).first, 1, symX) #> The first stored term is `Z`
@test isHermitian(hReal) && !isExplicitZeroOp(hReal) && !isIdentity(hReal)
#>> For exact core data types, the merge follows the overflow checking policy of `PauliSum`
hTwice = PauliSum([pauli"X", pauli"Z"], Int8[100, 100])
stamp!(indexTerm(hTwice, 1, false).first, 1, symX) #> `hTwice` stores `X` twice
@test !isExplicitZeroOp(hTwice) #> `isExplicitZeroOp` simply fails instead of throwing
@test_throws OverflowError isIdentity(hTwice)
@test isHermitian(hTwice) #> Settled by the stored terms, without merging them
@test isExplicitZeroOp(PauliSum(Int8)) && !isExplicitZeroOp(pauli"I")

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

    @test_throws ArgumentError toPauliStr([0, 4], posRea)
    @test_throws ArgumentError toPauliStr([-1], negRea)
    @test toPauliStr([0, 4], posRea, pauli"X") == pauli"X"
    @test_throws ArgumentError (@pauli_str [0, 4])
    @test toPauliStr([0, 1, 2, 3], posRea) == pauli"IZXY"
    @test toPauliStr(UInt8[2, 2], negImg) == pauli"-iXX"
    @test toPauliStr(Int[], negRea) == pauli"-"

    @test (@inferred toPauliStr("-iXZ")) == pauli"-iXZ"
    @test toPauliStr("-1XX", pauli"Z") == pauli"Z"
    @test toPauliStr("X-X", pauli"Z") == pauli"Z"
    @test_throws "phase marker" toPauliStr("-1XX")
    @test_throws "`str[4]='-'`" toPauliStr("-iX-Z")
    @test_throws "`str[2]='x'`" toPauliStr("Xx")
end

@testset "sumCoeffs" begin
    h = PauliSum([pauli"XI", pauli"YI", pauli"ZI", pauli"IZ"], [1e16, 5.0, 1.0, -1e16])
    @test sumCoeffs(h) == Complex(6.0) #> Naive accumulation gives 4.0
    #> `selector` receives each term as a `Pair` of its string and its coefficient (at the 
    #> core data type of `h`, whatever `R` is), same as in the term-filtering construction 
    #> of `PauliSum`; hence one `selector` picks the same terms in both
    notYI = term->(term.first != pauli"YI")
    @test sumCoeffs(notYI, h) == Complex(1.0) #> Naive accumulation gives 0.0
    isPos = term->(real(term.second) > 0)
    @test sumCoeffs(isPos, h) === sumCoeffs(PauliSum(isPos, h)) === Complex(1e16 + 6.0)
    @test sumCoeffs(term->(term isa Pair{PauliStr, ComplexF64}), h, Float32) == 6.0f0
    @test sumCoeffs(h, Float32) == Complex{Float32}(6.0)
    @test sumCoeffs(PauliSum([pauli"X"], [1//3])) == Complex(1//3) #> Exact fallback path
    #> A floating-point overflow gives an infinite sum, as plain additions do
    @test sumCoeffs(PauliSum([pauli"X", pauli"Z"], [1e308, 1e308])) == Inf
    #> A fixed-width integer overflow throws instead of wrapping around; a wider `R` 
    #> accumulates the same terms safely
    hBig = PauliSum([pauli"X", pauli"Z"], [typemax(Int), 1])
    @test_throws OverflowError sumCoeffs(hBig)
    @test sumCoeffs(hBig, Int128) == Complex{Int128}(typemax(Int)) + 1
    #> A non-concrete `R` is rejected by the core-type check, whatever the core type of `h`
    @test_throws "core data type" sumCoeffs(h, Signed)
    @test_throws "core data type" sumCoeffs(h, AbstractFloat)
    @test_throws "core data type" sumCoeffs(PauliSum([pauli"X"], [3]), AbstractFloat)
    @test_throws ArgumentError sumCoeffs(h, Rational{<:Signed})

    #> `checkForm` (`true` by default): the accumulation follows the term order of `h`, so 
    #> a sum that is not in the canonical form is rejected unless the check is lifted
    #>> Canonical subsets of a canonical sum are accepted
    @test sumCoeffs(curtail(h, 2.0)) == Complex(5.0)
    #>> Retained zero-coefficient terms (canonical, not simplified) are accepted and do 
    #>> not alter the result: after `setCoeff!`, or kept by `simplification=false`
    hs = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    setCoeff!(hs, 0.0, 1)
    @test isCanonical(hs) && sumCoeffs(hs) == sumCoeffs(PauliSum(hs)) == Complex(1.0)
    hk = PauliSum([pauli"XI", pauli"YY", pauli"ZI", pauli"IZ"], [1e16, 5.0, 1.0, -1e16])
    hk = sub(hk, pauli"YY"=>5.0, false)
    @test countTerms(hk) == 4 && iszero(hk.coeff[end])
    @test sumCoeffs(hk) == sumCoeffs(PauliSum(hk)) == Complex(1.0)
    #>> Equal strings (through the aliasing form of `indexTerm`)
    hd = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    stamp!(indexTerm(hd, 1, false).first, 1, symX) #> `Z` -> `X`
    @test !isCanonical(hd)
    @test_throws ArgumentError sumCoeffs(hd)
    @test_throws ArgumentError sumCoeffs(term->true, hd, Float32)
    @test sumCoeffs(hd; checkForm=false) == Complex(3.0)
    @test sumCoeffs(PauliSum(hd)) == Complex(3.0) #> The merged rebuild
    #>> A phase left inside a stored string (through the aliasing form of `indexTerm`)
    hp = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    indexTerm(hp, 1, false).first.phase = negRea
    @test !isCanonical(hp)
    @test_throws ArgumentError sumCoeffs(hp)
    @test sumCoeffs(hp; checkForm=false) == Complex(3.0) #> Stored coefficients as they are
    @test sumCoeffs(canonicalize!(hp)) == Complex(-1.0)  #> The phase is absorbed first
end

@testset "curtail" begin
    h = PauliSum([pauli"XI", pauli"IZ", pauli"ZZ"], [1.0, 1e-16, -0.5im])
    @test curtail(h) == PauliSum([pauli"XI", pauli"ZZ"], [1.0, -0.5im]) #> Default: `8eps`
    @test curtail(h, 0.5) == PauliSum([pauli"XI"], [1.0])
    @test curtail(h, 0.5; inclusive=true) == PauliSum([pauli"XI", pauli"ZZ"], [1.0, -0.5im])
    @test curtail(h, 0.4; relative=true) == PauliSum([pauli"XI", pauli"ZZ"], [1.0, -0.5im])
    @test curtail(h, 0.6; relative=true) == PauliSum([pauli"XI"], [1.0])
    @test curtail(h, 0; inclusive=true) == h
    @test_throws DomainError curtail(h, -1.0)
    @test curtail(PauliSum(Float64)) == PauliSum(Float64)
    #> Exact coefficient types: the default threshold only removes exact zeros
    @test curtail(PauliSum([pauli"X"], [1//10^9])) == PauliSum([pauli"X"], [1//10^9])
    #> `checkForm` (`true` by default): a non-canonical `h` is rejected unless the check 
    #> is lifted, in which case the retained terms simply keep the order of `h`
    hd = PauliSum([pauli"X", pauli"Z"], [1.0, 2.0])
    stamp!(indexTerm(hd, 1, false).first, 1, symX) #> `Z` -> `X`: equal strings
    @test_throws ArgumentError curtail(hd)
    @test_throws ArgumentError curtail(hd, 1.5)
    @test_throws DomainError curtail(hd, -1.0) #> `tolerance` is validated before the form
    @test curtail(hd, checkForm=false).coeff == ComplexF64[2.0, 1.0]
    @test curtail(hd, 1.5, checkForm=false).coeff == ComplexF64[2.0]
end

#> Tests for the site manipulators: a list of 130 single-site operators with no period 
#> (every displaced copy of it differs from it on the sites they share), which spans three 
#> words of `.x` and `.z` when `UInt` is 64-bit.
longList = PauliSym[(symI, symZ, symX, symY)[begin + count_ones(7k) % 4] for k in 1:130]

@testset "sanitize!" begin
    w = 8 * sizeof(UInt) #> Number of sites packed into one word of `.x` and `.z`
    #> The bits above site `.n` (written here through the fields) are cleared from the 
    #> final word of both buffers, in place.
    s = PauliStr(3)
    fill!(s.x, ~zero(UInt))
    fill!(s.z, ~zero(UInt))
    @test s != pauli"YYY"
    @test sanitize!(s) === s && s == pauli"YYY"
    #> Only the final word can hold such bits: the words before it are left as they are
    s = PauliStr(w + 2)
    fill!(s.x, ~zero(UInt))
    @test sanitize!(s) == PauliStr(w + 2, symX)
    @test first(s.x) == ~zero(UInt) && last(s.x) == UInt(3)
    #> A final word without padding is left as it is, and so is a string without any word.
    @test last(sanitize!(PauliStr(2w, symY)).x) == ~zero(UInt)
    @test sanitize!(PauliStr()) == PauliStr()
    #> A string that already satisfies the invariant is unchanged, along with its phase.
    @test sanitize!(pauli"-iXZY") == pauli"-iXZY"
end

@testset "shift!" begin
    #> The operator on site `k` moves to site `k + nShift` by default, and to site 
    #> `k - nShift` when `lowToHigh=false`. The vacated sites carry the identity, and the 
    #> operators pushed out of `1:n` are dropped.
    @test shift!(pauli"XYZI", 2) == pauli"IIXY"
    @test shift!(pauli"IXYZ", 2, false) == pauli"YZII"
    @test shift!(pauli"XYZ", 3) == shift!(pauli"XYZ", 1000, false) == pauli"III"
    #> The string is mutated in place, and keeps its site count and its phase.
    s = pauli"-iXZI"
    @test shift!(s, 1) === s && s == pauli"-iIXZ"
    #> Nothing to shift
    @test shift!(pauli"-XYZ", 0) == pauli"-XYZ"
    @test shift!(PauliStr(), 5) == PauliStr()
    @test_throws DomainError shift!(pauli"XYZ", -1)

    #> Strings of several words: a shift moves whole words and bits within a word, and the 
    #> operators pushed above site `n` leave no bit behind in the final word.
    ids = n->fill(symI, n)
    @test shift!(PauliStr(longList), 64) == PauliStr(vcat(ids(64), longList[1:66]))
    @test shift!(PauliStr(longList), 67) == PauliStr(vcat(ids(67), longList[1:63]))
    @test shift!(PauliStr(longList), 67, false) == PauliStr(vcat(longList[68:130], ids(67)))
    @test shift!(PauliStr(longList), 130) == shift!(PauliStr(longList), 1000, false) == 
          PauliStr(130)
end

@testset "paste!" begin
    #> By default, the sites `srcRange` of `src` (all of them when unspecified) overwrite 
    #> those of `dst` from site `dstStart` upward; the part of the selection that falls 
    #> above site `n` of `dst` is truncated.
    @test paste!(pauli"ZZZZZ", 2, pauli"XIY") == pauli"ZXIYZ"
    @test paste!(pauli"ZZZ", 2, pauli"XIY") == pauli"ZXI"
    @test paste!(pauli"ZZZZ", 2, pauli"XIY", 2:3) == pauli"ZIYZ"
    #> `lowToHigh=false`: site `last(srcRange)` of `src` lands on site `dstStart`, and the 
    #> rest of the selection extends downward; its part that falls below site 1 of `dst` 
    #> is truncated.
    @test paste!(pauli"ZZZZZ", 4, pauli"XIY", lowToHigh=false) == pauli"ZXIYZ"
    @test paste!(pauli"ZZZ", 1, pauli"XIY", lowToHigh=false) == pauli"YZZ"
    @test paste!(pauli"ZZZZ", 3, pauli"XIY", 1:2, lowToHigh=false) == pauli"ZXIZ"
    #> `dst` is mutated in place and keeps its phase, while `src` is left intact.
    dst, src = pauli"-ZZZ", pauli"iXY"
    @test paste!(dst, 2, src) === dst && dst == pauli"-ZXY" && src == pauli"iXY"
    #> An empty `srcRange` selects nothing, wherever it lies.
    @test paste!(pauli"ZZ", 1, pauli"XX", 3:2) == paste!(pauli"ZZ", 2, PauliStr()) == 
          pauli"ZZ"
    #> Input validation
    @test_throws DomainError paste!(pauli"ZZ", 0, pauli"X")
    @test_throws DomainError paste!(pauli"ZZ", 3, pauli"X")
    @test_throws DomainError paste!(PauliStr(), 1, pauli"X")
    @test_throws DomainError paste!(pauli"ZZ", 1, pauli"X", 1:2)
    @test_throws DomainError paste!(pauli"ZZ", 1, pauli"X", 0:1)

    #> Strings of several words: the pasted window can start and end inside a word, cover 
    #> whole words in between, and be read from other words of `src` than the ones it is 
    #> written to in `dst`, with or without an offset of bits.
    src = PauliStr(longList)
    zs = n->fill(symZ, n)
    @test paste!(PauliStr(200, symZ), 65, src) == PauliStr(vcat(zs(64), longList, zs(6)))
    @test paste!(PauliStr(140, symZ), 6, src) == PauliStr(vcat(zs(5), longList, zs(5)))
    @test paste!(PauliStr(66, symZ), 1, src, 65:130) == PauliStr(longList[65:130])
    @test paste!(PauliStr(10, symZ), 3, src, 62:67) == 
          PauliStr(vcat(zs(2), longList[62:67], zs(2)))
    @test paste!(PauliStr(100, symZ), 70, src, lowToHigh=false) == 
          PauliStr(vcat(longList[61:130], zs(30)))
    #> `dst` can be `src` itself
    @test paste!(src, 6, src) == PauliStr(vcat(longList[1:5], longList[1:125]))
end

@testset "stamp!" begin
    #> By default, a window of `nSite` sites (one when unspecified) is overwritten from 
    #> site `startSite` upward; its part that falls above site `n` of `dst` is truncated.
    @test stamp!(pauli"ZZZ", 2, symX) == pauli"ZXZ"
    @test stamp!(pauli"ZZZZ", 2, symY, 2) == pauli"ZYYZ"
    @test stamp!(pauli"ZZZ", 2, symX, 3) == pauli"ZXX"
    @test stamp!(pauli"XYZ", 1, symI, 2) == pauli"IIZ"
    #> `lowToHigh=false`: the window ends at site `startSite` and extends downward; its 
    #> part that falls below site 1 is truncated.
    @test stamp!(pauli"ZZZZ", 3, symX, 2, lowToHigh=false) == pauli"ZXXZ"
    @test stamp!(pauli"ZZZ", 1, symX, 3, lowToHigh=false) == pauli"XZZ"
    #> `dst` is mutated in place and keeps its phase; an empty window changes nothing.
    s = pauli"-iZZ"
    @test stamp!(s, 1, symY) === s && s == pauli"-iYZ"
    @test stamp!(pauli"XYZ", 2, symI, 0) == pauli"XYZ"
    #> Input validation
    @test_throws DomainError stamp!(pauli"ZZ", 0, symX)
    @test_throws DomainError stamp!(pauli"ZZ", 3, symX)
    @test_throws DomainError stamp!(PauliStr(), 1, symX)
    @test_throws DomainError stamp!(pauli"ZZ", 1, symX, -1)

    #> Strings of several words: a window across a word boundary, one that ends at the 
    #> last site of a word, and one that covers every word.
    window = (lo, hi, op)->PauliStr([lo <= k <= hi ? op : symZ for k in 1:130])
    @test stamp!(PauliStr(130, symZ), 60, symY, 10) == window(60, 69, symY)
    @test stamp!(PauliStr(130, symZ), 64, symX, 3, lowToHigh=false) == window(62, 64, symX)
    @test stamp!(PauliStr(130, symZ), 1, symX, 130) == PauliStr(130, symX)
end

@testset "reframe" begin
    h = PauliSum([pauli"XZ", pauli"YIX"], [1, 2]) #> Stored on 3 sites as `XZI` and `YIX`
    #> Every string is rebuilt within a frame of `nSite` sites, aligned with the frame at 
    #> the low end by default and at the high end when `lowToHigh=false`: the uncovered 
    #> sites of a wider frame carry `filler` (the identity when unspecified), and a 
    #> narrower frame crops the string.
    @test reframe(h, 5) == PauliSum([pauli"XZIII", pauli"YIXII"], [1, 2])
    @test reframe(h, 5, lowToHigh=false) == PauliSum([pauli"IIXZI", pauli"IIYIX"], [1, 2])
    @test reframe(h, 4, filler=symY) == PauliSum([pauli"XZIY", pauli"YIXY"], [1, 2])
    @test reframe(h, 4, filler=symY, lowToHigh=false) == 
          PauliSum([pauli"YXZI", pauli"YYIX"], [1, 2])
    @test reframe(h, 2) == PauliSum([pauli"XZ", pauli"YI"], [1, 2])
    @test reframe(h, 2, lowToHigh=false) == PauliSum([pauli"ZI", pauli"IX"], [1, 2])
    @test countSites(reframe(h, 70)) == 70 && reframe(reframe(h, 70), 3) == h
    #> The frame defaults to the site count of `h`, and the result shares no data with `h`.
    r = reframe(h)
    @test r == h && r.coeff !== h.coeff && first(r.str) !== first(h.str)
    #> The reframed strings are collected by the `PauliSum` constructor: the equal ones are 
    #> merged, and a zero coefficient is only retained when `simplification=false`.
    @test reframe(h, 0) == PauliSum([PauliStr()], [3])
    hc = PauliSum([pauli"XZ", pauli"XY"], [1, -1])
    @test reframe(hc, 1) == PauliSum(Int)
    @test reframe(hc, 1, simplification=false) == PauliSum([pauli"X"], [0], false)
    #> The phase of a stored string is transferred, and so ends up in the coefficient.
    hp = PauliSum([pauli"XZ"], [2])
    indexTerm(hp, 1, false).first.phase = posImg
    @test reframe(hp, 3) == PauliSum([pauli"XZI"], [2im])
    #> The zero operator stays as it is; `nSite` must be non-negative
    @test reframe(PauliSum(Float64), 4) == PauliSum(Float64)
    @test_throws DomainError reframe(h, -1)

    #> Strings of several words
    hl = PauliSum([PauliStr(longList)], [1.5])
    @test reframe(hl, 66, lowToHigh=false) == PauliSum([PauliStr(longList[65:130])], [1.5])
    @test reframe(hl, 140, filler=symY) == 
          PauliSum([PauliStr(vcat(longList, fill(symY, 10)))], [1.5])
end

@testset "compareBits: shifted windows" begin
    #> `compareSites` passes the same window for both buffers, so the shifted-window paths 
    #>  of `compareBits` (`wordShift != 0`, `bitOffset != 0`) are reachable only directly
    w1  = UInt8[0b0110_1001, 0b1101_0010]
    w2  = UInt8[0x00, 0x40, 0x13] #> Bits 14:21 of `w2` equal bits 3:10 of `w1`
    w2f = UInt8[0x00, 0x40, 0x12] #> `w2` with bit 17 flipped
    @test  compareBits(w1, UInt8[0xd2], 9:16, 1:8) #> Word-aligned shift
    @test  compareBits(w1, w2, 3:10, 14:21)        #> Misaligned, in either direction
    @test  compareBits(w2, w1, 14:21, 3:10)
    @test !compareBits(w1, w2f, 3:10, 14:21)
    @test !compareBits(w2f, w1, 14:21, 3:10)
    @test !compareBits(w1, w2, 3:20, 14:31)        #> Bits 11:16 of `w1` vs. zeros past `w2`
    @test_throws ArgumentError compareBits(w1, w2, 1:4, 1:5)
end

@testset "compareSites" begin
    #> Explicit sites, with the phases ignored
    @test  compareSites(pauli"XZIY", pauli"XZIY")
    @test  compareSites(pauli"XZIY", pauli"-iXZIY")
    @test !compareSites(pauli"XZIY", pauli"YZIY") #> Site 1 differs
    @test !compareSites(pauli"XZIY", pauli"XZIX") #> Site 4 differs

    #> The window selects the compared sites
    @test  compareSites(pauli"XZIY", pauli"XZIX", finalSite=3)
    @test !compareSites(pauli"XZIY", pauli"XZIX", finalSite=4, startSite=4)
    @test  compareSites(pauli"XZIY", pauli"YZIY", finalSite=4, startSite=2)
    @test  compareSites(pauli"XZIY", pauli"YZIY", finalSite=2, startSite=3) #> Empty window

    #> Sites beyond the explicit site count carry the identity
    @test  compareSites(pauli"XX", pauli"XXII")
    @test  compareSites(pauli"XXII", pauli"XX")
    @test !compareSites(pauli"XX", pauli"XXIX")
    @test  compareSites(pauli"XX", pauli"XXIX", finalSite=3)
    @test !compareSites(pauli"X", pauli"Z", finalSite=1000)
    @test  compareSites(pauli"X", pauli"Z", finalSite=1000, startSite=2)
    @test  compareSites(PauliStr(), PauliStr()) #> Default `finalSite` is 0: empty window
    @test  compareSites(PauliStr(), pauli"-")
    @test  compareSites(pauli"X", pauli"Z", finalSite=0)

    #> Window edges at the word boundary (site 64 | site 65), and buffers of unequal length
    a = stamp!(PauliStr(66), 64, symX)
    b = stamp!(PauliStr(66), 65, symX)
    @test  compareSites(a, b, finalSite=63)
    @test !compareSites(a, b, finalSite=64)
    @test !compareSites(a, b, finalSite=65, startSite=65)
    @test  compareSites(a, b, finalSite=66, startSite=66)
    @test  compareSites(a, PauliStr(a, 64)) #> Cropping keeps the `X` on site 64
    @test !compareSites(b, PauliStr(b, 64)) #> Cropping drops the `X` on site 65
    @test !compareSites(a, PauliStr())      #> Zero-site string: no word at all
    @test  compareSites(a, PauliStr(), finalSite=63)
    @test  compareSites(PauliStr(), b, finalSite=64)

    #> Input validation
    @test_throws DomainError compareSites(pauli"X", pauli"X", startSite=0)
    @test_throws DomainError compareSites(pauli"X", pauli"X", finalSite=1, startSite=-1)

    @testset "Cross-check with `indexSite`" begin
        syms = [symI, symZ, symX, symY]
        rng = MersenneTwister(1234)
        for _ in 1:3000
            a = PauliStr(PauliSym[rand(rng, syms) for _ in 1:rand(rng, 0:70)])
            #> `b`: a cropped or identity-padded copy of `a`, or an unrelated string
            b = if rand(rng, Bool)
                PauliStr(a, rand(rng, 0:70))
            else
                PauliStr(PauliSym[rand(rng, syms) for _ in 1:rand(rng, 0:70)])
            end
            #> Overwrite one site of `b` half of the time
            if !iszero(countSites(b)) && rand(rng, Bool)
                stamp!(b, rand(rng, 1:countSites(b)), rand(rng, syms))
            end

            iFinal = rand(rng, 0:72)
            iStart = rand(rng, 1:72)
            ref = all(indexSite(a, k, true) == indexSite(b, k, true) for k in iStart:iFinal)
            @test compareSites(a, b, finalSite=iFinal, startSite=iStart) == ref
        end
    end
end

@testset "Integer site indices and counts" begin
    #> Any `Integer` is accepted at the surface and converted to `Int` on entry
    @test PauliStr(UInt(3)) == PauliStr(3)
    @test PauliStr(pauli"XZ", UInt8(3)) == pauli"XZI"
    @test indexSite(pauli"XZ", UInt(2)) == symZ
    @test toString(pauli"XZ", Int8(2)) == "Z₂"
    @test indexTerm(PauliSum([pauli"X"], [1]), UInt(1)).first == pauli"X"
    @test shift!(pauli"XIII", UInt(1)) == pauli"IXII"
    @test paste!(pauli"XIII", UInt8(1), pauli"ZZ", UInt(1):UInt(2)) == pauli"ZZII"
    @test stamp!(pauli"XIII", UInt(2), symZ, Int8(2)) == pauli"XZZI"
    @test countSites(reframe(PauliSum([pauli"X"], [1]), UInt(4))) == 4
    @test  compareSites(pauli"XX", pauli"XXIX", finalSite=UInt(3), startSite=UInt8(2))
    @test !compareSites(pauli"XX", pauli"XXIX", finalSite=Int128(4))
    @test_throws DomainError PauliStr(typemax(UInt))
    @test_throws DomainError indexSite(pauli"XZ", typemax(UInt), true)
end

end
