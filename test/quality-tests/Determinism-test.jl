using Test
using Paulimorphic
using Paulimorphic: evalMagnitude, combine

#> Bit-reproducibility checks of the determinism guarantees of Paulimorphic: for identical 
#> labeled inputs, the functions below must return identical outputs on every platform 
#> (operating system, CPU, hardware FMA availability, word size) and, where documented, 
#> independently of the order in which the inputs are listed or stored. Every expected 
#> value is a fixed constant obtained from an independent re-implementation of the same 
#> formulas evaluated with exact IEEE-754 arithmetic (one correctly rounded operation at a 
#> time, no contraction), not from a run on any particular machine. Hence, a failure on one 
#> machine points at a platform-dependent code path (e.g., `abs(::Complex)`, whose last bit 
#> depends on `Core.Intrinsics.have_fma`), a storage-order dependence, or a changed 
#> accumulation order. The constants are tied to the implemented algorithms: a deliberate 
#> change of an algorithm must regenerate them.

@testset "Determinism Test" begin

#> Both branches of `Base.Math._hypot` (Julia 1.12, no-rescaling regime), which 
#> `abs(::ComplexF64)` dispatches to by `Core.Intrinsics.have_fma`. Since `fma` is exact on 
#> every machine (software fallback) and `a*b + c` is never contracted, both values can be 
#> computed anywhere, whereas `abs(c)` only equals the branch selected by the local 
#> hardware.
function hypotBranch(x::Float64, y::Float64, useFMA::Bool)
    ax, ay = abs(x), abs(y)
    ay > ax && ((ax, ay) = (ay, ax))
    ay <= ax * sqrt(eps(Float64) / 2) && return ax
    if useFMA
        h = sqrt(fma(ax, ax, ay * ay))
        hsq = h * h
        axsq = ax * ax
        h -= (fma(-ay, ay, hsq - axsq) + fma(h, h, -hsq) - fma(ax, ax, -axsq)) / (2h)
    else
        h = sqrt(ax * ax + ay * ay)
        if h <= 2ay
            delta = h - ay
            h -= (delta * (delta - 2 * (ax - ay)) + ax * (2delta - ax)) / (2h)
        else
            delta = h - ax
            h -= (delta * delta + (ay * (4delta - ay) + 2delta * (ax - 2ay))) / (2h)
        end
    end
    h
end

#> The same labeled graph with another internal layout of its adjacency sets: reversed 
#> edge insertion order, then a rehash of every set
function relayout(g::StrictGraph)
    h = StrictGraph(countVertices(g), reverse(listEdges(g)))
    foreach(s -> sizehint!(s, 4 * length(s) + 16), h.adjacency)
    h
end

#> Generic coefficients whose `abs` differs between FMA and non-FMA hardware, one per 
#> correction branch of `evalMagnitude`
cx = 0.23665594309283294 + 0.03906366194959983im
cy = 0.3050071890346679 + 0.30436799247744367im
cxMagnitude = 0.2398583020995329  #> `evalMagnitude(cx)`; `abs(cx)` is 1 ulp larger with FMA
cyMagnitude = 0.43089356018346203 #> `evalMagnitude(cy)`; `abs(cy)` is 1 ulp larger with FMA

@testset "evalMagnitude is bitwise reproducible" begin
    for (c, golden, fmaValue) in ((cx, cxMagnitude, 0.23985830209953293), 
                                  (cy, cyMagnitude, 0.4308935601834621))
        x, y = reim(c)
        @test hypotBranch(x, y, true) == fmaValue != hypotBranch(x, y, false) == golden
        @test evalMagnitude(c) === golden
        #> Symmetries of the formula: identical bits for every sign flip and swap
        @test evalMagnitude(complex(-x, y)) === evalMagnitude(complex(x, -y)) === golden
        @test evalMagnitude(complex(y, x)) === evalMagnitude(complex(-y, -x)) === golden
    end

    #> Rescaled regimes (no spurious overflow or underflow of the squared components)
    @test evalMagnitude(1.5e300 + 0.7e300im) === 1.655294535724685e300
    @test evalMagnitude(3e-300 + 2e-300im) === 3.6055512754639895e-300

    #> Single precision follows the same formula in `Float32` arithmetic
    @test evalMagnitude(0.2366559f0 + 0.03906366f0im) === 0.23985827f0
    @test evalMagnitude(0.41982356f0 + 0.3570103f0im) === 0.5510973f0
end

@testset "PauliSum: canonical term order" begin
    #> Strings are ordered by site count, weight, highest differing site, and phase, and 
    #> every phase is absorbed into the coefficient
    strs = [pauli"IX", pauli"-ZI", pauli"iYI", pauli"II", pauli"XI", pauli"IZ"]
    h = PauliSum(strs, 1.0)
    golden = [pauli"II" => 1.0 + 0.0im, pauli"ZI" => -1.0 + 0.0im, 
              pauli"XI" => 1.0 + 0.0im, pauli"YI" => 0.0 + 1.0im, 
              pauli"IZ" => 1.0 + 0.0im, pauli"IX" => 1.0 + 0.0im]
    @test listTerms(h) == golden
    @test isCanonicalized(h, checkDuplicateTerm=true, checkZeroCoeffTerm=true)
    @test PauliSum(reverse(strs), 1.0) == PauliSum(strs[[4, 1, 6, 3, 2, 5]], 1.0) == h
    h2 = canonicalize!(deepcopy(h))
    @test h2.str == h.str && h2.coeff == h.coeff

    #> Equal strings are ordered by `(evalMagnitude(c), real(c), imag(c))`: the two 
    #> coefficients below tie in magnitude, so the real parts decide, whereas `abs(cx)` on 
    #> FMA hardware exceeds `abs(c2)` and would put `c2` first
    c2 = complex(cxMagnitude)
    h3 = PauliSum([pauli"X", pauli"X"], [c2, cx], false)
    @test h3.coeff == [cx, c2] && isCanonicalized(h3)
end

@testset "PauliSum: permutation-invariant merging of equal strings" begin
    #> With a wide dynamic range, plain left-to-right summation depends on the order of 
    #> the terms; the constructor accumulates every run of equal strings in the canonical 
    #> order with Neumaier summation, so the result depends only on the multiset of terms 
    #> and has the same bits everywhere. (The exact sum is 1 + 2^-53 + 2^-60; the pinned 
    #> value is what the compensated sum in the canonical order returns, not the correctly 
    #> rounded sum, which Neumaier summation does not guarantee for such a dynamic range.)
    wide = [2.0^54, -2.0^54, 1.0, 2.0^-53, 2.0^-60]
    @test foldl(+, wide) != foldl(+, reverse(wide))
    golden = PauliSum([pauli"X"], [1.0])
    for perm in ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 3, 4, 2, 5], [4, 5, 1, 2, 3])
        @test PauliSum(fill(pauli"X", 5), wide[perm]) == golden
    end

    #> Complex coefficients: both components are compensated, and the canonical order is 
    #> decided by `evalMagnitude` and then the real and imaginary parts
    cWide = [2.0^54 + 1.0im, -2.0^54 + 2.0^-53*im, 1.0 - 1.0im, 2.0^-53 + 2.0^-60*im, 
             2.0^-60 + 0.0im]
    goldenC = PauliSum([pauli"X"], [complex(1.0, 2.0^-53 + 2.0^-60)])
    for perm in ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 1, 5, 2, 4])
        @test PauliSum(fill(pauli"X", 5), cWide[perm]) == goldenC
    end

    #> `sumCoeffs` accumulates in the canonical term order, which the input order cannot 
    #> disturb (the strings below are listed in their canonical order)
    strs = [pauli"ZI", pauli"XI", pauli"YI", pauli"IZ", pauli"IX"]
    hw = PauliSum(strs, wide)
    @test hw == PauliSum(reverse(strs), reverse(wide)) && hw.coeff == wide
    @test sumCoeffs(hw) === 1.0 + 2.0^-52 + 0.0im
end

@testset "n-ary `+` and `combine`: one merge, invariant under permutation" begin
    s = pauli"X"
    h1, h2, h3 = PauliSum([s], [1.0]), PauliSum([s], [1e-17]), PauliSum([s], [-1.0])

    #> Explicitly associated binary chains round after every step (deterministic, but the 
    #> association decides the result)
    @test countTerms((h1 + h2) + h3) == 0
    @test countTerms((h1 + h3) + h2) == 1

    #> A chain of three or more terms is merged once, in the canonical order
    golden = PauliSum([s], [1e-17])
    @test h1 + h2 + h3 == h3 + h2 + h1 == h2 + h1 + h3 == golden
    @test (h1 + h2 + h3).coeff == [1.0e-17 + 0.0im]
    @test combine([h1, h2, h3]) == combine([h3, h1, h2]) == golden
    @test combine(Float64, [h2, h3, h1]) == golden
    @test s + h2 + h3 == h3 + s + h2 == golden #> Mixed `PauliStr`/`PauliSum` chains
end

@testset "encodeElecHam: one compensated accumulation across all fragments" begin
    #> Same scenario as in the `Molecular.jl` unit tests: the identity coefficient collects 
    #> 1 and 2^-53 from the diagonal of the spin-1 one-body matrix and another 2^-53 from 
    #> the cross two-body tensor. Any scheme that rounds the spin-1 fragment on its own 
    #> returns 1 (a tie, rounded to even) and can never recover the lost 2^-53, even with a 
    #> compensated merge of the finalized fragments; one compensated accumulation over all 
    #> contributions returns the exact value 1 + 2^-52.
    enc3 = toDiracEnc(genJordanWignerEnc(3))
    hA = [2.0 0.0; 0.0 2.0^-52]
    gA = zeros(2, 2, 2, 2)
    hB = zeros(1, 1)
    gB = zeros(1, 1, 1, 1)
    gX = zeros(2, 2, 1, 1)
    gX[1, 1, 1, 1] = 2.0^-51
    golden = PauliSum([pauli"III", pauli"ZII", pauli"IZI", pauli"IIZ", pauli"ZIZ"], 
                      [1.0 + 2.0^-52, -1.0, -2.0^-53, -2.0^-53, 2.0^-53])
    @test encodeElecHam(NormalOrder(), enc3, (hA, gA), (hB, gB), gX) == golden
    @test encodeElecHam(PairedOrder(), enc3, (hA, gA), (hB, gB), gX) == golden
end

@testset "magnitude thresholds are evaluated by evalMagnitude" begin
    #> `cxMagnitude` is exactly on the threshold under `evalMagnitude`, whereas `abs(cx)` 
    #> on FMA hardware exceeds it
    h = PauliSum([pauli"X"], [cx])
    @test countTerms(curtail(h, cxMagnitude)) == 0
    @test countTerms(curtail(h, cxMagnitude; inclusive=true)) == 1
    @test countTerms(curtail(h, prevfloat(cxMagnitude))) == 1

    #> The frustration edge weight `1.0 * evalMagnitude(cx)` lands on the threshold as well
    h2 = PauliSum([pauli"X", pauli"Z"], [cx, 1.0])
    vertices, edges = getFrustrationInfo(h2, cxMagnitude)
    @test vertices == [pauli"Z", pauli"X"] && isempty(edges)
    vertices, edges = getFrustrationInfo(h2, cxMagnitude; inclusive=true)
    @test vertices == [pauli"Z", pauli"X"] && edges == [(1, 2)]
end

@testset "genRootGraph: fixed labeled outputs" begin
    #> Each case pins the labeled root graph and the vertex-edge correspondence of one 
    #> line graph, so that the outputs cannot silently depend on the iteration order of 
    #> the adjacency sets (which varies with the hashing of the vertex type, the word size, 
    #> and the insertion history). The inputs cover every initial branch of the labeling 
    #> (zero, one, two adjacent, two non-adjacent, and three shared neighbors of the basic 
    #> nodes) and the documented choices for the Whitney exceptions.
    cycle6 = StrictGraph(6, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)])
    triangle = StrictGraph(3, [(1, 2), (1, 3), (2, 3)])
    k4 = StrictGraph(4, [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
    octahedron = StrictGraph(6, [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 6), 
                                 (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)])
    lK5 = StrictGraph(10, [(1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (1, 9), (2, 4), 
                           (2, 5), (2, 6), (2, 8), (2, 10), (3, 5), (3, 6), (3, 7), (3, 8), 
                           (3, 9), (3, 10), (4, 7), (4, 8), (4, 9), (4, 10), (5, 6), 
                           (5, 9), (5, 10), (6, 7), (6, 8), (7, 8), (7, 9), (8, 10), 
                           (9, 10)])
    lMixed = StrictGraph(8, [(1, 3), (1, 5), (1, 8), (2, 4), (2, 6), (2, 7), (3, 7), 
                             (3, 8), (4, 6), (5, 8), (6, 7)])
    k5Edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), 
               (4, 5)]
    cases = (
        #> C6 = L(C6): no shared neighbor of the basic nodes
        (cycle6, [(1, 2), (1, 4), (2, 3), (3, 5), (4, 6), (5, 6)], 
                 [(1, 2), (2, 3), (3, 5), (5, 6), (4, 6), (1, 4)]), 
        #> K3: one shared neighbor; the triangle (not the claw) is returned
        (triangle, [(1, 2), (1, 3), (2, 3)], [(1, 2), (2, 3), (1, 3)]), 
        #> K4 = L(K_{1,4}): two adjacent shared neighbors
        (k4, [(1, 2), (2, 3), (2, 4), (2, 5)], [(1, 2), (2, 3), (2, 4), (2, 5)]), 
        #> Octahedron = L(K4): two non-adjacent shared neighbors; one of the two valid 
        #> correspondences (its twin vertices 1 and 6 take the edges (1, 2) and (3, 4))
        (octahedron, [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], 
                     [(1, 2), (2, 3), (1, 3), (2, 4), (1, 4), (3, 4)]), 
        #> A relabeled L(K5): three shared neighbors, one of them a cross node
        (lK5, k5Edges, [(1, 2), (2, 3), (4, 5), (1, 3), (2, 4), (2, 5), (1, 5), (3, 5), 
                        (1, 4), (3, 4)]), 
        #> A relabeled line graph of a 7-vertex root with a triangle, a bridge, and a cycle
        (lMixed, [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 7)], 
                 [(5, 6), (2, 3), (4, 5), (1, 2), (6, 7), (1, 3), (3, 4), (5, 7)]), 
    )
    for (g, rootEdges, labels) in cases
        n = countVertices(g)
        cache = fill((0, 0), n)
        ok, root = genRootGraph(g, cache)
        @test ok
        @test listEdges(root) == rootEdges
        @test cache == labels
        @test countVertices(root) == maximum(maximum, rootEdges)
        #> Another storage layout and another vertex type of the same labeled graph
        cache2 = fill((0, 0), n)
        ok2, root2 = genRootGraph(relayout(g), cache2)
        @test ok2 && root2 == root && cache2 == labels
        cache16 = fill((zero(Int16), zero(Int16)), n)
        ok16, root16 = genRootGraph(StrictGraph(Int16, g), cache16)
        @test ok16 && root16 == root && cache16 == labels
    end

    #> A non-line graph is returned as is, with the buffer untouched
    claw = StrictGraph(4, [(1, 2), (1, 3), (1, 4)])
    cacheClaw = fill((0, 0), 4)
    ok, returned = genRootGraph(claw, cacheClaw)
    @test !ok && returned === claw && all(==((0, 0)), cacheClaw)
end

end
