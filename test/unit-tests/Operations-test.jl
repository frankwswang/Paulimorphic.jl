using Test
using Paulimorphic

@testset "Operations.jl" begin

@test string(pauli"") == "I"
@test string(pauli"I") == "I"
@test string(pauli"II") == "I"
@test string(pauli"IIX") == "+X₃"

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


@testset "getFrustrationInfo" begin

    # split the `nodes => edges` Pair; normalize edges to a Set (order-agnostic)
    gfg(args...; kw...) = (r = getFrustrationInfo(args...; kw...); (r.first, Set(r.second)))

    # edges must be strictly-upper-triangular, in range, and unique
    wellFormed(nodes, edges) =
        all(e -> 1 <= e[1] < e[2] <= length(nodes), edges) && allunique(collect(edges))

    # Single-site canonical (isless) order is  I < Z < X < Y.
    X, Y, Z, Ig = PauliStr([symX]), PauliStr([symY]), PauliStr([symZ]), PauliStr([symI])

    @testset "vector overload — no merge, input order preserved" begin
        # {X,Y,Z} pairwise anticommute ⇒ complete triangle, nodes in input order
        nodes, edges = gfg([X, Y, Z])
        @test nodes == [X, Y, Z]
        @test edges == Set([(1, 2), (1, 3), (2, 3)])

        # duplicates are NOT merged on the vector path; all commute ⇒ no edges
        nodes, edges = gfg([X, X, Ig])
        @test nodes == [X, X, Ig]
        @test isempty(edges)

        # singleton ⇒ no edges
        _, edges = gfg([X])
        @test isempty(edges)
    end

    @testset "PauliSum merge semantics into the graph" begin
        # two X's each below nodeThreshold individually, but their coeffs SUM above it,
        # so after merging a single X node survives (would vanish if merge were skipped)
        nodes, edges = gfg(PauliSum([0.06, 0.06, 1.0], [X, X, Z]); nodeThreshold = 0.1)
        @test nodes == [Z, X]                      # canonical order Z < X
        @test edges == Set([(1, 2)])               # Z, X anticommute

        # exact cancellation: X + (−X) merges to zero and is dropped, leaving only Z
        nodes, edges = gfg(PauliSum([1.0, -1.0, 1.0], [X, X, Z]); nodeThreshold = 0.0)
        @test nodes == [Z]
        @test isempty(edges)
    end

    @testset "node filtering & index alignment — front drop" begin
        # canonical [I, Z, X]; I (coeff 0.01) filtered ⇒ survivors [Z, X].
        # The buggy version mis-indexes the unfiltered prefix [I, Z] (I commutes),
        # so it finds no edge; the correct version sees Z–X anticommute.
        nodes, edges = gfg(PauliSum([0.01, 1.0, 1.0], [Ig, Z, X]); nodeThreshold = 0.1)
        @test nodes == [Z, X]
        @test edges == Set([(1, 2)])
        @test !isempty(edges)                      # trips on the old bug (would be empty)
    end

    @testset "node filtering & index alignment — middle drop" begin
        ZI, IZ = PauliStr([symZ, symI]), PauliStr([symI, symZ])
        XX, YY = PauliStr([symX, symX]), PauliStr([symY, symY])
        # canonical [ZI, IZ, XX, YY]; IZ (coeff 0.01) is the middle drop ⇒ [ZI, XX, YY].
        # ZI anticommutes with both XX and YY; XX, YY commute.
        nodes, edges = gfg(PauliSum([1.0, 0.01, 1.0, 1.0], [ZI, IZ, XX, YY]); nodeThreshold = 0.1)
        @test nodes == [ZI, XX, YY]
        @test edges == Set([(1, 2), (1, 3)])
        # the buggy version returns {(1,3),(2,3)} here — pin the exact correct set
        @test (2, 3) ∉ edges && (1, 2) ∈ edges
    end

    @testset "edge & node thresholds are strict (>)" begin
        # canonical [Z(0.5), X(1.0)]; |c_i·c_j| = 0.5 exactly at edgeThreshold ⇒ excluded
        _, atBound = gfg(PauliSum([1.0, 0.5], [X, Z]); nodeThreshold = 0.0, edgeThreshold = 0.5)
        @test isempty(atBound)
        _, below = gfg(PauliSum([1.0, 0.5], [X, Z]); nodeThreshold = 0.0, edgeThreshold = 0.4)
        @test below == Set([(1, 2)])

        # node coeff exactly at nodeThreshold is excluded ⇒ only X (coeff 1.0) survives
        nodes, _ = gfg(PauliSum([1.0, 0.5], [X, Z]); nodeThreshold = 0.5, edgeThreshold = 0.0)
        @test nodes == [X]
    end

    @testset "threshold default & empty result" begin
        # nodeThreshold defaults to edgeThreshold: setting only edgeThreshold drops the 0.5 node
        nodes, _ = gfg(PauliSum([0.5, 1.0], [X, Z]); edgeThreshold = 0.5)
        @test nodes == [Z]

        # every coeff below threshold ⇒ empty graph (zero survives no strict `> 0`)
        nodes, edges = gfg(PauliSum([0.0, 0.0], [X, Z]))
        @test isempty(nodes) && isempty(edges)
    end

    @testset "complex coefficients (abs path)" begin
        # canonical [Z(1+0im), X(0+1im)]; both |·| = 1 > 0.5 kept; |1·i| = 1 > 0 ⇒ edge
        nodes, edges = gfg(PauliSum([1im, 1.0 + 0im], [X, Z]);
                           nodeThreshold = 0.5, edgeThreshold = 0.0)
        @test nodes == [Z, X]
        @test edges == Set([(1, 2)])
    end

    @testset "multi-site commutation feeds the graph (vector path)" begin
        XX = PauliStr([symX, symX])   # X₁X₂
        ZZ = PauliStr([symZ, symZ])   # Z₁Z₂ — commutes with XX (two anticommutes cancel)
        YI = PauliStr([symY, symI])   # Y₁
        _, edges = gfg([XX, ZZ, YI])
        @test edges == Set([(1, 3), (2, 3)])   # XX–ZZ commute; both anticommute with Y₁
    end

    @testset "overloads agree on a PauliSum's own (canonical) strings" begin
        # With no filtering, the PauliSum overload equals the vector overload applied to
        # the sum's stored strings — same canonical order, same edges.
        ham = PauliSum([1.0, 1.0, 1.0, 1.0], [X, Y, Z, Ig])
        nSum, eSum = gfg(ham; nodeThreshold = -1.0, edgeThreshold = -1.0)
        nVec, eVec = gfg(ham.str)              # Memory{PauliStr} is an AbstractVector
        @test nSum == nVec
        @test eSum == eVec
    end

    @testset "structural invariants" begin
        nodes, edges = gfg([X, Y, Z])
        @test wellFormed(nodes, edges)
        nodes, edges = gfg(PauliSum([0.01, 1.0, 1.0], [Ig, X, Z]); nodeThreshold = 0.1)
        @test wellFormed(nodes, edges)
    end
end

end