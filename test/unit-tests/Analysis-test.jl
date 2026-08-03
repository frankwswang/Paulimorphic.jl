using Test
using Paulimorphic

@testset "Analysis.jl" begin

@testset "getFrustrationInfo" begin
    #> Edges must be strictly-upper-triangular, in range, and unique
    wellFormed(nodes, edges) =
    all(e -> 1 <= e[1] < e[2] <= length(nodes), edges) && allunique(collect(edges))

    #> Single-site canonical (isless) order is  I < Z < X < Y.
    X, Y, Z, Id = PauliStr([symX]), PauliStr([symY]), PauliStr([symZ]), PauliStr([symI])

    @testset "vector overload — no merge, input order preserved" begin
        #> {X, Y, Z} pairwise anticommute -> complete triangle, nodes in input order
        nodes, edges = getFrustrationInfo([X, Y, Z])
        @test nodes == [X, Y, Z]
        @test edges == [(1, 2), (1, 3), (2, 3)]

        #> Duplicates are NOT merged on the vector path; all commute -> no edges
        nodes, edges = getFrustrationInfo([X, X, Id])
        @test nodes == [X, X, Id]
        @test isempty(edges)

        #> Singleton -> no edges
        _, edges = getFrustrationInfo([X])
        @test isempty(edges)
    end

    @testset "PauliSum merge semantics into the graph" begin
        #> Two X's each below threshold individually, but their coeffs SUM above it,
        #> so after merging a single X node survives (would vanish if merge were skipped)
        h = PauliSum([X, X, Z], [0.06, 0.06, 1.0])
        nodes, edges = getFrustrationInfo(curtail(h, 0.1))
        @test nodes == [Z, X]   #>> canonical order Z < X
        @test edges == [(1, 2)] #>> Z, X anticommute

        #> Exact cancellation: X + (−X) merges to zero and is dropped, leaving only Z
        nodes, edges = PauliSum([X, X, Z], [1.0, -1.0, 1.0]) |> getFrustrationInfo
        @test nodes == [Z]
        @test isempty(edges)
    end

    @testset "node filtering & index alignment — front drop" begin
        #> [Id, Z, X] with I (coeff 0.01) filtered -> [Z, X].
        h = PauliSum([Id, Z, X], [0.01, 1.0, 1.0])
        nodes, edges = getFrustrationInfo(curtail(h, 0.1))
        @test nodes == [Z, X]
        @test edges == [(1, 2)]
    end

    @testset "node filtering & index alignment — middle drop" begin
        ZI, IZ = PauliStr([symZ, symI]), PauliStr([symI, symZ])
        XX, YY = PauliStr([symX, symX]), PauliStr([symY, symY])
        #> [ZI, IZ, XX, YY] with IZ (coeff 0.01) filtered -> [ZI, XX, YY].
        h = PauliSum([ZI, IZ, XX, YY], [1.0, 0.01, 1.0, 1.0])
        nodes, edges = getFrustrationInfo(curtail(h, 0.1))
        @test nodes == [ZI, XX, YY]
        @test edges == [(1, 2), (1, 3)]
        #> the buggy version returns {(1,3),(2,3)} here — pin the exact correct set
        @test (2, 3) ∉ edges && (1, 2) ∈ edges
    end

    @testset "edge thresholds are strict (>) by default" begin
        #> canonical [Z(0.5), X(1.0)]; |c_i·c_j| = 0.5 exactly at edgeThreshold -> excluded
        _, atBound = getFrustrationInfo(PauliSum([X, Z], [1.0, 0.5]), 0.5)
        @test isempty(atBound)
        _, below = getFrustrationInfo(PauliSum([X, Z], [1.0, 0.5]), 0.4)
        @test below == [(1, 2)]

        #> every coeff below threshold -> empty graph (zero survives no strict `> 0`)
        nodes, edges = getFrustrationInfo(PauliSum([X, Z], [0.0, 0.0]))
        @test isempty(nodes) && isempty(edges)
    end

    @testset "complex coefficients (abs path)" begin
        #> canonical [Z(1+0im), X(0+1im)]; both |·| = 1 > 0.5 kept; |1·i| = 1 > 0 -> edge
        h = PauliSum([X, Z], [1im, 1.0 + 0im])
        nodes, edges = getFrustrationInfo(curtail(h, 0.5))
        @test nodes == [Z, X]
        @test edges == [(1, 2)]
    end

    @testset "multi-site commutation feeds the graph (vector path)" begin
        XX = PauliStr([symX, symX])   #> X₁X₂
        ZZ = PauliStr([symZ, symZ])   #> Z₁Z₂ — commutes with XX (two anticommutes cancel)
        YI = PauliStr([symY, symI])   #> Y₁
        _, edges = getFrustrationInfo([XX, ZZ, YI])
        @test edges == [(1, 3), (2, 3)]   #> XX–ZZ commute; both anticommute with Y₁
    end

    @testset "structural invariants" begin
        nodes, edges = getFrustrationInfo([X, Y, Z])
        @test wellFormed(nodes, edges)
        h = PauliSum([Id, X, Z], [0.01, 1.0, 1.0])
        nodes, edges = getFrustrationInfo(curtail(h, 0.5))
        @test wellFormed(nodes, edges)
    end
end

end