using Test
using Paulimorphic

@testset "Fermionic.jl" begin

@testset "checkMajEncoding" begin

#> Valid encodings
@test checkMajEncoding([pauli"X", pauli"Y"])       #>> Smallest case: one fermionic mode
@test checkMajEncoding([pauli"X", pauli"Y"], true) #>> `explicitError=true` must not throw

#>> Jordan–Wigner Majoranas for 3 modes: (Z^(i-1)) X|Y (I^(n-i))
jwStrs = [pauli"XII", pauli"YII", pauli"ZXI", pauli"ZYI", pauli"ZZX", pauli"ZZY"]
@test checkMajEncoding(jwStrs)
@test checkMajEncoding(@view jwStrs[begin:end]) #>> Generic `AbstractVector` input

#>> A negative real phase is still Hermitian
@test checkMajEncoding([PauliStr(pauli"X", 1, Paulimorphic.negRea), pauli"Y"])

#> Term-count condition
@test !checkMajEncoding(PauliStr[])       #>> Empty input
@test !checkMajEncoding([pauli"X"])       #>> Odd number of terms
@test !checkMajEncoding(jwStrs[1:3])      #>> Odd subset of a valid encoding
@test_throws ArgumentError checkMajEncoding(PauliStr[], true)
@test_throws ArgumentError checkMajEncoding([pauli"X"], true)

#> Uniform-site-count condition
#>> `pauli"X"` and `pauli"YI"` anticommute under implicit-identity semantics, but their
#>>> explicit widths (`.n`) differ, so the encoding check must reject the pair
@test !checkMajEncoding([pauli"X", pauli"YI"])
@test_throws ArgumentError checkMajEncoding([pauli"X", pauli"YI"], true)

#> Hermiticity condition
@test !checkMajEncoding([pauli"X", PauliStr(pauli"Y", 1, Paulimorphic.posImg)])
@test !checkMajEncoding([PauliStr(pauli"X", 1, Paulimorphic.negImg), pauli"Y"])
@test_throws ArgumentError begin
    checkMajEncoding([pauli"X", PauliStr(pauli"Y", 1, Paulimorphic.posImg)], true)
end

#> Pairwise-anticommutation condition
@test !checkMajEncoding([pauli"XI", pauli"IX"]) #>> Disjoint supports commute
@test !checkMajEncoding([pauli"X", pauli"X"])   #>> Duplicate terms commute
@test !checkMajEncoding([pauli"I", pauli"X"])   #>> Identity commutes with everything
@test_throws ArgumentError checkMajEncoding([pauli"XI", pauli"IX"], true)

#>> Only a late pair fails: [XI, YI, ZX, IZ] is valid except that XI and IZ commute,
#>>> so the pair loop must reach (1, 4) instead of stopping early
@test !checkMajEncoding([pauli"XI", pauli"YI", pauli"ZX", pauli"IZ"])

end

end