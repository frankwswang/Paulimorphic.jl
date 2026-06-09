using Test
using Paulimorphic

@testset "Operations.jl" begin

@test string(pauli"") == "I"
@test string(pauli"I") == "I"
@test string(pauli"II") == "I"
@test string(pauli"IIX") == "X₃"

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
sympl(a, b) = sum(a.zStr .* b.xStr) + sum(a.xStr .* b.zStr)
bl2 = true
for a in twoSite, b in twoSite
    bl2 *= checkCommute(a, b) == iseven(sympl(a, b))
    bl2 *= checkAntiCom(a, b) == isodd(sympl(a, b))
end
@test bl2

end