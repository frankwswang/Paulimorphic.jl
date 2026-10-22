# Examples

## Transverse-field Ising model

The following code builds the transverse-field Ising Hamiltonian on an open chain of `n` sites,

$$\hat{H} = -J \sum_{i=1}^{n-1} \hat{Z}_i \hat{Z}_{i+1} - h \sum_{i=1}^{n} \hat{X}_i,$$

as a `PauliSum`, and then inspects a few of its properties.

```jldoctest; output = false
using Paulimorphic

n, J, h = 4, 1.0, 0.5

zz = [stamp!(PauliStr(n), i, symZ, 2) for i in 1:n-1] # ZᵢZᵢ₊₁: stamp Z onto sites i, i+1
xs = [stamp!(PauliStr(n), i, symX)    for i in 1:n  ] # Xᵢ:     stamp X onto site i

H = PauliSum([zz; xs], [fill(-J, n-1); fill(-h, n)])  # Canonical-form `PauliSum`

countTerms(H), countSites(H)  # `(7, 4)`
isHermitian(H)                # `true`

# Small operators can also be written out with the `pauli"..."` literal and assembled with 
# the overloaded `*` and `+`. Since every `PauliSum` is kept in canonical form, the two 
# constructions compare equal:
H2 = (-J) * pauli"ZZII" + (-J) * pauli"IZZI" + (-J) * pauli"IIZZ" + 
     (-h) * pauli"XIII" + (-h) * pauli"IXII" + (-h) * pauli"IIXI" + (-h) * pauli"IIIX"
H == H2  # `true`

# `H` commutes with the global spin flip X₁X₂X₃X₄:
checkCommute(H, pauli"XXXX")  # `true`

# The frustration graph of `H` has one edge for each pair of anticommuting terms:
terms, edges = getFrustrationInfo(H)  # `edges` lists index pairs into `terms`
length(edges)                         # `6`

# output

6
```
