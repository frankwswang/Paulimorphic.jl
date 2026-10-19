# Paulimorphic.jl

[**Paulimorphic**](https://github.com/frankwswang/Paulimorphic.jl) is a Julia software package for **constructing**, **transforming**, and **analyzing** Pauli operators. In particular, Pauli strings are stored in a bit-packed symplectic representation together with their group phases, and their linear combinations are kept in a canonical form. Based on this core design, **Paulimorphic** aims to serve as a robust and performant toolkit for algebraic manipulation and structural analysis on linear operators in the qubit basis.

## Features

* **Efficient and deterministic representation** of Pauli strings (`PauliStr`) and Pauli sums (`PauliSum`).
* **Operator algebra**: operator summation, scalar–operator and operator–operator multiplication, Hermitian adjoints, commutation and anticommutation evaluations, etc.
* **Multi-level data manipulation**: shifting and pasting single-site operators; reframing and truncating Pauli sums, etc.
* **Frustration-graph analysis**: constructing the anticommutation graph of operators, backed by lightweight graph-theory analysis functionalities, such as connected components, breadth-first search, isomorphism detection, and root-graph reconstruction.
* **Fermion-to-qubit encodings**: Jordan–Wigner, parity, and Bravyi–Kitaev Majorana encodings, conversion between Majorana and Dirac forms, and validity checkers for both.
* **Molecular electronic Hamiltonians**: encoding from one- and two-body molecular integral tensors with spin-sectored encodings, selectable operator ordering (`NormalOrder` or `PairedOrder`), and validation of integral symmetries (Hermiticity, particle exchange, and index-pair transposition).

## Example

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

## Setup

### OS and hardware platform support

* Windows (x86-64)
* Generic Linux (x86-64)
* macOS (Apple silicon)

### Julia (64-bit) compatibility

Paulimorphic requires Julia **1.12 or later** and aims to support the [**current stable release** of 64-bit Julia](https://julialang.org/downloads/#current_stable_release) as soon as possible. The earliest supported release, the current stable release, and the latest prerelease are continuously tested; the results can be found [here](https://github.com/frankwswang/Paulimorphic.jl/actions/workflows/CI.yml).

### Installation in the Julia [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/)

Type `]` in the default [Julian mode](https://docs.julialang.org/en/v1/stdlib/REPL/#The-Julian-mode) to switch to the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode):

```julia
(@v1.x) pkg>
```

Type the following command and hit the *Enter* key to install Paulimorphic:

```julia
(@v1.x) pkg> add Paulimorphic
```

After the installation completes, hit the *Backspace* key to go back to the Julian mode and use [`using`](https://docs.julialang.org/en/v1/base/base/#using) to load Paulimorphic:

```julia
julia> using Paulimorphic
```

## Documentation Contents

```@contents
Pages = ["types.md", "functions.md"]
Depth = 2
```

## Notation and Conventions

The public docstrings and tutorials in this documentation follow a several typesetting conventions:

* Decorative symbol ``\hat{}`` marks an operator (e.g., ``\hat{X}`` for a single-site Pauli ``X`` operator).

* Symbol ``\mathrm{i}`` denotes the imaginary unit.

* Double square brackets ``\llbracket`` and ``\rrbracket``, as in ``\hat{O}\llbracket f_1, \ldots, f_N \rrbracket``, denote the action of an operator on functions.

* single square brackets ``[`` and ``]``, as in ``\mathtt{inte}[\mathtt{c_1}, \mathtt{a_1}, \ldots, \mathtt{c_N}, \mathtt{a_N}]``, denote the Cartesian indexing of a tensor following the default [array-indexing style](https://docs.julialang.org/en/v1/manual/arrays/#man-array-indexing) in Julia.

* Mathematical variable represented by typewriter symbols (i.e., a monospace format) stand for identifiers of code objects (e.g., ``\\mathtt{inte}`` directly mirrors `inte`). A mathematical variable may still partially reference a code object outside of this format when the code object serves as a decorative symbol within the context. For instance, ``i_k`` appeared in a docstring of code expression may reference `k` in the code.
