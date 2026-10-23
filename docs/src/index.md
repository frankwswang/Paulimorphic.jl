# Paulimorphic.jl

[**Paulimorphic**](https://github.com/frankwswang/Paulimorphic.jl) is a Julia software package for **constructing**, **transforming**, and **analyzing** Pauli operators. In particular, Pauli strings are stored in a bit-packed symplectic representation together with their group phases, and their linear combinations are kept in a canonical form. Based on this core design, **Paulimorphic** aims to serve as a robust and performant toolkit for algebraic manipulation and structural analysis on linear operators in the qubit basis.

## Features

* **Efficient and deterministic representation** of Pauli strings (`PauliStr`) and Pauli sums (`PauliSum`).
* **Operator algebra**: operator summation, scalar–operator and operator–operator multiplication, Hermitian adjoints, commutation and anticommutation evaluations, etc.
* **Multi-level data manipulation**: shifting and pasting single-site operators; reframing and truncating Pauli sums, etc.
* **Frustration-graph analysis**: constructing the anticommutation graph of operators, backed by lightweight graph-theory analysis functionalities, such as connected components, breadth-first search, isomorphism detection, and root-graph reconstruction.
* **Fermion-to-qubit encodings**: Jordan–Wigner, parity, and Bravyi–Kitaev Majorana encodings, conversion between Majorana and Dirac forms, and validity checkers for both.
* **Molecular electronic Hamiltonians**: encoding from one- and two-body molecular integral tensors with spin-sectored encodings, selectable operator ordering (`NormalOrder` or `PairedOrder`), and validation of integral symmetries (Hermiticity, particle exchange, and index-pair transposition).

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

## Contents

### Examples

```@contents
Pages = ["examples/ising.md"]
Depth = 2
```

### Types

```@contents
Pages = [
    "types/operators.md", 
    "types/collections.md", 
    "types/labels.md"
]
Depth = 2
```

### Functions

```@contents
Pages = [
    "functions/manipulations.md", 
    "functions/inspections.md", 
    "functions/representations.md", 
    "functions/extensions.md", 
]
Depth = 2
```

### Development Guide

```@contents
Pages = ["development.md"]
Depth = 2
```

## Notation

The public docstrings and tutorials in this documentation follow several typesetting conventions:

* Decorative symbol ``\,\hat{}\,`` marks an operator (e.g., ``\hat{X}`` for a single-site Pauli ``X`` operator).

* Symbol ``\mathrm{i}`` denotes the imaginary unit.

* Double square brackets ``\llbracket`` and ``\rrbracket``, as in ``\hat{O}\llbracket f_1, \ldots, f_N \rrbracket``, denote the action of an operator on functions.

* Single square brackets ``[`` and ``]``, as in ``\mathtt{inte}[c_1, a_1, \ldots, c_N, a_N]``, denote the Cartesian indexing of a tensor following the default [array-indexing style](https://docs.julialang.org/en/v1/manual/arrays/#man-array-indexing) in Julia.

* Mathematical variables represented by typewriter symbols (i.e., in a monospace format) stand for identifiers of code objects (e.g., ``\mathtt{inte}`` directly mirrors `inte`). A mathematical variable may still partially reference a code object outside of this format when the code object serves as a decorative symbol within the context. For instance, ``i_k`` appearing in a docstring of code expression may reference `k` in the code.

* Every public type alias is always presented by its name in the docstrings, error messages, and throughout this documentation, regardless of whether that name is the same as its default display (stringification) format (e.g., [`SignedReal`](@ref)).
