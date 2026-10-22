# Development Guide

## Naming conventions

The naming of objects defined in Paulimorphic should follow a set of rules such that their 
names can serve as the indicators of their functionalities and potential extensions.

### Functions

Function names are in camelCase. Except for common-operation names that can be described by 
single verbs (e.g., [`Paulimorphic.add`](@ref)) and well-established algorithm names (e.g., 
[`breadthFirstSearch`](@ref)), they should start with a verb. The following verb prefixes 
carry fixed meanings:

* `is*` verifies a property of a single object (e.g., [`isHermitian`](@ref)).

* `check*` performs a relational check between multiple objects 
  (e.g., [`checkCommute`](@ref)) or a multi-condition validity check (e.g., 
  [`checkNBodyInteTensor`](@ref)).

* `count*`, `list*`, `get*`, `index*`, `find*`, and `contain*` query information about 
  their input without altering it (e.g., [`countTerms`](@ref)).

* `to*` returns a newly allocated object derived from its input, typically in another 
  representation (e.g., [`toString`](@ref)); `as*` is reserved for views that share data 
  with the input.

* `gen*` and `build*` construct new objects (e.g., [`genLineGraph`](@ref), 
  [`buildMajoranaFrame`](@ref)).

* `eval*` evaluates the well-defined **value** of an input object or of an operation on 
  objects. For instance, the value can be a number (e.g., [`evalPhase`](@ref)) or an 
  operator (e.g., [`evalCommute`](@ref)).

A function that (optionally) mutates any of its arguments should have a name that ends 
with `!`; when exactly one argument is mutated, it is typically the first non-function, 
non-type argument. A function is exempt from this rule when the only arguments it 
(optionally) mutates are reusable cache-like data structures, which must be named with the 
suffix `!Self` (e.g., `cache!Self` in [`breadthFirstSearch`](@ref)).

### Types

Types and type aliases are in PascalCase (e.g., [`PauliStr`](@ref)). In particular, the 
following types should obey additional rules:

* The members of an [`enum`](https://docs.julialang.org/en/v1/base/base/#Base.Enums.@enum) 
  type should be named in camelCase with at least one upper-case letter (e.g., `symX` under [`PauliSym`](@ref)).

### Non-type constants

The names of non-type constants should start with `CONSTVAR!!`, followed by a compound word 
in camelCase (e.g., `Paulimorphic.CONSTVAR!!maximalFullShowSize`).

### Common abbreviations

The following abbreviations are used inside compound names (and their first letters may be 
capitalized to accommodate the applied letter-case style):

* `str`: "string"

* `sym`: "symbol"

* `enc`: "encoding"

* `inte`: "integral"

* `ham`: "Hamiltonian"

* `elec`: "electron", "electronic"

* `com`: "commutation", "commutator"

## Export policy

A public object is also exported unless it is one of the following:

* A function to which a method extended from another module directly forwards, i.e., the 
  extension is a thin wrapper of at least one of its methods. For instance, `Base.:*` 
  forwards to [`Paulimorphic.mul`](@ref), thus `mul` is public but not exported.

* A function that manipulates instances of [`PauliStr`](@ref) or [`PauliSum`](@ref) but does not 
  necessarily preserve or restore their canonical forms (e.g., 
  [`Paulimorphic.setCoeff!`](@ref)).

* A function that is designed as a lower-level building block for exported functions 
  (e.g., [`Paulimorphic.genNBodyOperator`](@ref)).
