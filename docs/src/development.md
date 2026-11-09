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
  type should be named in camelCase with at least one upper-case letter (e.g., `symX` under 
  [`PauliSym`](@ref)).

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

## Optional argument policy

The optional arguments of a function method (i.e., its arguments with default values) 
should be declared as positional or keyword arguments according to the following rules:

* When a method has only one optional argument, it should be a positional argument by 
  default (e.g., `lowToHigh` in [`shift!`](@ref)).

* When a method has multiple optional arguments, in general, at most one of them should be 
  a positional argument, and the others should be keyword arguments (e.g., `srcRange` and 
  `lowToHigh` in [`paste!`](@ref), respectively). More than one optional positional 
  argument is allowed only when each later one is either semantically dependent on the 
  earlier ones or closely related to them in functionality but (intended to be) less 
  important, so that all the optional arguments are positioned in decreasing order of 
  importance (e.g., `phase` after `siteOp` in a constructor of [`PauliStr`](@ref)). 
  Optional arguments that are closely related in functionality and equally important should 
  instead all be keyword arguments, which are position-agnostic (e.g., `denseString` and 
  `omitPlusSign` in [`toString`](@ref)).

* When an argument is a keyword argument in one method of a function, it should also be a 
  keyword argument in every other method of the function that takes it, even when it is the 
  only optional argument of that method (e.g., `omitPlusSign` in method 
  [`toString(::PhaseFactor; omitPlusSign::Bool=false)`](@ref)).

## Argument conversion policy

A public function that takes an `Integer`-based argument (e.g., an index or a count) but 
hands an equivalent `Int`-based datum to its internal subroutines should perform the 
conversion at the beginning of its body using `Paulimorphic.toBitIntData`, which throws a 
well-formatted error when the conversion is not possible. Accordingly, the docstring of 
such a function need not state any condition on the argument beyond its being 
`Integer`-based, whereas the lower-level internal functions behind it should only receive 
the corresponding `Int`-based variables as arguments.

## Export policy

A public object is also exported unless it is one of the following:

* A function to which a method extension of another module's function directly forwards, 
  i.e., the extension is a thin wrapper of at least one of its methods. For instance, 
  `Base.:*` forwards to [`Paulimorphic.mul`](@ref), thus `mul` is public but not exported.

* A function that manipulates instances of [`PauliStr`](@ref) or [`PauliSum`](@ref) but  
  does not necessarily preserve or restore their canonical forms.

* A function that is designed as a lower-level building block for exported functions 
  (e.g., [`Paulimorphic.genNBodyOperator`](@ref)).
