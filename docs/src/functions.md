```@meta
CurrentModule = Paulimorphic
```

# Functions

## Manipulations

### Algebra

```@docs
add
sub
mul
scale!
toAdjoint
evalCommute
evalAntiCom
```

### Modifications

```@docs
canonicalize!
sanitize!
sortStrings!
setCoeff!
curtail
reframe
shift!
paste!
stamp!
```

## Inspections

### Graphs

```@docs
countVertices
countEdges
containEdge
getDegree
listDegrees
listEdges
listComponents
attachEdge!
removeEdge!
breadthFirstSearch
isIsomorphic
genLineGraph
genRootGraph
decompose
```

### Operators

```@docs
countSites
countWeight
countTerms
indexSite
indexTerm
collectTerms
sumCoeffs
isIdentity
isHermitian
checkCommute
checkAntiCom
getFrustrationInfo
checkMajoranaEnc
checkDiracEnc
checkSpinSectoredEnc
```

### Structures

```@docs
isIndexLabel
findAxisPermViolation
checkInteHermiticity
checkInteParticleExchange
checkIntePairTransposition
checkNBodyInteTensor
```

## Representations

### Text serialization

```@docs
@pauli_str
toPauliStr
toString
toPhase
```

### Qubits encoding

```@docs
genJordanWignerEnc
genParityEnc
genBravyiKitaevEnc
toMajoranaPair
toDiracEnc
toMajoranaEnc
buildMajoranaFrame
formatSpinSectoredEnc
formatMolecularInteData
genNBodyOperator
gen1BodyOperatorSum
genNBodyOperatorSum
encodeElecHam
```

### Numerical computation

```@docs
evalPhase
toMatrix
extendType
```

## Extensions

### `Base` methods

```@docs
Base.isless(::PauliStr, ::PauliStr)
Base.show(::IO, ::PauliStr)
Base.show(::IO, ::MIME"text/plain", ::PauliStr)
```
