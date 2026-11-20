export toMatrix, toMatrix!

public isIndexLabel, findAxisPermViolation

using SparseArrays: spzeros, SparseMatrixCSC

"""
    isIndexLabel(candidate::Tuple{Integer, Vararg{Integer}}, 
                 iStart::Integer=first(candidate)) -> Bool

Return `true` if `candidate` is considered a valid integer-index label, of which every two 
elements are equal if and only if the objects they index are considered the same object. In 
other words, `candidate` should obey the following canonical form: every element must be 
an `Integer` no smaller than `iStart` and no larger than one plus the maximum of all its 
preceding elements; in particular, the first element must equal `iStart`. Hence, the 
distinct values in a valid `candidate` always form a contiguous integer range starting at 
`iStart`, and for any fixed `iStart`, exactly one valid labeling exists per equality 
pattern. If `candidate` violates this canonical form, `isIndexLabel` returns `false`.
"""
function isIndexLabel(candidate::NonEmptyTuple{Integer}, 
                      iStart::Integer=first(candidate))::Bool
    labelMax = iStart - 1

    for i in eachindex(candidate)
        label = candidate[i]
        (iStart <= label <= labelMax+1) || (return false)
        labelMax = max(labelMax, label)
    end

    true
end


"""
    toMatrix(::Type{T}, sym::PauliSym) where {T<:SignedReal} -> 
    $(SparseMatrixCSC|>repr){Complex{T}, Int}

    toMatrix(sym::PauliSym) -> $(SparseMatrixCSC|>repr){Complex{Int}, Int}

Return the 2-by-2 matrix representation (in the Pauli-Z eigenbasis) of the single-site 
Pauli operator tagged by `sym::`[`PauliSym`](@ref), with element type as `Complex{T}`. When 
`T` is unspecified, method `toMatrix(sym::PauliSym)` defaults it to `Int`.

    toMatrix(::Type{T}, sym::PauliSym, nSite::Integer; reverseKron::Bool=false
             ) where {T<:SignedReal} -> 
    $(SparseMatrixCSC|>repr){Complex{T}, Int}

    toMatrix(sym::PauliSym, nSite::Integer; reverseKron::Bool=false) -> 
    $(SparseMatrixCSC|>repr){Complex{Int}, Int}

Return a sparse square matrix of order `2^nSite` (`nSite` must be positive) that represents 
the operator that acts as `sym` on the first of `nSite` sites and as the identity on the 
other sites (in the Pauli-Z eigenbasis). The format of the returned matrix (and 
correspondingly argument constraints) is the same as that for method 
[`toMatrix(h, nSite; reverseKron)`](@ref `toMatrix(::PauliSum)`) with 
`h=PauliSum(T, [PauliStr(1, sym)])`, where `reverseKron` controls whether the order of 
the chained Kronecker product is in the same or reverse order as the sites are displayed in 
a Pauli string.

For all four method signatures, `T` is subject to the same restriction as the core data 
type of any [`PauliSum`](@ref) instance and defaults to `Int`; the returned matrix is also 
newly allocated on every call and shares no data with any global variable.

# Examples
```jldoctest
julia> toMatrix(symY) == [0 -im; im 0]
true

julia> toMatrix(symX, 2) == toMatrix(pauli"XI")
true

julia> toMatrix(symX, 2, reverseKron=true) == kron(toMatrix(symI), toMatrix(symX))
true
```
"""
function toMatrix(::Type{T}, sym::PauliSym) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    mat = spzeros(Complex{T}, 2, 2)
    toMatrix!(mat, sym)
end

toMatrix(::Type{T}, sym::PauliSym, nSite::Integer; reverseKron::Bool=false
         ) where {T<:SignedReal} = 
toMatrix(T, PauliStr(1, sym), nSite; reverseKron)

toMatrix(sym::PauliSym) = toMatrix(Int, sym)

toMatrix(sym::PauliSym, nSite::Integer; reverseKron::Bool=false) = 
toMatrix(Int, sym, nSite; reverseKron)


#> Largest site count `n` for which the (square) matrix order `2^n` (and the length 
#> `2^n + 1` of the column pointers for a `SparseMatrixCSC`) is representable as an `Int`.
const CONSTVAR!!matrixRepMaxSiteCount = 8sizeof(Int) - 2

function getPauliOpMatOrder(nSite::Int)
    if !(0 <= nSite <= CONSTVAR!!matrixRepMaxSiteCount)
        nSiteRange = 0:CONSTVAR!!matrixRepMaxSiteCount
        throw(DomainError(nSite, "The site count `nSite` of a Pauli operator must be in "*
                                 "$nSiteRange so that the order of the operator's matrix "*
                                 "representation `2^nSite` can be expressed as an `Int`."))
    end
    one(Int) << nSite
end


#> Obtain the formatted info of k-th term in `h::PauliSum` as `xMasks[k]` / `zMasks[k]` and 
#> the associated `coeffs[k]`, independent of the column-major Pauli-Z basis |b> they act 
#> on: xStr(xMasks[k]) zStr(zMasks[k]) |b> = (-1)^count_ones(zMasks[k] & b) |b ⊻ xMasks[k]>
#> The assembly is done in `T`, which is deterministic when `h` is in the canonical form.
#> The masks and `b` share one bit layout: by default, site `k` is bit `nSite-k` (so that 
#> site `1` is the most significant bit of the basis-state index, from right to left); when 
#> `reverseKron=true`, site `k` is bit `k-1`, which is the bit layout of `PauliStr`.
#> NOTE: `nSite` can be lower than `countSites(h)` since this is an internal function.
function formatPauliOpMatInfo(::Type{T}, h::PauliSum, nSite::Int=countSites(h); 
                              reverseKron::Bool=false) where {T<:SignedReal}
    bitsMask = (UInt∘getPauliOpMatOrder)(nSite) - one(UInt) #> Also check `nSite` bound
    nShift = 8sizeof(UInt) - nSite #> Positive, since `nSite` has passed the check above

    nTerm = countTerms(h)
    coeffT = Complex{T}
    coeffs = Memory{coeffT}(undef, nTerm)
    xMasks = Memory{UInt}(undef, nTerm)
    zMasks = Memory{UInt}(undef, nTerm)

    @inbounds for k in 1:nTerm
        str = h.str[begin+k-1]

        #> Obtain Pauli X/Z mask of `str` as a single `UInt`, since `nSite` is constrained 
        #> by `CONSTVAR!!matrixRepMaxSiteCount`, lower than the bit count in one `UInt`. 
        #> Bits above `nSite` are cleared (via `bitsMask`) regardless of layout of `str`.
        xMask, zMask = map((:x, :z)) do prprty
            buffer = getproperty(str, prprty)
            word = isempty(buffer) ? zero(UInt) : (first(buffer) & bitsMask)
            #>>  Case when `reverseKron=` | `true` | `false`
            #>>  00XXXX (4 sites, 6 bits) | 004321 | 001234  (number-labeled sites)
            reverseKron ? word : (bitreverse(word) >> nShift)
        end

        phaseY = PhaseFactor(count_ones(xMask & zMask) & 3) #> Compute im^nPauliY
        phase = mul(str.phase, phaseY)
        coeff = coeffT(h.coeff[begin+k-1])
        phase == posRea || (coeff = adaptMul( coeff, (coeffT∘evalPhase)(phase) ))
        xMasks[begin+k-1] = xMask
        zMasks[begin+k-1] = zMask
        coeffs[begin+k-1] = coeff
    end

    xMaskCount = (length∘Set)(xMasks) #> Number of unique `xMask` in `xMasks`
    xMaskCount => (xMasks, zMasks, coeffs) #> There may be repeated elements in `xMasks`
end


function genPauliOpMatEntryEvaluator(::Type{T}, 
                                     xMasks::AbstractVector{UInt}, 
                                     zMasks::AbstractVector{UInt}, 
                                     coeffs::AbstractVector{C}
                                     ) where {T<:SignedReal, C<:Complex{<:SignedReal}}
    extT = extendType(T, C)
    nTerm = length(coeffs)

    if !(nTerm == length(xMasks) == length(zMasks))
        throw(ArgumentError("`xMasks`, `zMasks`, and `coeffs` must have the same length."))
    end

    #> Permuted indices are stable and zero-based: used for grouping strings of same `xMask`
    #> NOTE: The grouping is fixed here, so `xMasks` must not be modified afterwards.
    posPerm = (sortperm(xMasks) .-= firstindex(xMasks))

    function evaluateEntry(processor::F, zBitStr::UInt, nzCount::Int
                           ) where {F<:AbstractCallable}
        nzCount < 0 && throw(DomainError(nzCount, "`nzCount` must be non-negative."))
        naiveSum = zero(extT)
        sResidue = zero(extT)

        @inbounds for (i, shift) in enumerate(posPerm)
            coeff = extT(coeffs[begin+shift]) #> `coeff` is represented in the extended type
            xMask = xMasks[begin+shift]
            zMask = zMasks[begin+shift]

            if !iszero(coeff)
                #> Update the coefficient associated with a Pauli-Z basis state labeled by 
                #> `zBitStr` when a Pauli term P(`xMask`, `zMask`, `coeff`) acts on it:
                #> P |zBitStr> = coeff * (-1)^count_ones(zMask & zBitStr) |zBitStr ⊻ xMask>
                termCoeff = condNeg(coeff, isodd(count_ones(zMask & zBitStr)))

                #> Add the coefficient associated with a Pauli term of the same `xMask`
                naiveSum, sResidue = neumaierAdd(naiveSum, termCoeff, sResidue)
            end

            #> At the end of current `xMask` group
            if i == nTerm || xMask != xMasks[begin+posPerm[begin+i]]
                #>> Process the accumulated coefficient sum from the current group
                nzRow = reinterpret(Int, xor(xMask, zBitStr)) + 1
                nzCol = reinterpret(Int, zBitStr) + 1
                coeffSum = Complex{T}(absorbResidue(naiveSum, sResidue))
                if !iszero(coeffSum)
                    #>> `(nzRow, nzCol)` points to `coeffSum` for terms with equal `xMask`
                    processor((nzRow, nzCol)=>coeffSum, nzCount)
                    nzCount += 1
                end

                #>> Reset both summation components to zero
                naiveSum = zero(extT)
                sResidue = zero(extT)
            end
        end

        nzCount
    end
end


function checkPauliSumMatConfig(::Type{T}, h::PauliSum, nSite::Integer
                                ) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)

    nSiteInt = toBitIntData(nSite, :nSite)
    nSiteMin = countSites(h)
    if nSiteInt < nSiteMin
        throw(DomainError(nSite, "`nSite` must be at least `countSites(h)==$nSiteMin`."))
    end

    nSiteInt
end


"""
    toMatrix(::Type{T}, str::PauliStr, nSite::Integer=countSites(str); 
             reverseKron::Bool=false) where {T<:SignedReal} -> 
    $(SparseMatrixCSC|>repr){Complex{T}, Int}

    toMatrix(str::PauliStr, nSite::Integer=countSites(str); reverseKron::Bool=false) -> 
    $(SparseMatrixCSC|>repr){Complex{Int}, Int}

Return the matrix representation (in the Pauli-Z eigenbasis) of `str` on `nSite` sites (by 
default, equal to [`countSites(str)`](@ref `countSites`)) as a sparse square matrix of 
order `2^nSite` with element type `Complex{T}`. `T` is subject to the same restriction as 
the core data type of any [`PauliSum`](@ref) instance and defaults to `Int`. The returned 
matrix is newly allocated and shares no data with `str`.

The format of the returned matrix (and correspondingly argument constraints) is the same as 
that for method [`toMatrix(h, nSite; reverseKron)`](@ref `toMatrix(::PauliSum)`) with 
`h=PauliSum(T, [str])`, where `reverseKron` controls whether the order of the chained 
Kronecker product is in the same or reverse order as the sites are displayed in a Pauli 
string. In particular, a zero-site `str` yields by default a 1-by-1 matrix holding 
[`evalPhase(str)`](@ref `evalPhase(::PauliStr)`).

# Examples
```jldoctest
julia> toMatrix(pauli"XI") == [0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]
true

julia> toMatrix(pauli"X", 2) == toMatrix(pauli"XI") == kron(toMatrix(symX), toMatrix(symI))
true

julia> let str=pauli"-iXYZ"
           mats = [toMatrix(indexSite(str, k)) for k in 1:3]
           toMatrix(str) == evalPhase(str) .* reduce(kron, mats)
       end
true

julia> let v=toMatrix(pauli"IIX", reverseKron=true) * [1, 0, 0, 0, 0, 0, 0, 0]
            reshape(v, 2, 2, 2)[1, 1, 2] == 1
       end
true

julia> let w=toMatrix(pauli"IIX") * [1, 0, 0, 0, 0, 0, 0, 0]
            reshape(w, 2, 2, 2)[2, 1, 1] == 1
       end
true
```
"""
toMatrix(::Type{T}, str::PauliStr, nSite::Integer=countSites(str); 
         reverseKron::Bool=false) where {T<:SignedReal} = 
toMatrix(PauliSum(T, genMemory(str)), nSite; reverseKron)

toMatrix(str::PauliStr, nSite::Integer=countSites(str); reverseKron::Bool=false) = 
toMatrix(Int, str, nSite; reverseKron)


"""
    toMatrix(::Type{T}, h::PauliSum{T1}, nSite::Integer=countSites(h); 
             requireCanonical::Bool=true, reverseKron::Bool=false
             ) where {T<:SignedReal, T1<:SignedReal} -> 
    $(SparseMatrixCSC|>repr){Complex{T}, Int}

    toMatrix(h::PauliSum{T}, nSite::Integer=countSites(h); requireCanonical::Bool=true, 
             reverseKron::Bool=false) where {T<:SignedReal} -> 
    $(SparseMatrixCSC|>repr){Complex{T}, Int}

Return the matrix representation of `h` on `nSite` sites as a sparse square matrix of order 
`2^nSite` with element type `Complex{T}`. `T` is subject to the same restriction as 
the core data type of any [`PauliSum`](@ref) instance and defaults to that of `h`. The 
returned matrix is newly allocated and shares no data with `h`. An empty `h` yields a zero 
matrix of order `2^nSite`.

# Site count
The value of `nSite` must be in [`countSites(h)`](@ref `countSites`)`:(8sizeof(Int) - 2)`, 
where the upper bound keeps the axial index of the returned matrix (and related internal 
array structures) representable as an `Int`; otherwise, a `DomainError` is thrown. When 
`nSite` is larger than its default value `countSites(h)`, the string associated with each 
term of `h` acts as the identity on the sites above `countSites(h)`, following the implicit 
identity-padding convention of [`PauliStr`](@ref).

# Basis ordering convention
The returned matrix `res` is always expressed with respect to the eigenbasis of the (tensor 
product of) single-site Pauli-Z operators, i.e., the computational basis. When 
`reverseKron=false` (by default), the matrix representation of the (phase-absorbed) Pauli 
string associated with each term in `h` matches the Kronecker product (via 
`Base.kron`) of (single-site) Pauli matrices in ascending-site order from left to right 
(i.e., same as the text literals such as `pauli"ZZY"` used to generate [`PauliStr`](@ref)):

```math
M_1 \\otimes M_2 \\otimes \\cdots \\otimes M_{\\mathtt{nSite}}, 
```

where ``M_k`` denotes the matrix returned by `toMatrix(T, sym)` for the single-site 
operator `sym` in the string on site ``k``. Accordingly, the (one-based) axial index 
`i::Int` (`1 <= i <= 2^nSite`) of `res` labels the computational-basis state for which 
the corresponding bitstring is also the binary representation of `i-1` as a decimal number 
(i.e., a lexicographic order):

```math
\\mathtt{res[i, j]} = \\langle \\mathtt{bitstring(i-1)[(end-nSite+1):end]} \\lvert
                      \\mathtt{h}
                      \\lvert \\mathtt{bitstring(j-1)[(end-nSite+1):end]} \\rangle
```

In contrast, when `reverseKron=true`, the order of the Kronecker product for the 
(single-site) Pauli matrices associated with the same Pauli string is chained in a 
reversed order:

```math
M_{\\mathtt{nSite}} \\otimes \\cdots \\otimes M_2 \\otimes M_1, 
```

and the bitstring labeling each axial index is also reversed accordingly.

Compared to the lexicographic order, the reversed order can be more convenient for 
numerical computations in mainly two respects: 
- It aligns with arrays with a column-major linear layout. For instance, for a `v::Vector` 
  expressed in the reversed-order basis, the `k`-th axis of 
  `reshape(v, ntuple(_->2, nSite))` indexes the state of site `k`, whereas in the 
  lexicographic-order basis it indexes the state of site `nSite + 1 - k`.
- It keeps the identity padding separable: appending sites (e.g., from `pauli"X"` to 
  `pauli"XI"`) does not change the vector indices of the existing basis states. For 
  instance, for `mat1 = toMatrix(h, nSite; reverseKron=true)` (assuming `nSite > 0` and 
  `h` has at least one term), `mat2 = toMatrix(h, nSite+1; reverseKron=true)` is a 
  block-diagonal matrix that is the direct sum of `mat1` (i.e., 
  `mat2 == cat(mat1, mat1, dims=(1, 2))`), whereas in the lexicographic-order basis, the 
  entries of two `mat1` are interleaved in `mat2`.

# Evaluation of the entries
Each entry of the returned matrix is evaluated at the precision level of 
`complex($(extendType|>repr)(T, T1))` by accumulating over every term that has a 
non-zero contribution to it (i.e., every term whose matrix representation has a non-zero 
value at that entry index). When this precision level is a floating-point one, the 
accumulation is realized by 
[Neumaier summation](https://doi.org/10.1002/zamm.19740540106). If the resulting value of 
the entry is exactly zero, the entry value is not stored. To ensure deterministic entry 
values regardless of different formats of `h` with equivalent mathematical expressions, a 
check for the canonical form of `h` is performed at the beginning of the function call. 
If `h` is already in the canonical form, it is directly used for the subsequent 
accumulation subroutine. Otherwise, an `ArgumentError` is thrown when 
`requireCanonical=true`, or a copy equivalent to `PauliSum(h, false)` is constructed and 
used for the subroutine when `requireCanonical=false`.

# Examples
```jldoctest
julia> let h=PauliSum(Float64, [pauli"ZI", pauli"IZ"], [1, 1])
            toMatrix(h) == [2 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -2] && 
            toMatrix(h, 3) == toMatrix(PauliSum([pauli"ZII", pauli"IZI"], [1, 1]))
       end
true

julia> let hRaw=PauliSum([pauli"XI", pauli"IZ"], [1, 2])
           reverse!(hRaw.str)
           reverse!(hRaw.coeff)
           !isCanonical(hRaw) && 
           toMatrix(hRaw, requireCanonical=false) == toMatrix(PauliSum(hRaw))
       end
true

julia> let h=PauliSum([pauli"XI", pauli"IZ"], [1, 2]), mat1=toMatrix(h, reverseKron=true)
           mat2 = toMatrix(h)
           mat2 == [2 0 1 0; 0 -2 0 1; 1 0 2 0; 0 1 0 -2] && 
           mat1 == [2 1 0 0; 1 2 0 0; 0 0 -2 1; 0 0 1 -2] && 
           toMatrix(h, 3) == kron(mat2, toMatrix(symI))   && 
           toMatrix(h, 3, reverseKron=true) == kron(toMatrix(symI), mat1)
       end
true
```
"""
function toMatrix(::Type{T}, h::PauliSum{T1}, nSite::Integer=countSites(h); 
                  requireCanonical::Bool=true, reverseKron::Bool=false
                  ) where {T<:SignedReal, T1<:SignedReal}
    nSiteInt = checkPauliSumMatConfig(T, h, nSite)
    order = getPauliOpMatOrder(nSiteInt)

    checkPauliOpForm(h, requireCanonical)
    hLocal = requireCanonical ? h : lazyCanonicalize(h, false)
    matInfo = formatPauliOpMatInfo(extendType(T, T1), hLocal, nSiteInt; reverseKron)
    nXM, (xMasks, zMasks, coeffs) = matInfo

    colPtrs = Vector{Int}(undef, order+1)
    maxSave = adaptMul(order, nXM)
    nzvRows = Vector{Int}(undef, maxSave)
    nzvList = Vector{Complex{T}}(undef, maxSave)
    permBuf = Memory{Int}(undef, nXM)
    rowsBuf = Memory{Int}(undef, nXM)
    valsBuf = Memory{Complex{T}}(undef, nXM)

    evaluator = genPauliOpMatEntryEvaluator(T, xMasks, zMasks, coeffs)

    nzCount = 0
    @inbounds for iCol in 0:(order-1)     #> Zero-based column indexer
        zBasisBitStr = UInt(iCol)         #> Bit-string label in Pauli-Z basis for `iCol`
        colPtrs[begin+iCol] = nzCount + 1 #> `colPtrs` must start with `1`

        nStart = nzCount
        nzCount = evaluator(zBasisBitStr, nzCount) do pair, nCountLocal
            nzvRows[begin+nCountLocal] = first(pair.first)
            nzvList[begin+nCountLocal] = pair.second
        end
        nFinal = nzCount - 1 #> `nzCount` has been updated to the `nStart` for next column

        #> Reformat non-zero-element row indices in an ascending order for column `iCol`
        if nFinal > nStart #>> `iCol` has at least two non-zero entries
            nEntry = nFinal - nStart + 1
            nzRowSector = @view nzvRows[(begin+nStart):(begin+nFinal)]
            permBSector = @view permBuf[begin:(begin+nEntry-1)]
            #>> `QuickSort` works in place, whereas `sortperm!` by default may allocate
            sortperm!(permBSector, nzRowSector; alg=QuickSort) #>> `permBSector` one-based
            for iEntry in 0:(nEntry-1) #>> Store reordered vals and rows in the buffers
                iSource = nStart + permBuf[begin+iEntry] - 1
                rowsBuf[begin+iEntry] = nzvRows[begin+iSource]
                valsBuf[begin+iEntry] = nzvList[begin+iSource]
            end

            for iEntry in 0:(nEntry-1) #>> Write in vals and rows from the buffers
                nzvRows[begin+nStart+iEntry] = rowsBuf[begin+iEntry]
                nzvList[begin+nStart+iEntry] = valsBuf[begin+iEntry]
            end
        end
    end
    colPtrs[end] = nzCount + 1

    nzvTotalNum = nzCount
    for buffer in (nzvRows, nzvList) #> Release the unused capacity
        resize!(buffer, nzvTotalNum)
        sizehint!(buffer, nzvTotalNum)
    end

    SparseMatrixCSC(order, order, colPtrs, nzvRows, nzvList)
end

toMatrix(h::PauliSum{T}, nSite::Integer=countSites(h); requireCanonical::Bool=true, 
         reverseKron::Bool=false) where {T<:SignedReal} = 
toMatrix(T, h, nSite; requireCanonical, reverseKron)


"""
    toMatrix!(dst::AbstractMatrix{C}, sym::PauliSym
              ) where {T<:SignedReal, C<:Union{Complex{T}, T}} -> 
    typeof(dst)

    toMatrix!(dst::AbstractMatrix{C}, sym::PauliSym, nSite::Integer; 
              reverseKron::Bool=false
              ) where {T<:SignedReal, C<:Union{Complex{T}, T}} -> 
    typeof(dst)

    toMatrix!(dst::AbstractMatrix{C}, str::PauliStr, nSite::Integer=countSites(str); 
              reverseKron::Bool=false
              ) where {T<:SignedReal, C<:Union{Complex{T}, T}} -> 
    typeof(dst)

    toMatrix!(dst::AbstractMatrix{C}, h::PauliSum{T1}, nSite::Integer=countSites(h); 
              requireCanonical::Bool=true, reverseKron::Bool=false
              ) where {T<:SignedReal, C<:Union{Complex{T}, T}, T1<:SignedReal} -> 
    typeof(dst)

Overwrite `dst` in place with the matrix representation of a [`PauliSum`](@ref), a 
[`PauliStr`](@ref), or a [`PauliSym`](@ref) on `nSite` sites, and return `dst`. `dst` must 
be a square matrix of order `2^nSite` (or `2` for the method signature without argument 
`nSite`); otherwise, an `ArgumentError` is thrown. Its element core type `T` is subject to 
the same restriction as the core data type of any [`PauliSum`](@ref) instance. Every 
element of `dst` is overwritten by the corresponding entry of the matrix representation 
converted to `C<:Union{Complex{T}, T}`, and `dst` shares no data with the represented 
operator once the function successfully executes. Therefore, a real `C` requires every 
entry to be real; otherwise, an `InexactError` is thrown (e.g., `pauli"Y"`, whose entries 
are `±im`, cannot be written into a `Matrix{Float64}`). 

For consistency, the overwritten `dst` is element-wise equal (including its basis ordering 
convention) to the square matrix returned by [`toMatrix`](@ref `toMatrix`). Thus, the 
function arguments with correspondingly the same names are also subject to the same 
constraints. For instance, `reverseKron` controls whether the order of the chained 
Kronecker product is in the same or reverse order as the sites are displayed in a Pauli 
string.

!!! info
    Unlike `toMatrix`, this function does not construct a sparse matrix; its typical use is 
    to fill a preallocated `dst` that is a dense or custom-type matrix. In particular, 
    `dst` should support efficient scalar `Base.setindex!` (e.g., a `Matrix` or a view of 
    one). Writing into a `dst::$(SparseMatrixCSC|>repr)` using this function is expected to 
    be prohibitively slow beyond small sizes.

!!! warning
    If an error is thrown after the initial verifications (the shape and the element type 
    of `dst`, as well as the form of `h`) during the reassignment of the entries of `dst` 
    (e.g., an entry cannot be properly updated), `dst` may be left partially overwritten.

# Examples
```jldoctest
julia> toMatrix!(zeros(Int, 4, 4), pauli"ZZ") == [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
true

julia> toMatrix!(zeros(Int, 4, 4), symZ, 2) == [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
true

julia> toMatrix!(zeros(Int, 4, 4), symZ, 2, reverseKron=true) == 
       [1 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
true

julia> let mat=zeros(ComplexF64, 2, 2)
            toMatrix!(mat, pauli"X" + pauli"Y") === mat && 
            mat == [0 1-im; 1+im 0]
        end
true
```
"""
function toMatrix!(dst::Union{AbstractMatrix{Complex{T}}, AbstractMatrix{T}}, 
                   h::PauliSum{T1}, nSite::Integer=countSites(h); 
                   requireCanonical::Bool=true, reverseKron::Bool=false
                   ) where {T<:SignedReal, T1<:SignedReal}
    nSiteInt = checkPauliSumMatConfig(T, h, nSite)
    if !(size(dst, 1) == size(dst, 2) == getPauliOpMatOrder(nSiteInt))
        throw(ArgumentError("`dst` must be a square matrix of order `2^$nSiteInt`."))
    end

    function assign!(pair::Pair{NTuple{2, Int}, Complex{T}}, ::Int)
        (iRow, iCol), matVal = pair
        dst[begin+iRow-1, begin+iCol-1] = matVal
    end

    checkPauliOpForm(h, requireCanonical)
    hLocal = requireCanonical ? h : lazyCanonicalize(h, false) #> Format `h` before mutation
    fill!(dst, zero(Complex{T})) #> Reset `dst` with non-zero entries
    traverseMatrixEntries(assign!, T, hLocal, nSiteInt; requireCanonical=false, reverseKron)

    dst
end

toMatrix!(dst::AbstractMatrix{<:SRealOrComplex}, str::PauliStr, 
          nSite::Integer=countSites(str); reverseKron::Bool=false) = 
toMatrix!(dst, PauliSum( Int, genMemory(str) ), nSite; reverseKron)

toMatrix!(dst::AbstractMatrix{<:SRealOrComplex}, sym::PauliSym, nSite::Integer; 
          reverseKron::Bool=false) = 
toMatrix!(dst, PauliStr(1, sym), nSite; reverseKron)

function toMatrix!(dst::AbstractMatrix{C}, sym::PauliSym) where {C<:SRealOrComplex}
    checkCoreDataTypeForPauliSum(real(C))
    if size(dst) != (2, 2)
        throw(ArgumentError("`dst` must be a 2-by-2 matrix."))
    end

    if sym == symY && (C <: Real)
        formattedImag = complex(C)(CONSTVAR!!pauliYMatEles[begin+1])
        throw(InexactError(nameof(C), C, formattedImag))
    end

    dstVec = vec(dst)

    if sym == symI
        dstVec .= CONSTVAR!!qubitIMatEles
    elseif sym == symX
        dstVec .= CONSTVAR!!pauliXMatEles
    elseif sym == symY
        dstVec .= CONSTVAR!!pauliYMatEles
    else
        dstVec .= CONSTVAR!!pauliZMatEles
    end

    dst
end


"""
    traverseMatrixEntries(processor::Union{Function, Type}, ::Type{T}, h::PauliSum{T1}, 
                          nSite::Integer=countSites(h); requireCanonical::Bool=true, 
                          reverseKron::Bool=false
                          ) where {T<:SignedReal, T1<:SignedReal} -> 
    Int

Traverse the entries of the matrix representation (with element type set to `Complex{T}`) 
of `h` on `nSite` sites without explicitly constructing the matrix, and return the total 
number of the non-zero entries. During the traversal, if an entry at (one-based) 
`index::NTuple{2, Int}` holds a non-zero `val::Complex{T}`, i.e., it is the `count+1`-th 
non-zero entry, `processor` is invoked through

    processor((index=>val)::Pair{NTuple{2, Int}, Complex{T}}, count::Int)

The entries are traversed column by column in ascending order of the column index, whereas 
the order of the entries within a column is not guaranteed to be fixed.

For consistency, the traversed matrix representation is element-wise equal (including its 
basis ordering convention) to the square matrix returned by 
[`toMatrix(T, h, nSite; requireCanonical, reverseKron)`](@ref `toMatrix(::PauliSum)`). 
Thus, the function arguments with correspondingly the same names are also subject to the 
same constraints. `processor` is called only after `T`, `nSite`, and the form of `h` have 
been checked; an error thrown later (e.g., by an arithmetic overflow of an exact numeric 
type) ends the traversal after `processor` has been called for the entries before it.

!!! info
    Since the traversal evaluates the entries one at a time from the terms of `h`, its 
    memory use scales with the number of terms of `h` instead of the order of the matrix. 
    This makes the function suitable for operations that do not need the matrix as a whole 
    (e.g., multiplying the matrix by a vector).

# Examples
```jldoctest
julia> h = PauliSum([pauli"XI", pauli"IZ"], [1, 2]);

julia> x = [1, 2, 3, 4];

julia> y = zeros(Complex{Int}, 4);

julia> $(traverseMatrixEntries|>repr)(Int, h) do entry, count
           (iRow, iCol), val = entry
           y[iRow] += val * x[iCol]
       end
8

julia> y == toMatrix(h) * x
true
```
"""
function traverseMatrixEntries(processor::F, ::Type{T}, h::PauliSum{T1}, 
                               nSite::Integer=countSites(h); requireCanonical::Bool=true, 
                               reverseKron::Bool=false
                               ) where {T<:SignedReal, T1<:SignedReal, F<:AbstractCallable}
    nSiteInt = checkPauliSumMatConfig(T, h, nSite)
    order = getPauliOpMatOrder(nSiteInt)
    checkPauliOpForm(h, requireCanonical)
    hLocal = requireCanonical ? h : lazyCanonicalize(h, false)
    matInfo = formatPauliOpMatInfo(extendType(T, T1), hLocal, nSiteInt; reverseKron)
    xMasks, zMasks, coeffs = matInfo.second
    evaluator = genPauliOpMatEntryEvaluator(T, xMasks, zMasks, coeffs)

    nzCount = 0

    for iCol in 0:(order-1)
        zBasisBitStr = UInt(iCol)
        nzCount = evaluator(processor, zBasisBitStr, nzCount)
    end

    nzCount
end


"""
    findAxisPermViolation(tensor::AbstractArray{C, D}, axisPerm::NTuple{D, Int}, 
                          conjugated::Bool) where {T<:Real, C<:Union{Complex{T}, T}, D} -> 
    Union{Nothing, NTuple{D, Int}}

Return the first index `idx` (in column-major order) of `tensor` at which `tensor[idx...]` 
is not equal to `tensor[permIdx...]` (or to its complex conjugate if `conjugated` is 
`true`), where the "position" of `permIdx` along axis `d` (i.e., the index difference to 
the first index of that axis) equals the position of `idx` along axis 
`axisPerm[begin+d-1]`. If no such `idx` is found (i.e., the symmetry holds everywhere), 
`findAxisPermViolation` returns `nothing`. `axisPerm` must be a permutation of `1:D` such 
that `Base.isperm(axisPerm) == true` and only maps between axes of equal extent.
"""
function findAxisPermViolation(tensor::AbstractArray{<:RealOrComplex, D}, 
                               axisPerm::NTuple{D, Int}, conjugated::Bool) where {D}
    isperm(axisPerm) || throw(ArgumentError("`axisPerm` must be a permutation of `1:$D`."))
    for d in 1:D
        dSrc = axisPerm[begin+d-1]
        if size(tensor, d) != size(tensor, dSrc)
            throw(ArgumentError("`tensor` must have equal extents along axes `$d` and "*
                                "`$dSrc`."))
        end
    end

    iFirstAxial = map(first, axes(tensor))

    #> In-bounds safety: `axisPerm` only moves index positions between equal-extent axes
    @inbounds for carteIdx in CartesianIndices(tensor)
        idx = Tuple(carteIdx)
        pos = idx .- iFirstAxial #> It's necessary as axes may have different first indices
        permIdx = ntuple(d->pos[begin + axisPerm[begin+d-1] - 1], Val(D)) .+ iFirstAxial
        val = tensor[carteIdx]
        partner = tensor[permIdx...]
        matched = conjugated ? checkConj(val, partner) : (val == partner)
        matched || (return idx)
    end

    nothing
end
