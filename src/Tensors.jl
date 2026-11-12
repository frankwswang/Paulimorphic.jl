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


const ComplexSMat{T<:Real, TI<:Integer} = SparseMatrixCSC{Complex{T}, TI}

"""
    toMatrix(::Type{T}, sym::PauliSym) where {T<:SignedReal} -> 
    $(SparseMatrixCSC|>repr){Complex{T}, Int}

    toMatrix(sym::PauliSym) -> $(SparseMatrixCSC|>repr){Complex{Int}, Int}

Return the 2×2 matrix representation (in the Pauli-Z eigenbasis) of the single-site 
Pauli operator tagged by `sym::`[`PauliSym`](@ref), with element type as `Complex{T}`. When 
`T` is unspecified, method `toMatrix(sym::PauliSym)` defaults it to `Int`. The returned 
matrix is newly allocated on every call and shares no data with any global state.
"""
function toMatrix(::Type{T}, sym::PauliSym) where {T<:SignedReal}
    mat = spzeros(Complex{T}, 2, 2)
    toMatrix!(mat, sym)
end

toMatrix(::Type{T}, sym::PauliSym, nSite::Integer) where {T<:SignedReal} = 
toMatrix(T, PauliStr(1, sym), nSite)

toMatrix(sym::PauliSym) = toMatrix(Int, sym)

toMatrix(sym::PauliSym, nSite::Integer) = toMatrix(Int, sym, nSite)


#> Largest site count `n` for which the (square) matrix order `2^n` (and the length 
#> `2^n + 1` of the column pointers for a `SparseMatrixCSC`) is representable as an `Int`.
const CONSTVAR!!matrixRepMaxSiteCount = 8sizeof(Int) - 2

function getPauliOpMatOrder(nSite::Int)
    if !(0 <= nSite <= CONSTVAR!!matrixRepMaxSiteCount)
        nSiteRange = 0:CONSTVAR!!matrixRepMaxSiteCount
        throw(DomainError(nSite, "The site count `nSite` of a Pauli operator must be in "*
                                 "$nSiteRange so that the order of the operator's matrix "*
                                 "representation 2^nSite` can be expressed as a `Int`."))
    end
    one(Int) << nSite
end


#> Obtain the formatted info of k-th term in `h::PauliSum` as `xMasks[k]` / `zMasks[k]` and 
#> the associated `coeffs[k]`, independent of the column-major Pauli-Z basis |b> they act 
#> on: xStr(xMasks[k]) xStr(xMasks[k]) |b> = (-1)^count_ones(zMasks[k] & b) |b ⊻ xMasks[k]>
#> A deterministic assembly is enforced by a canonical checking via `checkPauliOpForm`.
function formatPauliOpMatInfo(h::PauliSum{T}, nSite::Int=countSite(h)) where {T<:SignedReal}
    checkPauliOpForm(h, true, checkSimplification=false)
    bitsMask = (UInt∘getPauliOpMatOrder)(nSite) - one(UInt) #> Also check `nSite` bound

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
            isempty(buffer) ? zero(UInt) : (first(buffer) & bitsMask)
        end

        phaseY = PhaseFactor(count_ones(xMask & zMask) & 3) #> Compute im^nPauliY
        phase = mul(str.phase, phaseY)
        coeff = h.coeff[begin+k-1]
        phase == posRea || (coeff = adaptMul( coeff, (coeffT∘evalPhase)(phase) ))
        xMasks[begin+k-1] = xMask
        zMasks[begin+k-1] = zMask
        coeffs[begin+k-1] = coeff
    end

    (xMasks, zMasks, coeffs) #> There may be repeated elements in `xMasks`
end


function genPauliOpMatEntryEvaluator(::Type{T}, 
                                     xMasks::AbstractVector{UInt}, 
                                     zMasks::AbstractVector{UInt}, 
                                     coeffs::AbstractVector{C}
                                     ) where {T<:SignedReal, C<:Complex{<:SignedReal}}
    accuT = extendType(T, C)
    accumulator = NeuAccumulator{UInt, accuT}()
    nTerm = length(coeffs)
    
    if !(nTerm == length(xMasks) == length(zMasks))
        throw(ArgumentError("`xMasks`, `zMasks`, and `zMasks` must have the same length."))
    end

    function evaluateEntry(processor::F, zBitStr::UInt, nzCount::Int
                           ) where {F<:AbstractCallable}
        nzCount < 0 && throw(DomainError(nzCount, "`nzCount` must be non-negative."))
        empty!(accumulator)

        @inbounds for iTerm in 0:(nTerm-1)     #> Zero-based term indexer
            coeff = accuT(coeffs[begin+iTerm]) #> `coeff` is represented in the extend type
            iszero(coeff) && continue          #> Skip column without any non-zero values

            xMask = xMasks[begin+iTerm]
            zMask = zMasks[begin+iTerm]

            #> Update the coefficient associated with a Pauli-Z basis state labeled by 
            #> `zBitStr` when a Pauli  term P(`xMask`, zMask`, `coeff`) acts on it:
            #> P |zBitStr> = coeff * (-1)^count_ones(zMask & zBitStr) |zBitStr ⊻ xMask>
            termCoeff = isodd(count_ones(zMask & zBitStr)) ? adaptNeg(coeff) : coeff

            #> Sum over coefficients associated with all Pauli terms with the same `xMask`
            updateNeuAccumulator!(accumulator, xMask=>termCoeff)
        end

        for pair in accumulator
            naiveSum, residue = pair.second
            nzRow = reinterpret(Int, xor(pair.first, zBitStr)) + 1
            nzCol = reinterpret(Int, zBitStr) + 1
            coeffSum = Complex{T}(adaptAdd(naiveSum, residue))

            if !iszero(coeffSum)
                #> Entry at `(nzRow, nzCol)` holds coeff-sum of all terms with equal `xMask`
                processor((nzRow, nzCol)=>coeffSum, nzCount)
                nzCount += 1
            end
        end

        nzCount
    end
end

#> At a target column `b` (one-based index) of the storing matrix `m`, a Pauli term 
#> P(`xMask`, `zMask`, `coeff`) only touch one entry: `m[(xMask ⊻ b) + 1, b]`.
function toMatrix(::Type{T}, h::PauliSum{T1}, nSite::Integer=countSites(h); 
                  canonicalization::Bool=true) where {T<:SignedReal, T1<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    nSite = toBitIntData(nSite, "nSite")
    # iszero(nSite) && (return spzeros(Complex{T}, 1, 1))
    order = getPauliOpMatOrder(nSite)

    hLocal = canonicalization ? lazyCanonicalize(h, false) : h
    xMasks, zMasks, coeffs = formatPauliOpMatInfo(hLocal, nSite)

    colPtrs = Vector{Int}(undef, order+1)
    maxSave = adaptMul(order, countTerms(hLocal))
    nzvRows = Vector{Int}(undef, maxSave)
    nzvList = Vector{Complex{T}}(undef, maxSave)
    permBuf = Memory{Int}(undef, maxSave)
    permBuf .= 0

    evaluator = genPauliOpMatEntryEvaluator(T, xMasks, zMasks, coeffs)

    nzCount = 0
    @inbounds for iCol in 0:(order-1)     #> Zero-based column indexer
        zBasisBitStr = UInt(iCol)         #> Bit-string label in Pauli-Z basis for `iCol`
        colPtrs[begin+iCol] = nzCount + 1 #> `colPtrs` must starts with `1`

        nStart = nzCount
        nzCount = evaluator(zBasisBitStr, nzCount) do pair, nCountLocal
            nzvRows[begin+nCountLocal] = first(pair.first)
            nzvList[begin+nCountLocal] = pair.second
        end
        nFinal = nzCount - 1 #> `nzCount` has been updated to the `nStart` for next column

        #> Reformat non-zero-element row indices in an ascending order for column `iCol`
        if nFinal > nStart #>> `iCol` has at least two non-zero entries
            nzRowSector = @view nzvRows[(begin+nStart):(begin+nFinal)]
            nzValSector = @view nzvList[(begin+nStart):(begin+nFinal)]
            permBSector = @view permBuf[begin:(begin+nFinal-nStart)]
            sortperm!(permBSector, nzRowSector)
            nzRowSector .= nzRowSector[permBSector]
            nzValSector .= nzValSector[permBSector]
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

toMatrix(::Type{T}, str::PauliStr, nSite::Integer=countSites(str)) where {T<:SignedReal} = 
toMatrix(T, PauliSum(Int, genMemory(str)), nSite)

toMatrix(h::PauliSum{T}, nSite::Integer=countSites(h); canonicalization::Bool=true
         ) where {T<:SignedReal} = 
toMatrix(T, h, nSite; canonicalization)

toMatrix(str::PauliStr, nSite::Integer=countSites(str)) = 
toMatrix(Int, str, nSite)


function toMatrix!(dst::AbstractMatrix{C}, h::PauliSum{T1}, nSite::Integer=countSites(h); 
                   canonicalization::Bool=true
                   ) where {T<:SignedReal, C<:RealOrComplex{T}, T1<:SignedReal}
    nSite = toBitIntData(nSite, "nSite")
    order = size(dst, 1)
    order == size(dst, 2) || throw(ArgumentError("`dst` must be a square matrix."))
    if getPauliOpMatOrder(nSite) != order
        throw(ArgumentError("The order of `dst` must equal `2^$nSite`."))
    end

    function assign!(pair::Pair{NTuple{2, Int}, Complex{T}}, ::Int)
        (iRow, iCol), matVal = pair
        dst[begin+iRow-1, begin+iCol-1] = matVal
    end

    fill!(dst, zero(C)) #> Reset `dst` as `processMatrixInfo` only updates non-zero entries
    processMatrixInfo(assign!, real(C), h, nSite; canonicalization)

    dst
end

toMatrix!(dst::AbstractMatrix{<:SRealOrComplex}, str::PauliStr, 
          nSite::Integer=countSites(str)) = 
toMatrix!(dst, PauliSum( Int, genMemory(str) ), nSite)

toMatrix!(dst::AbstractMatrix{<:SRealOrComplex}, sym::PauliSym, nSite::Integer) = 
toMatrix!(dst, PauliStr(1, sym), nSite)

function toMatrix!(dst::AbstractMatrix{T}, sym::PauliSym) where {T<:SRealOrComplex}
    if size(dst) != (2, 2)
        throw(ArgumentError("`dst` must be a 2-by-2 matrix."))
    end

    if sym == symY && (T <: Real)
        throw(ArgumentError("`eltype(dst)` must be a subtype of `Complex` for `sym=$sym`."))
    end

    dstVec = vec(dst)

    if sym == symI
        dstVec[begin] = dstVec[end] = one(T)
    elseif sym == symX
        dstVec .= CONSTVAR!!pauliXMatEles
    elseif sym == symY
        dstVec .= CONSTVAR!!pauliYMatEles
    else
        dstVec .= CONSTVAR!!pauliZMatEles
    end

    dst
end


function processMatrixInfo(processor::F, ::Type{T}, h::PauliSum{T1}, 
                           nSite::Integer=countSites(h); canonicalization::Bool=true
                           ) where {T<:SignedReal, T1<:SignedReal, F<:AbstractCallable}
    nSite = toBitIntData(nSite, "nSite")
    order = getPauliOpMatOrder(nSite)

    hLocal = canonicalization ? lazyCanonicalize(h, false) : h
    xMasks, zMasks, coeffs = formatPauliOpMatInfo(hLocal, nSite)
    evaluator = genPauliOpMatEntryEvaluator(T, xMasks, zMasks, coeffs)

    for iCol in 0:(order-1)
        zBasisBitStr = UInt(iCol)
        evaluator(processor, zBasisBitStr, 0)
    end

    hLocal
end


"""
    findAxisPermViolation(tensor::AbstractArray{C, D}, axisPerm::NTuple{D, Int}, 
                          conjugated::Bool) where {T<:Real, C<:Union{Complex{T}, T}, D} -> 
    Union{Nothing, NTuple{D, Int}}

Return the first index `idx` (in column-major order) of `tensor` at which the symmetry 
`tensor[idx...] == tensor[permIdx...]` (or `tensor[idx...] == tensor[permIdx...]'` if 
`conjugated` is `true`) fails, where the "position" of `permIdx` along axis `d` (i.e., the 
index difference to the first index of that axis) equals the position of `idx` along axis 
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
        partner = conjugated ? tensor[permIdx...]' : tensor[permIdx...]
        tensor[carteIdx] != partner && (return idx)
    end

    nothing
end
