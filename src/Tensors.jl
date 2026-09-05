export findAxisPermViolation

public isIndexLabel

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
    findAxisPermViolation(tensor::AbstractArray{C, D}, axisPerm::NTuple{D, Int}, 
                          conjugated::Bool) where {T<:Real, C<:Union{Complex{T}, T}, D} -> 
    Union{Nothing, NTuple{D, Int}}

Return the first index `idx` (in column-major order) of `tensor` at which the symmetry 
`tensor[idx...] == tensor[permIdx...]` (or `tensor[idx...] == tensor[permIdx...]'` if 
`conjugated` is `true`) fails, where `permIdx` is generated from `idx` by `axisPerm`. If no 
such `idx` is found (i.e., the symmetry holds everywhere), `findAxisPermViolation` returns 
`nothing`. `axisPerm` must be a permutation of `1:D` such that 
`Base.isperm(axisPerm) == true` and only maps between axes of equal extent.
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
