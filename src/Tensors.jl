"""
    isIndexLabel(candidate::Tuple{Integer, Vararg{Integer}}, 
                 iStart::Union{Integer, Missing}=1) -> Bool

Return `true` if `candidate` is considered a valid integer-index label. In other words, it 
should obey the following canonical form: every element must be an `Integer` no larger than 
one plus the maximum of all its preceding elements; the first element also must equal 
`iStart` when `iStart` is not `missing`. This form admits exactly one unique integer label 
per equality pattern, so every two elements of `candidate` are equal if and only if the 
objects they index are considered the same object. If `candidate` violates this canonical 
form, `isIndexLabel` returns `false`.
"""
function isIndexLabel(candidate::NonEmptyTuple{Integer}, iStart::MissingOr{Integer}=1)::Bool
    labelMax = ismissing(iStart) ? first(candidate) : (iStart - 1)

    for i in eachindex(candidate)
        label = candidate[i]
        bl = ismissing(iStart) ? (label <= labelMax+1) : (iStart <= label <= labelMax+1)
        bl || (return false)
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
