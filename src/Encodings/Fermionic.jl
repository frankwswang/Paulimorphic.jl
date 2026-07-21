export checkMajEncoding

"""
    checkMajEncoding(strs::AbstractVector{PauliStr}, explicitError::Bool=false) -> Bool

Return `true` if `strs` forms a valid Majorana-operator encoding: it must contain a 
positive even number `2m` of Pauli strings that all explicitly act on the same number of 
sites (i.e., [`countSites`](@ref) returns the same value), are all Hermitian (i.e., carry a 
real phase), and mutually anticommute (verified by [`checkAntiCom`](@ref)). Such a 
collection realizes a representation of `2m` Majorana operators `γ_1, ..., γ_2m` that are 
involutions (`γ_i^2 == I`) and satisfy the anticommutation relation: 

    γ_i γ_j + γ_j γ_i == 2δ_ij I

When `strs` fails to meet any necessary condition to form such an encoding, the failure is 
reported by returning `false` unless `explicitError=true`, in which case an `ArgumentError` 
identifying the violated condition (and the immediate offending terms) is thrown instead.
"""
function checkMajEncoding(strs::AbstractVector{PauliStr}, explicitError::Bool=false)::Bool
    nMode = length(strs)

    if iszero(nMode) || isodd(nMode)
        if explicitError
            throw(ArgumentError("`length(strs)` should return a positive even number."))
        else
            return false
        end
    end

    nSite = first(strs).n

    for (i, str) in enumerate(strs)
        if str.n != nSite
            if explicitError
                throw(ArgumentError("Every term in `strs` should explicitly act on the "*
                                    "same number of sites. Terms (1, $i) do not."))
            else
                return false
            end
        end

        if str.phase == posImg || str.phase == negImg
            if explicitError
                throw(ArgumentError("Every term in `strs` should be Hermitian (and an "*
                                    "involution). Term $i violates this condition due to "*
                                    "an imaginary phase."))
            else
                return false
            end
        end
    end

    @inbounds for i in 1:(nMode - 1), j in (i + 1):nMode
        if !checkAntiCom(strs[begin+i-1], strs[begin+j-1])
            if explicitError
                throw(ArgumentError("Every pair of distinct terms in `strs` should "*
                                    "anticommute. Terms `($i, $j)` do not."))
            else
                return false
            end
        end
    end

    true
end