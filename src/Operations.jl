export checkCommute, checkAntiCom, evalCommute, evalAntiCom, getFrustrationGraph



"""
    mul(s1::PauliStr, s2::PauliStr) -> PauliStr

Multiply two `PauliStr`, returning the product `s3::PauliStr` with the its associated phase 
folded into `.phase`.
"""
function mul(s1::PauliStr, s2::PauliStr)
    bl1 = iszero(s1.n)
    if bl1 || iszero(s2.n) #> Trivial case: s * I == s, phases combined
        controlStr = bl1 ? s2 : s1
        x3 = controlStr.x
        z3 = controlStr.z
        n3 = controlStr.n 
        phase = PhaseFactor((UInt8(s1.phase) + UInt8(s2.phase)) & 0x3)
    else
        x1, x2 = s1.x, s2.x
        z1, z2 = s1.z, s2.z
        n1, n2 = s1.n, s2.n
        len1, len2 = length(z1), length(z2)
        nWord, n3 = n1 <= n2 ? (len2, n2) : (len1, n1)
        z3 = Memory{UInt}(undef, nWord)
        x3 = Memory{UInt}(undef, nWord)
        nY1 = nY2 = nY3 = 0
        sgnParity = 0

        @inbounds for w in 1:nWord
            #> Pad shorter string with identities
            z1w = w <= len1 ? z1[begin+w-1] : zero(UInt)
            x1w = w <= len1 ? x1[begin+w-1] : zero(UInt)
            z2w = w <= len2 ? z2[begin+w-1] : zero(UInt)
            x2w = w <= len2 ? x2[begin+w-1] : zero(UInt)

            x3w = x1w ⊻ x2w
            z3w = z1w ⊻ z2w
            x3[begin+w-1] = x3w
            z3[begin+w-1] = z3w
            nY1 += count_ones(z1w & x1w)
            nY2 += count_ones(z2w & x2w)
            nY3 += count_ones(z3w & x3w)
            sgnParity += count_ones(x1w & z2w)
        end

        phase = (( Int(UInt8(s1.phase)) + Int(UInt8(s2.phase))
                   + 3*(nY1 + nY2) + nY3 + 2*(sgnParity & 1) ) & 3) |> UInt |> PhaseFactor
    end

    PauliStr(x3, z3, phase, n3)
end


function mul!(s::PauliStr, p::PhaseFactor)
    newPhase = PhaseFactor((UInt8(s.phase) + UInt8(p)) & 3)
    s.phase = newPhase
    s
end


"""
    checkCommute(str1::PauliStr, str2::PauliStr) -> Bool

Return `true` if the Pauli strings `str1` and `str2` commute and `false` if they anticommute
(any two Pauli strings do one or the other). When the two strings span different numbers of 
sites, only the overlapping words are examined — the extra sites from the longer string act 
against implicit identities and does not affect commutation.
"""
function checkCommute(str1::PauliStr, str2::PauliStr)::Bool
    z1, x1 = str1.z, str1.x
    z2, x2 = str2.z, str2.x
    nWord = min(length(z1), length(z2))   #> Shorter string is complemented with identities
    parity = 0
    @inbounds for w in 1:nWord
        parity += count_ones(z1[w] & x2[w]) + count_ones(x1[w] & z2[w])
    end
    iseven(parity)
end


"""
    checkAntiCom(str1::PauliStr, str2::PauliStr) -> Bool

Return `true` if `str1` and `str2` anticommute and `false` if they commute. It is the 
logical negation of [`checkCommute`](@ref).
"""
function checkAntiCom(str1::PauliStr, str2::PauliStr)::Bool
    !checkCommute(str1, str2)
end


function evalCommute(str1::PauliStr, str2::PauliStr)
    checkCommute(str1, str2) ? PauliStr() : PauliSum([1, -1], [str1, str2])
end

function evalAntiCom(str1::PauliStr, str2::PauliStr)
    checkAntiCom(str1, str2) ? PauliStr() : PauliSum([str1, str2])
end


function getFrustrationGraph(ham::PauliSum; 
                             edgeThreshold::Real=0, nodeThreshold::Real=edgeThreshold)
    strs = ham.str
    coeffs = ham.coeff

    validNodes = PauliStr[]
    nodeCoeffs = eltype(coeffs)[]
    for (coeff, str) in zip(coeffs, strs)
        if abs(coeff) > nodeThreshold
            push!(validNodes, str)
            push!(nodeCoeffs, coeff)
        end
    end

    nodeNum = length(validNodes)
    validEdges = NTuple{2, Int}[]

    for i in 1:(nodeNum-1), j in (i+1):nodeNum
        weight = abs(nodeCoeffs[begin+i-1] * nodeCoeffs[begin+j-1])
        weight > edgeThreshold && getFrustrationGraphCore!(validEdges, validNodes, (i, j))
    end

    validNodes => validEdges
end

function getFrustrationGraph(strings::AbstractVector{PauliStr})
    nodeNum = length(strings)

    validEdges = NTuple{2, Int}[]
    for i in 1:(nodeNum-1), j in i+1:nodeNum
        getFrustrationGraphCore!(validEdges, strings, (i, j))
    end

    strings => validEdges
end

function getFrustrationGraphCore!(validEdges::AbstractVector{NTuple{2, Int}}, 
                                  strings::AbstractVector{PauliStr}, 
                                  edge::NTuple{2, Int}) #> One-based index
    i, j = edge
    checkAntiCom(strings[begin+i-1], strings[begin+j-1]) && push!(validEdges, (i, j))
    strings => validEdges
end