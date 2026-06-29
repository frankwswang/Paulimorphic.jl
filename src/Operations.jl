export checkCommute, checkAntiCom, evalCommute, evalAntiCom, getFrustrationGraph

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
    strings = ham.string
     coeffs = ham.coeff
    nodeNum = 0

    validNodes = PauliStr[]
    for (coeff, str) in zip(coeffs, strings)
        if abs(coeff) > nodeThreshold
            nodeNum += 1
            push!(validNodes, str)
        end
    end

    validEdges = NTuple{2, Int}[]
    for i in 1:(nodeNum-1), j in i+1:nodeNum
        weight = abs(coeffs[begin+i-1] * coeffs[begin+j-1])
        weight > edgeThreshold && getFrustrationGraphCore!(validEdges, strings, (i, j))
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