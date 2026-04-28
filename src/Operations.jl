export checkCommute, checkAntiCom, evalCommute, evalAntiCom, getFrustrationGraph

function checkCommute(str1::PauliStr, str2::PauliStr)::Bool
    str1z = str1.zStr
    str1x = str1.xStr
    str2z = str2.zStr
    str2x = str2.xStr
    if length(str1z) == length(str2z)
        dot(str1z, str2x) + dot(str1x, str2z)
    else
        mapreduce(*, +, str1z, str2x) + mapreduce(*, +, str1x, str2z)
    end |> iszero
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
        if weight > edgeThreshold && checkAntiCom(strings[begin+i-1], strings[begin+j-1])
            push!(validEdges, (i, j))
        end
    end

    validNodes => validEdges
end