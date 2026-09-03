function checkSpinSectoredEnc((sec1Enc, sec2Enc)::NTuple{2, PairwiseSumEnc}, 
                              explicitError::Bool=false)::Bool
    if sec1Enc === sec2Enc || sec1Enc == sec2Enc
        if explicitError
            throw(ArgumentError("`sec1Enc` and `sec2Enc` should not be equal, as they "*
                                "correspond to separate spin sectors."))
        else
            false
        end
    else
        for (encLocal, title) in ((sec1Enc, "sec1Enc"), (sec2Enc, "sec2Enc"))
            if length(encLocal.first) != length(encLocal.second)
                if explicitError
                    throw(ArgumentError("`$title.first` should have the same length as "*
                                        "`$title.second`."))
                else
                    return false
                end
            end
        end
        compEnc = vcat(sec1Enc.first, sec2Enc.first) => vcat(sec1Enc.second, sec2Enc.second)
        checkDiracEnc(compEnc, explicitError)
    end
end


struct PairedOrder end #> (1', 1, 2', 2, ..., N', N)

struct NormalOrder end #> (1', 2', ..., N', N, ..., 2, 1)

const NBodyOpFormat = Union{PairedOrder, NormalOrder}


#> `modeIdxConfig` indexing style: spinSec -> (enc[spinSec][m]', enc[spinSec][n])
#> spinSec only takes two values to represent the two eigenfunctions of spin-1/2 systems
function genNBodyOperator(format::NBodyOpFormat, enc::NTuple{2, PairwiseSumEnc}, 
                          spinSecConfig::NonEmptyTuple{Bool, M}, 
                          modeIdxConfig::NonEmptyTuple{NTuple{2, Integer}, M}, 
                          checkEncoding::Bool=true)::PauliSum where {M}
    checkEncoding && checkSpinSectoredEnc(enc, true)

    if format isa PairedOrder
        mapreduce(*, spinSecConfig, modeIdxConfig) do isSec2, idxPair
            iCre, iAnn = idxPair #> `iCre` and `iAnn` belong to the same particle
            spinSec = enc[begin+isSec2]
            spinSec.second[begin+iCre-1] * spinSec.first[begin+iAnn-1]
        end
    else #> `format isa NormalOrder`
        mapreduce(*, (true, false), (0:(+1):M, M:(-1):0)) do isCreOp, particlePtr
            mapfoldl(*, particlePtr) do offset
                isSec2 = spinSecConfig[begin+offset]
                iOpPair = modeIdxConfig[begin+offset]
                ops = enc[begin+isSec2][begin+isCreOp]
                ops[begin+iOpPair[end-isCreOp]-1]
            end
        end
    end
end


function getOrbSecLabel(spinIdxFirstKeys::NonEmptyTuple{Tuple{Bool, Integer}, M}) where {M}
    buffer = ntuple(_->0, Val(M+1))
    header = 0

    for offset in 0:M
        keyTarget = spinIdxFirstKeys[begin+offset]
        matchedIdx = 0

        for idx in 1:offset
            keyLabeled = spinIdxFirstKeys[begin+idx-1]

            if keyTarget == keyLabeled
                matchedIdx = idx
                break
            end
        end

        label = iszero(matchedIdx) ? (header += 1) : buffer[begin+matchedIdx-1]
        buffer = buffer .+ ntuple(i->ifelse(i==offset+1, label, 0), Val(M+1))
    end

    buffer
end


function genNBodyOperatorSum(format::NBodyOpFormat, enc::NTuple{2, PairwiseSumEnc}, 
                             orbInte::AbstractArray{T, D}, 
                             spinSecConfig::NonEmptyTuple{Bool, M}, 
                             idxPairSymm::NonEmptyTuple{Bool}=ntuple(_->false, Val(M+1)); 
                             particleExch::Bool=true, 
                             hermiticity::Bool=true, 
                             iModeStart::NonEmptyTuple{Integer, M}=ntuple(_->1, Val(M+1)), 
                             checkInput::Bool=true) where {T<:RealOrComplex, D, M}
    N = M + 1
    D == 2N || throw(ArgumentError("`ndims(orbInte)` must equal `2M+2==$(2N)`."))
    if any(i <= 0 for i in iModeStart)
        throw(DomainError(iModeStart, "All elements of `iModeStart` must be positive."))
    end

    orbSecLabel = getOrbSecLabel(map(tuple, spinSecConfig, iModeStart))

    if checkInput
        checkSpinSectoredEnc(enc, true)
        inteShape = size(orbInte)
        orbSecExtents = ntuple(i->inteShape[begin+2i-2], Val(N))

        if particleExch
            for q in 2:N, p in 1:(q-1)
                if spinSecConfig[begin+p-1] == spinSecConfig[begin+q-1] &&
                     orbSecLabel[begin+p-1] !=   orbSecLabel[begin+q-1]
                    pHead, qHead = iModeStart[begin+p-1], iModeStart[begin+q-1]
                    pTail = pHead + size(orbInte, 2p-1) - 1
                    qTail = qHead + size(orbInte, 2q-1) - 1
                    if max(pHead, qHead) <= min(pTail, qTail)
                        throw(ArgumentError("Particles `$p` and `$q` have different mode "*
                                            "(index) windows, $(pHead:pTail) and "*
                                            "$(qHead:qTail), under the same spin sector. "*
                                            "When `particleExch=true`, these two windows "*
                                            "should have no index overlap."))
                    end
                end
            end
        end

        checkNBodyInteTensor(orbInte, orbSecExtents, orbSecLabel, idxPairSymm, true; 
                             particleExch, hermiticity)

        #> After the shape of `orbInte` is verified to be consistent with `orbSecExtents`
        for (n, isSec2, idx) in zip(1:N, spinSecConfig, eachindex(iModeStart))
            iStart = iModeStart[idx]
            windowSize = length(enc[begin+isSec2].first) - iStart + 1
            nOrb = orbSecExtents[begin+n-1]
            if nOrb > windowSize
                throw(ArgumentError("The window size (bounded by `iModeStart[$idx]`) for "*
                                    "`enc` is $windowSize. It is not large enough to be "*
                                    "associated with the $n-th axis of `orbInte`, which "*
                                    "has an extent of $nOrb."))
            end
        end
    end

    realT = (typeof∘inv∘one∘real)(T)
    coreT = Complex{realT}
    cache = Dict{PauliStr, coreT}()

    prefactor = one(realT)
    if particleExch
        for c in 1:maximum(orbSecLabel)
            prefactor /= (realT∘factorial∘count)(isequal(c), orbSecLabel)
        end
    end
    iFirstAxial = map(first, axes(orbInte))

    for carteIdx in CartesianIndices(orbInte)
        inteCoeff = orbInte[carteIdx]
        iszero(inteCoeff) && continue

        idxTuple = Tuple(carteIdx)
        iPairs = map(ntuple(identity, Val(N)), iModeStart) do i, iStart
            m, n = (2i - 1), 2i
            offset = (iStart, iStart) .- (iFirstAxial[begin+m-1], iFirstAxial[begin+n-1])
            offset .+ (idxTuple[begin+m-1], idxTuple[begin+n-1])
        end

        op = genNBodyOperator(format, enc, spinSecConfig, iPairs, false)

        for (str, encCoeff) in zip(op.str, op.coeff)
            opCoeff = prefactor * encCoeff * inteCoeff
            cache[str] = get(cache, str, zero(coreT)) + coreT(opCoeff)
        end
    end

    PauliSum((collect∘keys)(cache), (collect∘values)(cache))
end

function gen1BodyOperatorSum(oneSecEnc::PairwiseSumEnc, orbInte::AbstractArray{T, D}, 
                             idxPairSymm::Bool=false; hermiticity::Bool=true, 
                             iModeStart::Integer=1, checkInput::Bool=true) where 
                            {T<:RealOrComplex, D}
    if checkInput
        checkDiracEnc(oneSecEnc, true)

        nOrb = size(orbInte, 1)
        checkNBodyInteTensor(orbInte, (nOrb,), (1,), (idxPairSymm,), true; hermiticity)

        windowSize = length(oneSecEnc.first) - iModeStart + 1
        if nOrb > windowSize
            throw(ArgumentError("The window size (bounded by `iModeStart`) for "*
                                "`oneSecEnc` is $windowSize. It is not large enough to be "*
                                "associated with the first axis of `orbInte`, which has "*
                                "an extent of $nOrb."))
        end
    end

    genNBodyOperatorSum(PairedOrder(), (oneSecEnc, oneSecEnc), orbInte, (false,); 
                        hermiticity, iModeStart=(iModeStart,), checkInput=false)
end


function formatSpinSecEnc(enc::PairwiseSumEnc, spinOrbNumPair::NTuple{2, Integer})
    checkDiracEnc(enc, true)
    nOp = length(enc.first)
    if any(x->x<0, spinOrbNumPair)
        throw(DomainError(spinOrbNumPair, "Both elements in `spinOrbNumPair` should be "*
                                          "non-negative."))
    end
    nSpinOrb = sum(spinOrbNumPair)
    nOp < nSpinOrb && throw(ArgumentError("`enc` must support at least $nSpinOrb modes."))

    annOps, creOps = enc
    s1 = first(spinOrbNumPair)
    s2 = nOp - nSpinOrb
    spin1Enc = (@view annOps[ begin:(begin+s1-1)]) => (@view creOps[ begin:(begin+s1-1)])
    spin2Enc = (@view annOps[(begin+s1):(end-s2)]) => (@view creOps[(begin+s1):(end-s2)])
    (spin1Enc, spin2Enc)
end

function formatSpinSecEnc(enc::NTuple{2, PairwiseSumEnc}, 
                          spinOrbNumPair::NTuple{2, Integer})
    checkSpinSectoredEnc(enc, true)

    for (nOrb, sec, title) in zip(spinOrbNumPair, enc, ("first", "last"))
        nOp = length(sec.first)
        if nOp < nOrb
            throw(ArgumentError("`length($title(enc).first)=$nOp` should be no less than "*
                                "$nOrb."))
        end
    end

    enc
end


const MolInteTensor1B2B{T<:RealOrComplex, T1<:AbstractMatrix{T}, T2<:AbstractArray{T, 4}} = 
      Tuple{T1, T2}

const OptSpinSectoredEnc = Union{NTuple{2, PairwiseSumEnc}, PairwiseSumEnc}


function encodeElecHam(format::NBodyOpFormat, enc::OptSpinSectoredEnc, 
                       inte1B2BSpin1::MolInteTensor1B2B{T}, 
                       inte1B2BSpin2::MolInteTensor1B2B{T}, 
                       inte2BCross::AbstractArray{T, 4}; 
                       hermiticity::Bool=true, 
                       idxPairSymm::NTuple{2, Bool}=ntuple(_->(hermiticity && T<:Real), 2)
                       ) where {T<:RealOrComplex}
    nOrbSpin1, nOrbSpin2 = size.(Ref(inte2BCross), (1, 3))
    crossERIShape = (nOrbSpin1, nOrbSpin1, nOrbSpin2, nOrbSpin2)
    if size(inte2BCross) != crossERIShape
        throw(ArgumentError("The size of `inte2BCross` should be $crossERIShape."))
    end

    checkInput = false
    secEnc = formatSpinSecEnc(enc, (nOrbSpin1, nOrbSpin2))
    inte1BSpin1, inte2BSpin1 = formatSpinSecMolInteData(
        format, inte1B2BSpin1, nOrbSpin1; idxPairSymm=first(idxPairSymm), hermiticity)
    inte1BSpin2, inte2BSpin2 = formatSpinSecMolInteData(
        format, inte1B2BSpin2, nOrbSpin2; idxPairSymm=last(idxPairSymm),  hermiticity)

    #> Check cross-section two-body integrals
    if inte2BCross != inte2BSpin1 && inte2BCross != inte2BSpin2
        checkNBodyInteTensor(inte2BCross, (nOrbSpin1, nOrbSpin2), (1, 2), idxPairSymm, 
                             true; hermiticity)
    end

    sec1Symm, sec2Symm = idxPairSymm

    h1Spin1 = gen1BodyOperatorSum(first(secEnc), inte1BSpin1, false, sec1Symm; 
                                  hermiticity, checkInput)
    h1Spin2 = gen1BodyOperatorSum( last(secEnc), inte1BSpin2, true,  sec2Symm; 
                                  hermiticity, checkInput)

    h2Spin1 = genNBodyOperatorSum(format, secEnc, inte2BSpin1, (false, false), idxPairSymm; 
                                  hermiticity, checkInput)
    h2Spin2 = genNBodyOperatorSum(format, secEnc, inte2BSpin2, (true , true ), idxPairSymm; 
                                  hermiticity, checkInput)

    h2Cross = genNBodyOperatorSum(format, secEnc, inte2BCross, (false, true ), idxPairSymm; 
                                  hermiticity, checkInput)

    h1Spin1 + h1Spin2 + h2Spin1 + h2Spin2 + h2Cross
end

encodeElecHam(format::NBodyOpFormat, enc::OptSpinSectoredEnc, 
              orbInte1B2B::MolInteTensor1B2B{T}; hermiticity::Bool=true, 
              idxPairSymm::NTuple{2, Bool}=ntuple(_->(hermiticity && T<:Real), 2)) where 
             {T<:RealOrComplex} = 
encodeElecHam(format, enc, orbInte1B2B, orbInte1B2B, last(orbInte1B2B); 
              hermiticity, idxPairSymm)


function formatSpinSecMolInteData(::NormalOrder, spatialInteData::MolInteTensor1B2B{T}, 
                                  nSpinOrb::Integer=size(first(spatialInteData), 1); 
                                  hermiticity::Bool=true, 
                                  idxPairSymm::Bool=(hermiticity && T<:Real)) where 
                                 {T<:RealOrComplex}
    inte1B, inte2B = spatialInteData
    nOrbPair = (nSpinOrb, nSpinOrb)
    oneBodyPairSymm = (idxPairSymm,)
    twoBodyPairSymm = (idxPairSymm, idxPairSymm)

    checkNBodyInteTensor(inte1B, (nSpinOrb,), (1,  ), oneBodyPairSymm, true; hermiticity)
    checkNBodyInteTensor(inte2B,  nOrbPair,   (1, 1), twoBodyPairSymm, true; hermiticity)

    spatialInteData
end

function formatSpinSecMolInteData(::PairedOrder, spatialInteData::MolInteTensor1B2B{T}, 
                                  nSpinOrb::Integer=size(first(spatialInteData), 1); 
                                  hermiticity::Bool=true, 
                                  idxPairSymm::Bool=(hermiticity && T<:Real)) where 
                                 {T<:RealOrComplex}
    inte1B, inte2B = formatSpinSecMolInteData(NormalOrder(), spatialInteData, nSpinOrb; 
                                              idxPairSymm, hermiticity)
    eleT = (typeof∘inv∘one)(eltype(inte1B))
    newInte1B = similar(inte1B, eleT, size(inte1B))
    offset = nSpinOrb - 1
    prefactor = inv(2|>eleT)

    for j in 0:offset, i in 0:offset
        residue = zero(eleT)
        for k in 0:offset #> No residue contribution from cross-spin two-body integrals
            residue += inte2B[begin+i, begin+k, begin+k, begin+j]
        end
        newInte1B[begin+i, begin+j] = inte1B[begin+i, begin+j] - prefactor * residue
    end

    (newInte1B, inte2B)
end
