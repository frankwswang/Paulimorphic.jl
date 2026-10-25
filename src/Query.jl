struct MemoryChain{T, S<:Tuple{ Vararg{Memory} }} <: AbstractVector{T}
    len::Int
    seg::S

    function MemoryChain{T}(segments::Memory...) where {T}
        for (i, seg) in enumerate(segments)
            if !hasmethod(convert, Tuple{Type{T}, eltype(seg)})
                throw(ArgumentError("The element type of $i-th element of `segments` "*
                                    "cannot be converted to `T`."))
            end
        end

        l = mapreduce(length, +, segments, init=0)
        new{T, typeof(segments)}(l, segments)
    end

    function MemoryChain{T}(chain::MemoryChain{T1}) where {T, T1}
        if !hasmethod(convert, Tuple{Type{T}, T1})
            throw(ArgumentError("Cannot convert `chain::MemoryChain{$T1}` to a "*
                                "`MemoryChain{T}`."))
        end

        s = chain.seg
        new{T, typeof(s)}(chain.len, s)
    end
end

MemoryChain(segments::Memory...) = 
MemoryChain{mapreduce(eltype, promote_type, segments, init=Union{})}(segments...)

MemoryChain(chain::MemoryChain{T}) where {T} = MemoryChain{T}(chain)

const EmptyMemChain{T} = MemoryChain{T, Tuple{}}

Base.size(mc::MemoryChain) = (mc.len,)

Base.size(::EmptyMemChain) = (0,)

Base.IndexStyle(::Type{<:MemoryChain}) = IndexLinear()

function formatIndex(mc::MemoryChain{T}, oneBasedIdx::Int) where {T}
    segments = mc.seg
    eleCount = 0
    iSeg, iLocal = 0, 0

    for (i, seg) in enumerate(segments)
        localIdx = oneBasedIdx - eleCount
        segSpace = length(seg)

        if segSpace >= localIdx
            iLocal = localIdx
            iSeg = i
            break
        else
            eleCount += segSpace
        end
    end

    iSeg => iLocal #> Indices of and within the target segment; if found, both are one-based
end

function Base.getindex(mc::MemoryChain{T}, i::Int) where {T}
    @boundscheck checkbounds(mc, i)
    iSeg, iIdx = formatIndex(mc, i)
    rawElement = mc.seg[begin+iSeg-1][begin+iIdx-1]
    convert(T, rawElement)
end

Base.getindex(mc::EmptyMemChain, i::Int) = throw(BoundsError(mc, i))

function Base.setindex!(mc::MemoryChain, val, i::Integer)
    @boundscheck checkbounds(mc, i)
    iSeg, iIdx = formatIndex(mc, i)
    seg = mc.seg[begin+iSeg-1]
    setindex!(seg, val, firstindex(seg)+iIdx-1)
end

function Base.iterate(mc::MemoryChain, idxState::Integer=1)
    if idxState < 1
        throw(ArgumentError("`idxState=$idxState` is not a valid state value."))
    elseif idxState == length(mc)
        nothing
    else
        (mc[begin+idxState-1], idxState+1)
    end
end


"""
    genMemory(::T, obj) -> Memory{T}

    genMemory(obj::AbstractArray{T}) -> Memory{T}

    genMemory(obj::T) -> Memory{T}

Return a `Memory{T}` filled with the (potentially converted) entires from `obj`. If `obj` 
is not an `AbstractArray`, a one-element `Memory{T}` encapsulating `obj` will be returned.
"""
genMemory(::Type{T}, arr::AbstractArray) where {T} = Memory{T}(arr|>vec)

function genMemory(::Type{T}, obj) where {T}
    mem = Memory{T}(undef, 1)
    mem[] = obj
    mem
end

genMemory(arr::AbstractArray{T}) where {T} = genMemory(T, arr)

genMemory(arr::T) where {T} = genMemory(T, arr)


#> Neumaier-sum accumulator: key => (naiveSum, sumResidue)
const NeuAccumulator{K, T<:RealOrComplex} = Dict{K, NTuple{2, T}}

function updateNeuAccumulator!(dict::NeuAccumulator{<:Any, T}, 
                               data::Pair) where {T<:RealOrComplex}
    zeroVal = zero(T)
    key, value = data
    res = get(dict, key, nothing)
    keyAdded = isnothing(res)
    naiveSum, sumResidue = keyAdded ? (zeroVal, zeroVal) : res
    newSumPair = neumaierAdd(naiveSum, T(value), sumResidue)
    dict[key] = newSumPair
    keyAdded
end


#> Can be used for array copying when `a .= b` fails for two arrays with mismatched axes
function dumpTo!(dst::AbstractArray, src, iStart::Integer=firstindex(dst))
    if src isa AbstractArray
        copyto!(dst, iStart, src, firstindex(src), length(src))
    else
        fill!(dst, src)
    end
end
