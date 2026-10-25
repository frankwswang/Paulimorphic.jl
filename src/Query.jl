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
    genMemory(obj) -> Memory

Return a `Memory{T}` filled with the entires from `obj::AbstractArray{T}`. If `obj::T` is 
not an `AbstractArray`, a one-element `Memory{T}` encapsulating `obj` will be returned.
"""
genMemory(arr::AbstractArray) = Memory{eltype(arr)}(arr|>vec)

function genMemory(obj::T) where {T}
    mem = Memory{T}(undef, 1)
    mem[] = obj
    mem
end