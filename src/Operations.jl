export add, mul, scale!, checkCommute, checkAntiCom, evalCommute, evalAntiCom, toAdjoint

const PauliStrOrSum = Union{PauliStr, PauliSum}

const PauliStrToVal{T<:Real} = Pair{PauliStr, <:RealOrComplex{T}}

"""
    add(::Type{T}, str1::PauliStr, str2::PauliStr, 
        simplification::Bool=true) where {T<:Real} -> PauliSum{T}

    add(str1::PauliStr, str2::PauliStr, simplification::Bool=true) -> PauliSum{Int}

    str1 + str2 -> PauliSum{Int}

Add two `PauliStr`, returning their sum in which each string initially carries the nominal 
coefficient `one(Complex{T})` before its phase is absorbed (see the [`PauliSum`](@ref) 
constructors). When `T` is unspecified (in the case of the second signature), it defaults 
to `Int`; `T=Bool` is disallowed for the same reason as in the `PauliSum` constructor. If 
`simplification=true`, the same simplification procedure used by the `PauliSum` constructor 
(when the same-named argument is set to `true`) is applied to the result. Neither input is 
mutated, and the result does not reference any data in either input. Using the binary 
operator `+` is equivalent to invoking the second method signature (with the default 
value of `simplification`).
"""
add(::Type{T}, str1::PauliStr, str2::PauliStr, simplification::Bool=true) where {T<:Real} = 
PauliSum(T, [str1, str2], simplification)

add(str1::PauliStr, str2::PauliStr, simplification::Bool=true) = 
add(Int, str1, str2, simplification)

"""
    add(h1::PauliSum{T1}, h2::PauliSum{T2}, 
        simplification::Bool=true) where {T1<:Real, T2<:Real} -> 
    PauliSum{promote_type(T1, T2)}

    h1 + h2 -> PauliSum{promote_type(T1, T2)}

Add two `PauliSum`, returning their sum with the coefficient type automatically promoted. 
If `simplification=true`, the same simplification procedure used by the [`PauliSum`](@ref) 
constructor (when the same-named argument is set to `true`) is applied to the result. The 
result does not reference any data in either `h1` or `h2`.
"""
add(h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true) where 
   {T1<:Real, T2<:Real} = 
PauliSum(vcat(h1.str, h2.str), vcat(h1.coeff, h2.coeff), simplification)

"""
    add(h::PauliSum{T1}, term::Pair{PauliStr, <:Union{Complex{T2}, T2}}, 
        simplification::Bool=true) where {T1<:Real, T2<:Real} -> 
    PauliSum{promote_type(T1, T2)}

    h + term -> PauliSum{promote_type(T1, T2)}
    term + h -> PauliSum{promote_type(T1, T2)}

Add a coefficient-carrying `term` (e.g., as returned by [`indexTerm`](@ref)) to `h`, 
returning the sum with the coefficient type automatically promoted. The added string is 
`term.first`, whose phase is absorbed into its associated coefficient `term.second` by the 
`PauliSum` constructor. If `simplification=true`, the same simplification procedure used by 
the [`PauliSum`](@ref) constructor (when the same-named argument is set to `true`) is 
applied to the result. The result does not reference any data in either `h` or `term`.
"""
add(h::PauliSum{T1}, term::PauliStrToVal{T2}, simplification::Bool=true) where 
   {T1<:Real, T2<:Real} = 
PauliSum(vcat(h.str, term.first), vcat(h.coeff, term.second), simplification)

"""
    add(h::PauliSum{T}, s::PauliStr, simplification::Bool=true) where {T<:Real} -> 
    PauliSum{T}

    h + s -> PauliSum{T}
    s + h -> PauliSum{T}

Add a single `PauliStr` to `h`, i.e., shorthand for `add(h, s=>one(T), simplification)`: 
`s` is appended as a new term with its phase folded into the associated coefficient. If 
`simplification=true`, the same simplification procedure used by the `PauliSum` 
constructor (when the same-named argument is set to `true`) is applied to the result. The 
result does not reference any data in either `h` or `s`.
"""
add(h::PauliSum{T}, s::PauliStr, simplification::Bool=true) where {T<:Real} = 
add(h, s=>one(T), simplification)

Base.:+(h::PauliSum, obj::Union{PauliStrToVal, PauliStrOrSum}) = add(h, obj)
Base.:+(obj::Union{PauliStrToVal, PauliStr}, h::PauliSum) = add(h, obj)
Base.:+(s1::PauliStr, s2::PauliStr, s3::PauliStr...) = PauliSum(Int, [s1, s2, s3...])


"""
    mul(str::PauliStr, phase::PhaseFactor) -> PauliStr

    str * phase -> PauliStr
    phase * str -> PauliStr

Multiply a `PauliStr` by a `PhaseFactor`, returning a new `PauliStr`. The value of `phase` 
is folded into the result's field `.phase`. The result does not reference any data in `str`.
"""
mul(str::PauliStr, phase::PhaseFactor) = PauliStr(str, str.n, mul(str.phase, phase))

"""
    mul(str1::PauliStr, str2::PauliStr) -> PauliStr

    str1 * str2 -> PauliStr

Multiply two `PauliStr`, returning their product `s3 == str1 * str2` with its associated 
phase folded into `s3.phase`. When `str1` and `str2` explicitly act on different numbers 
of sites (i.e., `str1.n != str2.n`), the string with the smaller site count is temporarily 
promoted per the implicit identity-padding convention (see [`PauliStr`](@ref)), and `s3` 
explicitly acts on the larger site count; in particular, a zero-site operand contributes 
only its phase. Neither input is mutated, and the result does not reference any data in 
either input.
"""
function mul(str1::PauliStr, str2::PauliStr)
    bl1 = iszero(str1.n)
    if bl1 || iszero(str2.n) #> Trivial case: s * I == s, phases combined
        controlStr = bl1 ? str2 : str1
        x3 = controlStr.x
        z3 = controlStr.z
        n3 = controlStr.n
        phase = mul(str1.phase, str2.phase)
        PauliStr(x3, z3, phase, n3)
    else
        x1, x2 = str1.x, str2.x
        z1, z2 = str1.z, str2.z
        n1, n2 = str1.n, str2.n
        len1, len2 = length(z1), length(z2)
        nWord, n3 = n1 < n2 ? (len2, n2) : (len1, n1)
        s3 = PauliStr(n3)
        z3, x3 = s3.z, s3.x
        nY1 = nY2 = nY3 = 0
        nXZ = 0

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
            nXZ += count_ones(x1w & z2w)
        end

        imXpn = (Int(str1.phase) + Int(str2.phase) - 2nXZ - nY1 - nY2 + nY3)
        s3.phase = PhaseFactor(imXpn & 3)
        s3
    end
end

"""
    mul(str::PauliStr, coeff::Union{Real, Complex}, simplification::Bool=true) -> PauliSum

    str * coeff -> PauliSum
    coeff * str -> PauliSum

Multiply a `PauliStr` by a real or complex scalar `coeff`, returning the product 
`str * coeff` (equal to `coeff * str`). Unlike multiplication by a [`PhaseFactor`](@ref), a 
general scalar coefficient may not be stored within a `PauliStr`, so the result is always 
promoted to a `PauliSum`. If `simplification=true`, the same simplification procedure used 
by the `PauliSum` constructor (when the same-named argument is set to `true`) is applied to 
the result. The result does not reference any data in `str`.
"""
mul(str::PauliStr, coeff::RealOrComplex, simplification::Bool=true) = 
PauliSum([str], coeff, simplification)

"""
    mul(s::PauliStr, h::PauliSum, simplification::Bool=true) -> PauliSum
    mul(h::PauliSum, s::PauliStr, simplification::Bool=true) -> PauliSum

    s * h -> PauliSum
    h * s -> PauliSum

Multiply a `PauliStr` with a `PauliSum`, returning their product. Each stored string of `h` 
is multiplied by `s` on the matching side while the coefficients are carried over. If 
`simplification=true`, the same simplification procedure used by the `PauliSum` constructor 
(when the same-named argument is set to `true`) is applied to the result. The result does 
not reference any data in either `s` or `h`.
"""
function mul(s::PauliStr, h::PauliSum, simplification::Bool=true)
    newStrs = map(ele->mul(s, ele), h.str)
    PauliSum(newStrs, h.coeff, simplification)
end

function mul(h::PauliSum, s::PauliStr, simplification::Bool=true)
    newStrs = map(ele->mul(ele, s), h.str)
    PauliSum(newStrs, h.coeff, simplification)
end

"""
    mul(h1::PauliSum{T1}, h2::PauliSum{T2}, 
        simplification::Bool=true) where {T1<:Real, T2<:Real} -> 
    PauliSum{promote_type(T1, T2)}

    h1 * h2 -> PauliSum{promote_type(T1, T2)}

Multiply `h1` by `h2`, returning their product. If `simplification=true`, the same 
simplification procedure used by the `PauliSum` constructor (when the same-named argument 
is set to `true`) is applied to the result. The result does not reference any data in 
either `h1` or `h2`.
"""
function mul(h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
             ) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)
    cL, sL = h1.coeff, h1.str
    cR, sR = h2.coeff, h2.str
    m, n = length(cL), length(cR)

    cs = Memory{Complex{T}}(undef, m * n)
    ss = Memory{PauliStr}(undef, m * n)
    k = 0
    @inbounds for j in 1:n, i in 1:m
        k += 1
        cs[begin+k-1] =     cL[begin+i-1] * cR[begin+j-1]
        ss[begin+k-1] = mul(sL[begin+i-1],  sR[begin+j-1]) #> Phase folded into `.phase`
    end

    PauliSum(ss, cs, simplification)
end


"""
    mul(h::PauliSum, coeff::Union{Real, Complex}, simplification::Bool=true) -> PauliSum

    h * coeff -> PauliSum
    coeff * h -> PauliSum

Multiply `h` by a coefficient `coeff`, returning a new `PauliSum` whose coefficient type 
is automatically promoted. When `simplification=true`, the result is fully canonicalized; 
in particular, scaling by an exact zero returns an empty `PauliSum` as the zero operator. 
For in-place scaling without type promotion, see [`scale!`](@ref).
"""
mul(h::PauliSum, coeff::RealOrComplex, simplification::Bool=true) = 
PauliSum(h.str, h.coeff .* coeff, simplification)


"""
    mul(h::PauliSum, phase::PhaseFactor, simplification::Bool=true) -> PauliSum

    h * phase -> PauliSum
    phase * h -> PauliSum

Multiply `h` by a phase `phase`, returning a new `PauliSum` in the canonical form.
"""
mul(h::PauliSum, phase::PhaseFactor, simplification::Bool=true) = 
mul(h, evalPhase(phase), simplification)


Base.:*(op::PauliStrOrSum, num::PhaseOrCoeff) = mul(op, num)
Base.:*(num::PhaseOrCoeff, op::PauliStrOrSum) = mul(op, num)
Base.:*(op1::PauliStrOrSum, op2::PauliStrOrSum) = mul(op1, op2)


"""
    scale!(s::PauliStr, p::PhaseFactor) -> PauliStr

Multiply `s` in place by a phase `p` (absorbed into `s.phase`), returning the mutated `s`.
"""
function scale!(s::PauliStr, p::PhaseFactor)
    newPhase = PhaseFactor((UInt8(s.phase) + UInt8(p)) & 3)
    s.phase = newPhase
    s
end

"""
    scale!(h::PauliSum, p::PhaseFactor) -> PauliSum

Multiply `h` in place by a phase `p`, returning the mutated `h`. `p` is first converted to 
[`evalPhase`](@ref)`(p)`, then absorbed into `h.coeff`. This is to match the canonical form 
of `PauliSum` in which its phases are carried by coefficients rather than the stored 
`PauliStr`.
"""
scale!(h::PauliSum, p::PhaseFactor) = UInt8(p) > 0 ? scale!(h, evalPhase(p)) : h

"""
    scale!(h::PauliSum, c::Union{Real, Complex}) -> PauliSum

Multiply `h` in place by a coefficient `c`, returning the mutated `h`. `h.coeff` is scaled 
in place, thus each product `h.coeff[i] * c` must be able to be converted to 
`eltype(h.coeff)`. No re-canonicalization is performed.
"""
scale!(h::PauliSum, c::RealOrComplex) = (h.coeff .*= c; h)


"""
    checkCommute(str1::PauliStr, str2::PauliStr) -> Bool

Return `true` if the Pauli strings `str1` and `str2` commute and `false` if they 
anticommute (any two Pauli strings do one or the other). When the two strings explicitly 
act on different numbers of sites, commutation is evaluated under the implicit 
identity-padding convention (see [`PauliStr`](@ref)): the sites of the longer string 
beyond the shorter one's site count act against identities and never affect the result. 
The phases of both strings are also irrelevant to the result.
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


"""
    evalCommute(str1::PauliStr, str2::PauliStr) -> PauliSum{Int}

Return the commutator [`str1`, `str2`] = `str1`*`str2` - `str2`*`str1` as a `PauliSum`. 
The multiplications within the commutation follow the implicit identity-padding convention 
of [`mul(str1::PauliStr, str2::PauliStr)`](@ref), so when the two strings explicitly act on 
different numbers of sites, the returned sum explicitly acts on the larger site count. When 
the commutator is zero (i.e., when `str1` and `str2` commute), this function returns an 
empty `PauliSum` as the zero operator.
"""
function evalCommute(str1::PauliStr, str2::PauliStr)
    prod1 = mul(str1, str2)
    prod2 = mul(str2, str1)
    prod1 == prod2 ? PauliSum(Int) : PauliSum(Int, [prod1, scale!(prod2, PhaseFactor(2))])
end

"""
    evalCommute(h1::PauliSum{T1}, h2::PauliSum{T2}) where {T1<:Real, T2<:Real} -> 
    PauliSum{promote_type(T1, T2)}

Return the commutator [`h1`, `h2`] = `h1`*`h2` - `h2`*`h1` as a `PauliSum`. The 
multiplications within the commutation follow [`mul(h1::PauliSum, h2::PauliSum)`](@ref). 
When the commutator is zero, this function returns an empty `PauliSum` as the zero 
operator.
"""
function evalCommute(h1::PauliSum{T1}, h2::PauliSum{T2}) where {T1<:Real, T2<:Real}
    prod1 = mul(h1, h2)
    prod2 = mul(h2, h1)
    PauliSum(vcat(prod1.str, prod2.str), vcat(prod1.coeff, -prod2.coeff))
end


"""
    evalAntiCom(str1::PauliStr, str2::PauliStr) -> PauliSum{Int}

Return the anticommutator {`str1`, `str2`} = `str1`*`str2` + `str2`*`str1` as a `PauliSum`. 
The multiplications within the anticommutation follow the implicit identity-padding 
convention of [`mul(str1::PauliStr, str2::PauliStr)`](@ref), so when the two strings 
explicitly act on different numbers of sites, the returned sum explicitly acts on the 
larger site count. When the anticommutator is zero (i.e., when `str1` and `str2` 
anticommute), this function returns an empty `PauliSum` as the zero operator.
"""
function evalAntiCom(str1::PauliStr, str2::PauliStr)
    prod1 = mul(str1, str2)
    prod2 = mul(str2, str1)
    prod1 == prod2 ? PauliSum(Int, [prod1, prod2]) : PauliSum(Int)
end

"""
    evalAntiCom(h1::PauliSum{T1}, h2::PauliSum{T2}) where {T1<:Real, T2<:Real} -> 
    PauliSum{promote_type(T1, T2)}

Return the anticommutator {`h1`, `h2`} = `h1`*`h2` + `h2`*`h1` as a `PauliSum`. The 
multiplications within the anticommutation follow 
[`mul(h1::PauliSum, h2::PauliSum)`](@ref). When the anticommutator is zero, this function 
returns an empty `PauliSum` as the zero operator.
"""
function evalAntiCom(h1::PauliSum{T1}, h2::PauliSum{T2}) where {T1<:Real, T2<:Real}
    prod1 = mul(h1, h2)
    prod2 = mul(h2, h1)
    PauliSum(vcat(prod1.str, prod2.str), vcat(prod1.coeff, prod2.coeff))
end


"""
    toAdjoint(str::PauliStr) -> PauliStr

    str' -> PauliStr

Return the Hermitian adjoint (namely the Hermitian conjugate) of `str`. The returned string 
carries the same single-site Pauli operators as `str` with the phase conjugated and does 
not reference any data in `str`.
"""
function toAdjoint(str::PauliStr)
    phaseConj = PhaseFactor((0x4 - UInt8(str.phase)) & 0x3) #> conj(im^k) == im^(4-k mod 4)
    PauliStr(str, str.n, phaseConj)
end

"""
    toAdjoint(h::PauliSum{T}) where {T<:Real} -> PauliSum{T}

    h' -> PauliSum

Return the Hermitian adjoint (namely the Hermitian conjugate) of `h`. The result is in the 
canonical form and does not reference any data in `h`. For an input `h` already in its 
canonical form (e.g., as produced by the constructor of [`PauliSum`](@ref)), the adjoint is 
an involution: 

```jldoctest
julia> h = PauliSum([pauli"XXIXX", pauli"XXIYX"], [1, im]);

julia> (h')' == h
true
```
"""
toAdjoint(h::PauliSum) = PauliSum(map(toAdjoint, h.str), map(conj, h.coeff))

Base.adjoint(op::PauliStrOrSum) = toAdjoint(op)