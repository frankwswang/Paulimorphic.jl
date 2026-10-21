export scale!, checkCommute, checkAntiCom, evalCommute, evalAntiCom

public add, sub, mul, toAdjoint

const PauliStrOrSum = Union{PauliStr, PauliSum}

const PStrCoeffPair{T<:SignedReal} = Pair{PauliStr, <:SRealOrComplex{T}}

"""
    add(::Type{T}, str1::PauliStr, str2::PauliStr, simplification::Bool=true
        ) where {T<:$SignedReal} -> PauliSum{T}

    add(str1::PauliStr, str2::PauliStr, simplification::Bool=true) -> PauliSum{Int}

    str1 + str2 -> PauliSum{Int}

Add two `PauliStr`, returning their sum as a [`PauliSum`](@ref). When `T` is unspecified 
(in the case of the second signature), it defaults to `Int`; `T` is subject to the same 
restriction as in the `PauliSum` constructor. If `simplification=true`, the same 
simplification procedure used by the `PauliSum` constructor (when the same-named argument 
is set to `true`) is applied to the result. Neither input is mutated, and the result does 
not reference any data in either input. Using the binary operator `+` is equivalent to 
invoking the second method signature (with the default value of `simplification`).
"""
add(::Type{T}, str1::PauliStr, str2::PauliStr, simplification::Bool=true
    ) where {T<:SignedReal} = 
PauliSum(T, [str1, str2], simplification)

add(str1::PauliStr, str2::PauliStr, simplification::Bool=true) = 
add(Int, str1, str2, simplification)

"""
    add(::Type{T}, h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
        ) where {T<:$SignedReal, T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{T}

    add(h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
        ) where {T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{promote_type(T1, T2)}

    h1 + h2 -> PauliSum

Add two `PauliSum`, returning their sum with the coefficient type automatically promoted. 
If `simplification=true`, the same simplification procedure used by the [`PauliSum`](@ref) 
constructor (when the same-named argument is set to `true`) is applied to the result. The 
result does not reference any data in either `h1` or `h2`.

!!! info
    For the first method signature where `T` specifies the core type of the returned 
    `PauliSum`, when `simplification=true`, the precision level of the simplification 
    procedure is determined by `$(extendType|>repr)(T, complex( promote_type(T1, T2) ))`.
"""
function add(::Type{T}, h1::PauliSum, h2::PauliSum, simplification::Bool=true
             ) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    PauliSum(T, vcat(h1.str, h2.str), vcat(h1.coeff, h2.coeff), simplification)
end

add(h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true) where 
   {T1<:SignedReal, T2<:SignedReal} = 
add(promote_type(T1, T2), h1, h2, simplification)

"""
    add(::Type{T}, h::PauliSum{T1}, term::Pair{PauliStr, <:Union{Complex{T2}, T2}}, 
        simplification::Bool=true
        ) where {T<:$SignedReal, T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{T}

    add(h::PauliSum{T1}, term::Pair{PauliStr, <:Union{Complex{T2}, T2}}, 
        simplification::Bool=true
        ) where {T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{promote_type(T1, T2)}

Add a coefficient-carrying `term` (e.g., as returned by [`indexTerm`](@ref)) to `h`, 
returning the sum with the coefficient type automatically promoted. The added string is 
`term.first`, whose phase is absorbed into its associated coefficient `term.second` by the 
`PauliSum` constructor. If `simplification=true`, the same simplification procedure used by 
the [`PauliSum`](@ref) constructor (when the same-named argument is set to `true`) is 
applied to the result. The result does not reference any data in either `h` or `term`.

!!! info
    For the first method signature where `T` specifies the core type of the returned 
    `PauliSum`, when `simplification=true`, the precision level of the simplification 
    procedure is determined by `$(extendType|>repr)(T, complex( promote_type(T1, T2) ))`.
"""
function add(::Type{T}, h::PauliSum, term::PStrCoeffPair, simplification::Bool=true
             ) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    PauliSum(T, vcat(h.str, term.first), vcat(h.coeff, term.second), simplification)
end

add(h::PauliSum{T1}, term::PStrCoeffPair{T2}, simplification::Bool=true) where 
   {T1<:SignedReal, T2<:SignedReal} = 
add(promote_type(T1, T2), h, term, simplification)

"""
    add(::Type{T}, h::PauliSum, s::PauliStr, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    add(h::PauliSum{T}, s::PauliStr, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    h + s -> PauliSum
    s + h -> PauliSum

Add a single `PauliStr` to `h`, i.e., shorthand for `add(h, s=>one(T), simplification)`: 
`s` is appended as a new term with its phase folded into the associated coefficient. If 
`simplification=true`, the same simplification procedure used by the `PauliSum` 
constructor (when the same-named argument is set to `true`) is applied to the result. The 
result does not reference any data in either `h` or `s`.
"""
add(::Type{T}, h::PauliSum, s::PauliStr, simplification::Bool=true
    ) where {T<:SignedReal} = 
add(T, h, s=>one(T), simplification)

add(h::PauliSum{T}, s::PauliStr, simplification::Bool=true) where {T<:SignedReal} = 
add(T, h, s, simplification)

Base.:+(h::PauliSum, obj::PauliStrOrSum) = add(h, obj)
Base.:+(s::PauliStr, h::PauliSum) = add(h, s)
Base.:+(s1::PauliStr, s2::PauliStr, s3::PauliStr...) = PauliSum(Int, [s1, s2, s3...])


"""
    sub(::Type{T}, str1::PauliStr, str2::PauliStr, simplification::Bool=true
        ) where {T<:$SignedReal} -> PauliSum{T}

    sub(str1::PauliStr, str2::PauliStr, simplification::Bool=true) -> PauliSum{Int}

    str1 - str2 -> PauliSum{Int}

Subtract `str2` from `str1`, returning their difference as a [`PauliSum`](@ref). When `T` 
is unspecified (in the case of the second signature), it defaults to `Int`; `T` is subject 
to the same restriction as in the `PauliSum` constructor. If `simplification=true`, the 
same simplification procedure used by the `PauliSum` constructor (when the same-named 
argument is set to `true`) is applied to the result. Neither input is mutated, and the 
result does not reference any data in either input. Using the binary operator `-` is 
equivalent to invoking the second method signature (with the default value of 
`simplification`).
"""
function sub(::Type{T}, str1::PauliStr, str2::PauliStr, simplification::Bool=true
             ) where {T<:SignedReal}
    PauliSum(T, [str1, scale!(PauliStr(str2), negRea)], simplification)
end

sub(str1::PauliStr, str2::PauliStr, simplification::Bool=true) = 
sub(Int, str1, str2, simplification)

"""
    sub(::Type{T}, h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
        ) where {T<:$SignedReal, T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{T}

    sub(h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
        ) where {T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{promote_type(T1, T2)}

    h1 - h2 -> PauliSum

Subtract `h2` from `h1`, returning their difference with the coefficient type 
automatically promoted.  If `simplification=true`, the same simplification procedure used 
by the `PauliSum` constructor (when the same-named argument is set to `true`) is applied to 
the result. The result does not reference any data in either `h1` or `h2`.

!!! info
    For the first method signature where `T` specifies the core type of the returned 
    `PauliSum`, when `simplification=true`, the precision level of the simplification 
    procedure is determined by `$(extendType|>repr)(T, complex( promote_type(T1, T2) ))`.
"""
function sub(::Type{T}, h1::PauliSum, h2::PauliSum, simplification::Bool=true
             ) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    strs = vcat(h1.str, h2.str)
    half = @view strs[begin+countTerms(h1):end]
    map!(s->scale!(PauliStr(s), negRea), half)
    PauliSum(T, strs, vcat(h1.coeff, h2.coeff), simplification)
end

sub(h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true) where 
   {T1<:SignedReal, T2<:SignedReal} = 
sub(promote_type(T1, T2), h1, h2, simplification)


"""
    sub(::Type{T}, h::PauliSum{T1}, term::Pair{PauliStr, <:Union{Complex{T2}, T2}}, 
        simplification::Bool=true
        ) where {T<:$SignedReal, T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{T}

    sub(h::PauliSum{T1}, term::Pair{PauliStr, <:Union{Complex{T2}, T2}}, 
        simplification::Bool=true
        ) where {T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{promote_type(T1, T2)}

    h - term -> PauliSum

Subtract a coefficient-carrying `term` (e.g., as returned by [`indexTerm`](@ref)) from 
`h`, returning the difference with the coefficient type automatically promoted. The string 
of `term` is `term.first`, whose phase is absorbed into its associated coefficient 
`term.second` by the `PauliSum` constructor. If `simplification=true`, the same 
simplification procedure used by the [`PauliSum`](@ref) constructor (when the same-named 
argument is set to `true`) is applied to the result. The result does not reference any data 
in either `h` or `term`.

!!! info
    For the first method signature where `T` specifies the core type of the returned 
    `PauliSum`, when `simplification=true`, the precision level of the simplification 
    procedure is determined by `$(extendType|>repr)(T, complex( promote_type(T1, T2) ))`.
"""
function sub(::Type{T}, h::PauliSum, term::PStrCoeffPair, simplification::Bool=true
             ) where {T<:SignedReal}
    add(T, h, scale!(PauliStr(term.first), negRea)=>term.second, simplification)
end

sub(h::PauliSum{T1}, term::PStrCoeffPair{T2}, simplification::Bool=true) where 
   {T1<:SignedReal, T2<:SignedReal} = 
sub(promote_type(T1, T2), h, term, simplification)

"""
    sub(::Type{T}, h::PauliSum, s::PauliStr, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    sub(::Type{T}, s::PauliStr, h::PauliSum, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    sub(h::PauliSum{T}, s::PauliStr, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    sub(s::PauliStr, h::PauliSum{T}, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    h - s -> PauliSum
    s - h -> PauliSum

Subtract a single `PauliStr` `s` from `h`, or `h` from `s`, where `s` is treated 
effectively as a term whose associated coefficient being phase of `s`. If 
`simplification=true`, the same simplification procedure used by the `PauliSum` constructor 
(when the same-named argument is set to `true`) is applied to the result. The result does 
not reference any data in either `h` or `s`.
"""
sub(::Type{T}, h::PauliSum, s::PauliStr, simplification::Bool=true) where {T<:SignedReal} = 
sub(T, h, s=>one(T), simplification)

function sub(::Type{T}, s::PauliStr, h::PauliSum, simplification::Bool=true
             ) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    strs = vcat(s, h.str)
    body = @view strs[begin+1:end]
    map!(s->scale!(PauliStr(s), negRea), body)
    PauliSum(T, strs, vcat(one(T), h.coeff), simplification)
end

sub(h::PauliSum{T}, s::PauliStr, simplification::Bool=true) where {T<:SignedReal} = 
sub(T, h, s, simplification)

sub(s::PauliStr, h::PauliSum{T}, simplification::Bool=true) where {T<:SignedReal} = 
sub(T, s, h, simplification)

Base.:-(h::PauliStrOrSum, obj::PauliStrOrSum) = sub(h, obj)
Base.:-(h::PauliSum, obj::PStrCoeffPair) = sub(h, obj)


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
    mul(::Type{T}, str::PauliStr, coeff::Union{T1, Complex{T1}}, 
        simplification::Bool=true) where {T<:$SignedReal, T1<:$SignedReal} -> 
    PauliSum{T}

    mul(str::PauliStr, coeff::Union{T, Complex{T}}, 
        simplification::Bool=true) where {T<:$SignedReal} -> 
    PauliSum{T}

    str * coeff -> PauliSum
    coeff * str -> PauliSum

Multiply a `PauliStr` by a real or complex scalar `coeff`, returning the product 
`str * coeff` (equal to `coeff * str`). Unlike multiplication by a [`PhaseFactor`](@ref), a 
general scalar coefficient may not be stored within a `PauliStr`, so the result is always 
promoted to a `PauliSum`. If `simplification=true`, the same simplification procedure used 
by the `PauliSum` constructor (when the same-named argument is set to `true`) is applied to 
the result. The result does not reference any data in `str`.
"""
function mul(::Type{T}, str::PauliStr, coeff::SRealOrComplex, simplification::Bool=true
             ) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    PauliSum([str], Complex{T}(coeff), simplification)
end

mul(str::PauliStr, coeff::C, simplification::Bool=true) where {C<:SRealOrComplex} = 
mul(real(C), str, coeff, simplification)

"""
    mul(::Type{T}, s::PauliStr, h::PauliSum, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    mul(::Type{T}, h::PauliSum, s::PauliStr, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    mul(s::PauliStr, h::PauliSum{T}, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    mul(h::PauliSum{T}, s::PauliStr, simplification::Bool=true
        ) where {T<:$SignedReal} -> 
    PauliSum{T}

    s * h -> PauliSum
    h * s -> PauliSum

Multiply a `PauliStr` with a `PauliSum`, returning their product. Each stored string of `h` 
is multiplied by `s` on the matching side while the coefficients are carried over. If 
`simplification=true`, the same simplification procedure used by the `PauliSum` constructor 
(when the same-named argument is set to `true`) is applied to the result. The result does 
not reference any data in either `s` or `h`.
"""
function mul(::Type{T}, s::PauliStr, h::PauliSum, simplification::Bool=true
             ) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    newStrs = map(ele->mul(s, ele), h.str)
    PauliSum(newStrs, convert(Memory{Complex{T}}, h.coeff), simplification)
end

function mul(::Type{T}, h::PauliSum, s::PauliStr, simplification::Bool=true
             ) where {T<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    newStrs = map(ele->mul(ele, s), h.str)
    PauliSum(newStrs, convert(Memory{Complex{T}}, h.coeff), simplification)
end

mul(s::PauliStr, h::PauliSum{T}, simplification::Bool=true
    ) where {T<:SignedReal} = 
mul(T, s, h, simplification)

mul(h::PauliSum{T}, s::PauliStr, simplification::Bool=true) where {T<:SignedReal} = 
mul(T, h, s, simplification)

"""
    mul(::Type{T}, h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
        ) where {T<:$SignedReal, T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{T}

    mul(h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
        ) where {T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{promote_type(T1, T2)}

    h1 * h2 -> PauliSum

Multiply `h1` by `h2`, returning their product. If `simplification=true`, the same 
simplification procedure used by the `PauliSum` constructor (when the same-named argument 
is set to `true`) is applied to the result. The result does not reference any data in 
either `h1` or `h2`.

!!! info
    For the first method signature where `T` specifies the core type of the returned 
    `PauliSum`, each intermediate coefficient product is evaluated at the precision level 
    of `$(extendType|>repr)(T, complex( promote_type(T1, T2) ))` (no overflow protection 
    for `Integer`) and then converted to `Complex{T}`.
"""
function mul(::Type{T}, h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
             ) where {T<:SignedReal, T1<:SignedReal, T2<:SignedReal}
    checkCoreDataTypeForPauliSum(T)

    cL, sL = h1.coeff, h1.str
    cR, sR = h2.coeff, h2.str
    m, n = length(cL), length(cR)
    tempC = extendType(T, complex( promote_type(T1, T2) ))

    cs = Memory{Complex{T}}(undef, m * n)
    ss = Memory{PauliStr}(undef, m * n)
    k = 0
    @inbounds for j in 1:n, i in 1:m
        k += 1
        cs[begin+k-1] = tempC(cL[begin+i-1]) * tempC(cR[begin+j-1])
        ss[begin+k-1] = mul(sL[begin+i-1], sR[begin+j-1]) #> Phase folded into `.phase`
    end

    PauliSum(ss, cs, simplification)
end

mul(h1::PauliSum{T1}, h2::PauliSum{T2}, simplification::Bool=true
    ) where {T1<:SignedReal, T2<:SignedReal} = 
mul(promote_type(T1, T2), h1, h2, simplification)

"""
    mul(::Type{T}, h::PauliSum{T1}, coeff::Union{T2, Complex{T2}}, simplification::Bool=true
        ) where {T<:$SignedReal, T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{T}

    mul(h::PauliSum{T1}, coeff::Union{T2, Complex{T2}}, simplification::Bool=true
        ) where {T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{promote_type(T1, T2)}

    h * coeff -> PauliSum
    coeff * h -> PauliSum

Multiply `h` by a coefficient `coeff`, returning a new `PauliSum` whose coefficient type 
is automatically promoted. When `simplification=true`, the result is fully canonicalized; 
in particular, scaling by an exact zero returns an empty `PauliSum` as the zero operator. 
For in-place scaling without type promotion, see [`scale!`](@ref).

!!! info
    For the first method signature where `T` specifies the core type of the returned 
    `PauliSum`, each intermediate coefficient product is evaluated at the precision level 
    of `$(extendType|>repr)(T, complex( promote_type(T1, T2) ))` (no overflow protection 
    for `Integer`) and then converted to `Complex{T}`.
"""
function mul(::Type{T}, h::PauliSum{T1}, coeff::SRealOrComplex{T2}, 
             simplification::Bool=true
             ) where {T<:SignedReal, T1<:SignedReal, T2<:SignedReal}
    checkCoreDataTypeForPauliSum(T)
    tempC = extendType(T, complex( promote_type(T1, T2) ))
    tempCoeff = tempC(coeff)
    coeffs = map(x->Complex{T}(tempC(x) * tempCoeff), h.coeff)
    PauliSum(h.str, coeffs, simplification)
end

mul(h::PauliSum{T}, coeff::C, simplification::Bool=true
    ) where {T<:SignedReal, C<:SRealOrComplex} = 
mul(promote_type(T, real(C)), h, coeff, simplification)

"""
    mul(::Type{T}, h::PauliSum, phase::PhaseFactor, 
        simplification::Bool=true) where {T<:$SignedReal} -> PauliSum{T}

    mul(h::PauliSum{T}, phase::PhaseFactor, 
        simplification::Bool=true) where {T<:$SignedReal} -> PauliSum{T}

    h * phase -> PauliSum
    phase * h -> PauliSum

Multiply `h` by a phase `phase`, returning a new `PauliSum` in the canonical form. If 
`simplification=true`, the same simplification procedure used by the `PauliSum` constructor 
(when the same-named argument is set to `true`) is applied to the result. The result does 
not reference any data in `h`.
"""
mul(::Type{T}, h::PauliSum, phase::PhaseFactor, simplification::Bool=true
    ) where {T<:SignedReal} = 
mul(T, h, evalPhase(phase), simplification)

mul(h::PauliSum{T}, phase::PhaseFactor, simplification::Bool=true) where {T<:SignedReal} = 
mul(T, h, phase, simplification)


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
[`evalPhase(p)`](@ref `evalPhase`), then absorbed into `h.coeff`. This is to match the 
canonical form of `PauliSum` in which its phases are carried by coefficients rather than 
the stored `PauliStr`.
"""
scale!(h::PauliSum, p::PhaseFactor) = UInt8(p) > 0 ? scale!(h, evalPhase(p)) : h

"""
    scale!(h::PauliSum, c::Union{T, Complex{T}}) where {T<:$SignedReal} -> PauliSum

Multiply `h` in place by a coefficient `c`, returning the mutated `h`. `h.coeff` is scaled 
in place, thus each product `h.coeff[i] * c` must be able to be converted to 
`eltype(h.coeff)`. No re-canonicalization is performed.
"""
scale!(h::PauliSum, c::SRealOrComplex) = (h.coeff .*= c; h)


"""
    checkCommute(op1::PauliStr, op2::PauliStr) -> Bool

    checkCommute(op1::PauliStr, op2::PauliSum) -> Bool

    checkCommute(op1::PauliSum, op2::PauliStr) -> Bool

    checkCommute(op1::PauliSum, op2::PauliSum) -> Bool

Return `true` if `op1` and `op2` commute. For two `PauliStr`, `false` also means they 
anticommute. In other cases, no such dichotomy exists. When the two operators 
explicitly act on different numbers of sites, commutation is evaluated under the implicit 
identity-padding convention (see [`PauliStr`](@ref) and [`PauliSum`](@ref)): the sites of 
the longer Pauli string beyond the shorter one's site count act against identities and 
never affect the result.
"""
function checkCommute(op1::PauliStr, op2::PauliStr)::Bool
    z1, x1 = op1.z, op1.x
    z2, x2 = op2.z, op2.x
    nWord = min(length(z1), length(z2))   #> Shorter string is complemented with identities
    parity = 0
    @inbounds for w in 1:nWord
        parity += count_ones(z1[w] & x2[w]) + count_ones(x1[w] & z2[w])
    end
    iseven(parity)
end

checkCommute(op1::PauliStrOrSum, op2::PauliStrOrSum) = isempty(evalCommute(op1, op2).str)


"""
    checkAntiCom(op1::PauliStr, op2::PauliStr) -> Bool

    checkAntiCom(op1::PauliStr, op2::PauliSum) -> Bool

    checkAntiCom(op1::PauliSum, op2::PauliStr) -> Bool

    checkAntiCom(op1::PauliSum, op2::PauliSum) -> Bool

Return `true` if `op1` and `op2` anticommute; otherwise, return `false`. For two 
`PauliStr`, this function is the logical negation of [`checkCommute`](@ref). In other 
cases, no such dichotomy exists. When the operators explicitly act on different numbers of 
sites, the implicit identity-padding convention applies (see [`PauliStr`](@ref) and 
[`PauliSum`](@ref)).
"""
function checkAntiCom(op1::PauliStr, op2::PauliStr)::Bool
    !checkCommute(op1, op2)
end

checkAntiCom(op1::PauliStrOrSum, op2::PauliStrOrSum) = isempty(evalAntiCom(op1, op2).str)


"""
    evalCommute(op1::PauliStr, op2::PauliStr) -> PauliSum{Int}

    evalCommute(op1::PauliStr, op2::PauliSum{T}) where {T<:$SignedReal} -> PauliSum{T}

    evalCommute(op1::PauliSum{T}, op2::PauliStr) where {T<:$SignedReal} -> PauliSum{T}

    evalCommute(op1::PauliSum{T1}, op2::PauliSum{T2}
                ) where {T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{promote_type(T1, T2)}

Return the commutator 
``[\\hat{O}_1, \\hat{O}_2] = \\hat{O}_1 \\hat{O}_2 - \\hat{O}_2 \\hat{O}_1``, with 
``\\hat{O}_1`` and ``\\hat{O}_2`` denoting `op1` and `op2` respectively, as a `PauliSum`. 
The multiplications within the commutation follow the implicit identity-padding convention 
of [`$(mul|>repr)`](@ref), so when the two operators explicitly act on different numbers of 
sites, the returned sum explicitly acts on the larger site count. When the commutator is 
zero (i.e., when `op1` and `op2` commute), this function returns an empty `PauliSum` as the 
zero operator.
"""
function evalCommute(op1::PauliStr, op2::PauliStr)
    prod1 = mul(op1, op2)
    prod2 = mul(op2, op1)
    prod1 == prod2 ? PauliSum(Int) : PauliSum(Int, [prod1, scale!(prod2, PhaseFactor(2))])
end

function evalCommute(op1::PauliStrOrSum, op2::PauliStrOrSum)
    prod1 = mul(op1, op2)::PauliSum
    prod2 = mul(op2, op1)::PauliSum
    PauliSum(vcat(prod1.str, prod2.str), vcat(prod1.coeff, -prod2.coeff))
end


"""
    evalAntiCom(op1::PauliStr, op2::PauliStr) -> PauliSum{Int}

    evalAntiCom(op1::PauliStr, op2::PauliSum{T}) where {T<:$SignedReal} -> PauliSum{T}

    evalAntiCom(op1::PauliSum{T}, op2::PauliStr) where {T<:$SignedReal} -> PauliSum{T}

    evalAntiCom(op1::PauliSum{T1}, op2::PauliSum{T2}
                ) where {T1<:$SignedReal, T2<:$SignedReal} -> 
    PauliSum{promote_type(T1, T2)}

Return the anticommutator 
``\\{\\hat{O}_1, \\hat{O}_2\\} = \\hat{O}_1 \\hat{O}_2 + \\hat{O}_2 \\hat{O}_1``, with 
``\\hat{O}_1`` and ``\\hat{O}_2`` denoting `op1` and `op2` respectively, as a `PauliSum`. 
The multiplications within the anticommutation follow the implicit identity-padding 
convention of [`$(mul|>repr)`](@ref), so when the two operators explicitly act on different 
numbers of sites, the returned sum explicitly acts on the larger site count. When the 
anticommutator is zero (i.e., when `op1` and `op2` anticommute), this function returns an 
empty `PauliSum` as the zero operator.
"""
function evalAntiCom(op1::PauliStr, op2::PauliStr)
    prod1 = mul(op1, op2)
    prod2 = mul(op2, op1)
    prod1 == prod2 ? PauliSum(Int, [prod1, prod2]) : PauliSum(Int)
end

function evalAntiCom(op1::PauliStrOrSum, op2::PauliStrOrSum)
    prod1 = mul(op1, op2)::PauliSum
    prod2 = mul(op2, op1)::PauliSum
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
    toAdjoint(h::PauliSum{T}) where {T<:$SignedReal} -> PauliSum{T}

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
