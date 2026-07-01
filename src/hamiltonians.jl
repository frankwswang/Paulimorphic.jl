export hamiltonian

"""
    Hamiltonian{}

A constructor for classes of PauliSums to be automatically generated. 

The specific Hamiltonians which can be generated are among the following:
    - List of Encoding(s) for different free fermion Hamiltonians
        - Includes Jordan-Wigner, Balanced Jordan-Wigner, Ternary-Tree, and Kitaev-One-Local
    - One/Two Dimensional Lattice Hamiltonians 
        - Gapped 1D, 2D Square, 2D Honeycomb, 2D Kagome
    - 4-Regular Graph Hamiltonians
    - 2-local Hamiltonians 
    - Molecular Hamiltonians

# Inputs 
- `.n::Integer`: Number of qubits in the Hamiltonian 
- `.type::Symbol`: Type of Hamiltonian to be generated.
- `.type2::Symbol`: Secondary arguement for lattice types
- `encoding_list::Vector{PauliList}`: List of encodings to be used in the Hamiltonian generation. 
- `.coeff_type::Symbol`: Type of coefficients to be used in the Hamiltonian, including `:random`, `:uniform`, `:normal`, `:zeropmone`.

≡≡≡ Initialization Method(s) ≡≡≡

    hamiltonian(n::Integer, type::Symbol; type2::Symbol=:none, coeff_type::Symbol=:random)

Construct a PauliSum on `n` qubits. The class of Hamiltonian is determined by the `type` arguement, 
with secondary type arguement `type2` for lattice Hamiltonians. The coefficients of the PauliSum are 
determined by the `coeff_type` arguement, which takes one of four values: `:random`, `:uniform`, `:normal`, `:zeropmone`.

    hamiltonian(n::Integer, type::Symbol; type2::Symbol=:none)

Construct a PauliSum on `n` qubits. The class of Hamiltonian is determined by the `type` arguement, 
with secondary type arguement `type2` for lattice Hamiltonians. The coefficients of the PauliSum are 
set to all be 1.0, utilizing a version of PauliSum which does not specify coefficients.
"""

"""
    encoding_check(encoding::PauliList) -> Bool

Return whether every distinct pair of strings in `encoding` anticommutes. This is the
validity condition expected by the encoding-based Hamiltonian constructor.
"""
function encoding_check(encoding::PauliList)
    strs = encoding.str
    nTerm = length(strs)
    for i in 1:(nTerm-1)
        si = strs[begin+i-1]
        for j in (i+1):nTerm
            checkAntiCom(si, strs[begin+j-1]) || return false
        end
    end
    true
end

"""
    generate_encoding_hamiltonian_terms(n::Integer, encoding::PauliList) -> Vector{PauliStr}

Generate all quadratic products from the Pauli strings in `encoding` as

    mul(encoding[i], encoding[j]), for i < j,

returning the resulting Pauli terms in a flat vector. The returned strings are normalized
to unit phase (`PhaseFactor(0)`), so phase information is delegated to coefficient
generation.

An `ArgumentError` is thrown if any string in `encoding` does not act on exactly `n`
qubits.
"""
function generate_encoding_hamiltonian_terms(n::Integer, encoding::PauliList)
    nInt = Int(n)
    terms = PauliStr[]
    strs = encoding.str
    nMajorana = length(strs)
    sizehint!(terms, nMajorana * (nMajorana - 1) ÷ 2)

    for i in 1:(nMajorana-1)
        si = strs[begin+i-1]
        if si.n != nInt
            throw(ArgumentError("Encoding string length $(si.n) does not match n=$nInt."))
        end
        for j in (i+1):nMajorana
            sj = strs[begin+j-1]
            sj.n == nInt || throw(ArgumentError("Encoding string length $(sj.n) does not match n=$nInt."))
            # Drop reference phase on terms; coefficients are sampled separately.
            push!(terms, PauliStr(mul(si, sj), PhaseFactor(0)))
        end
    end

    terms
end

"""
    _edge_terms(nInt::Int, edges::Vector{NTuple{2, Int}}) -> Vector{PauliStr}

Internal helper for edge-based models. For each edge `(i, j)` it generates

    X_i X_j, Y_i Y_j, Z_i Z_j,

with identities on all other sites, and returns the concatenated term list.
"""
function _edge_terms(nInt::Int, edges::Vector{NTuple{2, Int}})
    paulis = (symX, symY, symZ)
    terms = PauliStr[]
    sizehint!(terms, 3 * length(edges))

    for (i, j) in edges
        for p in paulis
            siteOps = fill(symI, nInt)
            siteOps[i] = p
            siteOps[j] = p
            push!(terms, PauliStr(siteOps, PhaseFactor(0)))
        end
    end

    terms
end

"""
    _grid_shape(nInt::Int) -> NTuple{2, Int}

Internal helper that factors `nInt` into a near-square `(nRow, nCol)` rectangle used for
grid-style lattice embeddings.
"""
function _grid_shape(nInt::Int)
    nInt >= 1 || throw(DomainError(nInt, "`n` must be positive for lattice generation."))
    r = floor(Int, sqrt(nInt))
    while r > 1 && (nInt % r != 0)
        r -= 1
    end
    c = nInt ÷ r
    r, c
end

"""
    generate_lattice_hamiltonian_terms(n::Integer, type2::Symbol) -> Vector{PauliStr}

Generate edge-coupled lattice terms on `n` qubits for the selected lattice family `type2`.
The current implementation builds a lattice edge set first, then maps each edge to the
three same-axis couplings `XX`, `YY`, and `ZZ`.

Supported `type2` values:
- `:Gapped1D`: open chain plus alternating next-nearest-neighbor edges.
- `:Square2D`: rectangular nearest-neighbor grid.
- `:Honeycomb2D`: brick-wall embedding with staggered vertical bonds.
- `:Kagome2D`: simple strip of corner-sharing triangles.
"""
function generate_lattice_hamiltonian_terms(n::Integer, type2::Symbol)
    nInt = Int(n)
    nInt >= 1 || throw(DomainError(n, "`n` must be positive for lattice generation."))

    edges = Set{NTuple{2, Int}}()

    if type2 == :Gapped1D
        #> Nearest-neighbor chain with alternating next-nearest couplings.
        for i in 1:(nInt-1)
            push!(edges, (i, i+1))
        end
        for i in 1:(nInt-2)
            isodd(i) && push!(edges, (i, i+2))
        end
    elseif type2 == :Square2D
        #> Rectangular nearest-neighbor grid.
        nRow, nCol = _grid_shape(nInt)
        for r in 1:nRow, c in 1:nCol
            v = (r - 1) * nCol + c
            c < nCol && push!(edges, minmax(v, v + 1))
            r < nRow && push!(edges, minmax(v, v + nCol))
        end
    elseif type2 == :Honeycomb2D
        #> Brick-wall honeycomb embedding.
        iseven(nInt) || throw(ArgumentError("Honeycomb2D requires even n."))
        nRow, nCol = _grid_shape(nInt)
        for r in 1:nRow, c in 1:nCol
            v = (r - 1) * nCol + c

            #> Horizontal bonds.
            c < nCol && push!(edges, minmax(v, v + 1))

            #> Staggered vertical bonds (brick-wall embedding).
            if r < nRow
                connectDown = isodd(r + c)
                connectDown && push!(edges, minmax(v, v + nCol))
            end
        end
    elseif type2 == :Kagome2D
        #> Kagome strip built from corner-sharing triangles.
        nInt % 3 == 0 || throw(ArgumentError("Kagome2D requires n divisible by 3."))
        nCell = nInt ÷ 3

        #> A simple kagome strip built from corner-sharing triangles.
        for cell in 1:nCell
            a = 3 * (cell - 1) + 1
            b = a + 1
            c = a + 2

            push!(edges, (a, b))
            push!(edges, (b, c))
            push!(edges, (a, c))

            if cell < nCell
                nextA = 3 * cell + 1
                push!(edges, minmax(b, nextA))
                push!(edges, minmax(c, nextA))
            end
        end
    else
        throw(ArgumentError("Unsupported lattice type2=$type2."))
    end

    _edge_terms(nInt, collect(edges))
end

"""
    generate_four_regular_graph_hamiltonian_terms(n::Integer) -> Vector{PauliStr}

Generate edge-coupled terms on a simple 4-regular graph over `n` vertices. The graph is
chosen as the circulant graph `C_n(1,2)`, where each vertex `i` is connected to
`i ± 1` and `i ± 2` modulo `n`. For every edge, the generated terms are `XX`, `YY`, and
`ZZ` on that edge.
"""
function generate_four_regular_graph_hamiltonian_terms(n::Integer)
    nInt = Int(n)
    nInt >= 5 || throw(ArgumentError("A simple 4-regular graph requires n >= 5."))

    #> Circulant graph C_n(1,2): each vertex i connects to i±1 and i±2 (mod n).
    edges = Set{NTuple{2, Int}}()
    for i in 1:nInt
        j1 = mod1(i + 1, nInt)
        j2 = mod1(i + 2, nInt)
        push!(edges, minmax(i, j1))
        push!(edges, minmax(i, j2))
    end

    _edge_terms(nInt, collect(edges))
end

"""
    generate_two_local_hamiltonian_terms(n::Integer) -> Vector{PauliStr}

Generate all strictly 2-local Pauli terms on `n` qubits (no 1-local terms). For every
unordered pair of distinct sites `(i, j)` with `i < j`, the function emits the nine
combinations

    {X, Y, Z}_i ⊗ {X, Y, Z}_j.
"""
function generate_two_local_hamiltonian_terms(n::Integer)
    nInt = Int(n)
    nInt >= 0 || throw(DomainError(n, "`n` must be nonnegative."))

    paulis = (symX, symY, symZ)
    terms = PauliStr[]
    sizehint!(terms, 9 * nInt * (nInt - 1) ÷ 2)

    for i in 1:(nInt-1)
        for j in (i+1):nInt
            for p1 in paulis
                for p2 in paulis
                    siteOps = fill(symI, nInt)
                    siteOps[i] = p1
                    siteOps[j] = p2
                    push!(terms, PauliStr(siteOps, PhaseFactor(0)))
                end
            end
        end
    end

    terms
end

"""
    generate_coefficients(num_terms::Int, coeff_type::Symbol) -> Vector{<:Real}

Generate `num_terms` scalar coefficients according to `coeff_type`:
- `:random` gives i.i.d. uniform samples in `[0, 1)`.
- `:uniform` gives all ones.
- `:normal` gives i.i.d. standard normal samples.
- `:zeropmone` gives i.i.d. samples from `{-1.0, 0.0, 1.0}`.
"""
function generate_coefficients(num_terms::Int, coeff_type::Symbol)
    if coeff_type == :random
        return rand(num_terms)
    elseif coeff_type == :uniform
        return fill(1.0, num_terms)
    elseif coeff_type == :normal
        return randn(num_terms)
    elseif coeff_type == :zeropmone
        return rand([-1.0, 0, 1.0], num_terms)
    end

    throw(ArgumentError("Unsupported coeff_type=$coeff_type."))
end

"""
    hamiltonian(n::Integer, encoding_list::Vector{PauliList}; coeff_type::Symbol=:random) -> PauliSum

Construct a Hamiltonian from one or more fermion-to-qubit encodings. Each encoding must
contain mutually anticommuting Pauli strings, all with site count `n`. For each encoding,
all quadratic products `i < j` are generated and assigned sampled coefficients; all terms
from all encodings are then merged into a single `PauliSum`.
"""
function hamiltonian(n::Integer, encoding_list::Vector{PauliList}; coeff_type::Symbol=:random)
    nInt = Int(n)
    #> Validate each provided encoding before generating terms.
    for encoding in encoding_list
        #> Check pairwise anticommutation property.
        encoding_check(encoding) || throw(ArgumentError("Every PauliList in encoding_list must be mutually anticommuting."))
        for str in encoding.str
            str.n == nInt || throw(ArgumentError("All encoding strings must have length n=$nInt."))
        end
    end

    #> Generate and concatenate terms from each encoding.
    hamiltonian_coeffs = Float64[]
    hamiltonian_terms = PauliStr[]
    for encoding in encoding_list
        terms = generate_encoding_hamiltonian_terms(nInt, encoding)
        coeffs = generate_coefficients(length(terms), coeff_type)
        append!(hamiltonian_terms, terms)
        append!(hamiltonian_coeffs, coeffs)
    end

    PauliSum(hamiltonian_coeffs, hamiltonian_terms)
end

function hamiltonian(n::Integer, type::Symbol; type2::Symbol=:none, coeff_type::Symbol=:random)
    nInt = Int(n)
    #> Dispatch by Hamiltonian family and attach sampled coefficients.

    if type == :Lattice
        # Generate lattice Hamiltonian terms based on the secondary type
        hamiltonian_terms = generate_lattice_hamiltonian_terms(nInt, type2)
        hamiltonian_coefficients = generate_coefficients(length(hamiltonian_terms), coeff_type)
        return PauliSum(hamiltonian_coefficients, hamiltonian_terms)


    elseif type == :FourRegularGraph
        # Generate 4-regular graph Hamiltonian terms
        hamiltonian_terms = generate_four_regular_graph_hamiltonian_terms(nInt)
        hamiltonian_coefficients = generate_coefficients(length(hamiltonian_terms), coeff_type)
        return PauliSum(hamiltonian_coefficients, hamiltonian_terms)


    elseif type == :TwoLocal
        # Generate 2-local Hamiltonian terms
        hamiltonian_terms = generate_two_local_hamiltonian_terms(nInt)
        hamiltonian_coefficients = generate_coefficients(length(hamiltonian_terms), coeff_type)
        return PauliSum(hamiltonian_coefficients, hamiltonian_terms)
    end

    throw(ArgumentError("Unsupported hamiltonian type=$type."))
end