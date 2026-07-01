export PauliList, Jordan_Wigner_encoding, Ternary_Tree_encoding, Kitaev_One_Local_encoding

"""
This file is split into two components
1. A constructor for the Jordan-Wigner, Balanced Jordan-Wigner, Ternary-Tree, and Kitaev-One-Local Encodings
2. A fermionic Hamiltonian constructor given Spin Hamiltonian and an Encoding 

The first half of the file is a constructor for four standard encodings. These constructors 
are used most in the encoding generated Hamiltonians. For the encoding list we define a 
new structure based on the PauliStr operator, a list of Paulis without any reference phase. The Pauli list 
also has function encoding_check, which checks if all Pauli elements do indeed mutually anticommute. 

The second half of the file is dedicated to constructing fermionic Hamiltonians from their spin representation and 
a fermion-to-qubit encoding. This fermionic sum should be formatted [TBD: ALIGN WITH HARTREE-FOCK CODE]. The encoding 
variable can take two different types: a PauliList above, or a Line Graph. If a line graph is passed in, the function 
will detect the (a) fermion-to-qubit encoding that solves the Line graph Hamiltonian, and generate the PauliList. 
"""

function _jw_strings(n::Int)
    n >= 0 || throw(DomainError(n, "`n` must be nonnegative."))
    strings = String[]
    sizehint!(strings, 2n)

    for i in 1:n
        prefix = repeat("Z", i - 1)
        suffix = repeat("I", n - i)
        push!(strings, prefix * "X" * suffix)
        push!(strings, prefix * "Y" * suffix)
    end

    strings
end

function Jordan_Wigner_encoding(n::Integer)
    nInt = Int(n)
    strs = [@pauli_str s for s in _jw_strings(nInt)]
    PauliList(strs, false)
end

"""
    ternary_tree_strings(nmodes::Int, nqubits::Int) -> Vector{String}

Build a balanced (BFS-grown) ternary tree over qubit-indexed nodes and return
the 2*nmodes Majorana Pauli strings, each padded with 'I' to length `nqubits`.
Each root-to-leaf path spells a Pauli string: at each internal node on the path
we place the letter (X/Y/Z) of the branch descended, identity elsewhere.
"""
function ternary_tree_strings(nmodes::Int, nqubits::Int)
    nmodes >= 0 || throw(DomainError(nmodes, "nmodes must be nonnegative"))
    nmodes == 0 && return String[]

    nleaves = 2 * nmodes + 1           # Majoranas (odd); we keep 2*nmodes
    letters = ('X', 'Y', 'Z')

    # Each frontier entry is a partial path: a list of (qubit_index, letter).
    frontier = Vector{Tuple{Int,Char}}[ Tuple{Int,Char}[] ]
    next_qubit = 1

    # Grow the tree breadth-first: turn the oldest leaf into an internal node
    # (fresh qubit index) with three children, until we have enough leaves.
    while length(frontier) < nleaves
        path = popfirst!(frontier)
        q = next_qubit
        next_qubit += 1
        for L in letters
            push!(frontier, vcat(path, (q, L)))
        end
    end

    paths = frontier[1:(2 * nmodes)]   # keep 2*nmodes leaves

    result = String[]
    for path in paths
        chars = fill('I', nqubits)
        for (q, L) in path
            1 <= q <= nqubits || throw(ArgumentError(
                "qubit index $q exceeds nqubits=$nqubits; need at least $q qubits"))
            chars[q] = L
        end
        push!(result, String(chars))
    end
    return result
end

function Ternary_Tree_encoding(nmodes::Integer, nqubits::Integer)
    strs = [@pauli_str s for s in ternary_tree_strings(Int(nmodes), Int(nqubits))]
    return PauliList(strs, false)
end

#> Standard single-argument form used by the rest of the package/tests.
Ternary_Tree_encoding(n::Integer) = Ternary_Tree_encoding(n, n)

function Kitaev_One_Local_encoding(n::Integer)
    #> Construct the Kitaev-One-Local encoding for n qubits.
    #> Local qubits carry one-site X/Y/Z operators and ancilla tails are taken
    #> from a restricted Jordan-Wigner list.
    nInt = Int(n)
    nInt >= 0 || throw(DomainError(n, "`n` must be nonnegative."))
    nInt == 0 && return PauliList(PauliStr[], false)

    local_count = cld(2nInt, 3)
    ancilla_count = nInt - local_count
    ancilla_words = _jw_strings(ancilla_count)
    ancilla_num = length(ancilla_words)

    strings = String[]
    sizehint!(strings, 2nInt)

    for i in 1:local_count
        left = repeat("I", i - 1)
        right = repeat("I", local_count - i)
        ancilla = ancilla_num == 0 ? "" : ancilla_words[mod1(i, ancilla_num)]

        push!(strings, left * "X" * right * ancilla)
        push!(strings, left * "Y" * right * ancilla)
        push!(strings, left * "Z" * right * ancilla)
    end

    resize!(strings, min(length(strings), 2nInt))
    strs = [@pauli_str s for s in strings]
    PauliList(strs, false)
end


#%% SECOND HALF OF FILE HERE, TO BE FILLED IN LATER%%#