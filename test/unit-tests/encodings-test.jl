using Test
using Paulimorphic

@testset "encodings.jl" begin

function to_word(p::PauliStr)
	nBitPerWord = 8 * sizeof(UInt)
	chars = Vector{Char}(undef, p.n)

	for i in 1:p.n
		bitPos = i - 1
		w = (bitPos ÷ nBitPerWord) + 1
		mask = one(UInt) << (bitPos % nBitPerWord)
		z = !iszero(p.z[w] & mask)
		x = !iszero(p.x[w] & mask)
		chars[i] = z && x ? 'Y' : z ? 'Z' : x ? 'X' : 'I'
	end

	String(chars)
end

p1 = pauli"XZ"
p2 = PauliStr([symX, symZ], PhaseFactor(3))
p3 = PauliStr([symX, symZ], PhaseFactor(1))

plist_merge = PauliList([p1, p2, p3], true)
@test length(plist_merge.str) == 1
@test plist_merge.str[1] == PauliStr([symX, symZ], PhaseFactor(0))

plist_keep = PauliList([p1, p2, p3], false)
@test length(plist_keep.str) == 3
@test all(s -> s == PauliStr([symX, symZ], PhaseFactor(0)), plist_keep.str)

@test Int(p2.phase) == 3
@test Int(p3.phase) == 1

@testset "Jordan-Wigner small example" begin
	enc = Jordan_Wigner_encoding(3)
	@test length(enc.str) == 6
	@test sort([to_word(s) for s in enc.str]) == sort([
		"XII", "YII", "ZXI", "ZYI", "ZZX", "ZZY"
	])
end

@testset "Ternary-tree small example" begin
	enc = Ternary_Tree_encoding(6)
	terms = [to_word(s) for s in enc.str]
	println("Ternary-tree (n=6) generated terms:")
	for (i, t) in enumerate(terms)
		println(i, ": ", t)
	end
	@test length(enc.str) == 12
end

@testset "Kitaev-one-local small example" begin
	enc = Kitaev_One_Local_encoding(6)
	@test length(enc.str) == 12
	@test sort([to_word(s) for s in enc.str]) == sort([
		"XIIIXI", "YIIIXI", "ZIIIXI",
		"IXIIYI", "IYIIYI", "IZIIYI",
		"IIXIZX", "IIYIZX", "IIZIZX",
		"IIIXZY", "IIIYZY", "IIIZZY"
	])
end

end
