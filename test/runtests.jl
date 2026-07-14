using Paulimorphic
using Test

@testset "Paulimorphic tests" begin

    println("Number of threads used for testing: ", Threads.nthreads(), "\n")

    unit1 = "String Construction"
    println("Testing $(unit1)...")
    t1 = @elapsed @testset "$(unit1)" begin
        include("unit-tests/Strings-test.jl")
    end
    println("$(unit1) test finished in $t1 seconds.\n")

    unit2 = "String Operations"
    println("Testing $(unit2)...")
    t2 = @elapsed @testset "$(unit2)" begin
        include("unit-tests/Operations-test.jl")
    end
    println("$(unit2) test finished in $t2 seconds.\n")

    unit3 = "Graph-theory functions"
    println("Testing $(unit3)...")
    t3 = @elapsed @testset "$(unit3)" begin
        include("unit-tests/Graphs-test.jl")
    end
    println("$(unit3) test finished in $t3 seconds.\n")

    unit4 = "Encodings"
    println("Testing $(unit4)...")
    t4 = @elapsed @testset "$(unit4)" begin
        include("unit-tests/encodings-test.jl")
    end
    println("$(unit4) test finished in $t4 seconds.\n")
end
