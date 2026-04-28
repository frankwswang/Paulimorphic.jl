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
end
