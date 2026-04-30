using Test

@testset "MarineHydro.jl test suite" begin
    include("./consistency_with_Capytaine.jl")
    include("./greens_function.jl")
    include("./greens_function_differentiation.jl")
    include("./matrix_assembly.jl")
    include("./matrix_assembly_differentiation.jl")
    include("./consistency_with_analytical_solutions.jl")
    include("./outputs_differentiation.jl")
end
