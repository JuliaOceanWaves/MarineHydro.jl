using Test
using SafeTestsets

# Test group for CI parallelism; run everything by default (`]test MarineHydro`).
const GROUP = get(ENV, "GROUP", "All")

# Skip GPU files only when they fail to load (missing/optional packages).
# Real assertion failures must still fail the suite.
function include_optional_tests(path)
    try
        include(path)
    catch e
        root = e
        while root isa LoadError
            root = root.error
        end
        if root isa ArgumentError || root isa InitError
            @testset "$(basename(path))" begin
                @test_skip "Could not load $(basename(path)): $root"
            end
        else
            rethrow()
        end
    end
end

@testset "MarineHydro.jl test suite" begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Consistency with Capytaine" include("consistency_with_Capytaine.jl")
        @time @safetestset "Green's functions" include("greens_function.jl")
        @time @safetestset "Vectorized Rankine" include("rankine_vectorized.jl")
        @time @safetestset "Matrix assembly" include("matrix_assembly.jl")
        @time @safetestset "Consistency with analytical solutions" include("consistency_with_analytical_solutions.jl")
        @time @safetestset "Forward speed" include("forward_speed_tests.jl")
    end

    if GROUP == "All" || GROUP == "Differentiation"
        @time @safetestset "Green's function differentiation" include("greens_function_differentiation.jl")
        @time @safetestset "Matrix assembly differentiation" include("matrix_assembly_differentiation.jl")
        @time @safetestset "Mesh geometry AD rules" include("ad_rules_geometry.jl")
        @time @safetestset "Outputs differentiation" include("outputs_differentiation.jl")
        @time @safetestset "Solve problem differentiation" include("solve_problem_differentiation.jl")
        @time @safetestset "Diffraction differentiation" include("diffraction_differentiation.jl")
    end

    if GROUP == "All" || GROUP == "GPU"
        include_optional_tests("./gpu_cpu_smoke.jl")
        include_optional_tests("./gpu.jl")
    end
end
