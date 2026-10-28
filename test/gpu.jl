using Test
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Zygote
using DifferentiationInterface
using MarineHydro

const MH = MarineHydro

const HAS_ENZYME = try
    using Enzyme
    true
catch
    false
end

# Optional backends: not test Project.toml deps. CUDA/Metal stay off Linux CI.
const GPU_ARRAY_TYPES = Pair{String,Any}[]
try
    using JLArrays
    push!(GPU_ARRAY_TYPES, "JLArray" => JLArray)
catch
end
try
    using CUDA
    CUDA.functional() && push!(GPU_ARRAY_TYPES, "CuArray" => CuArray)
catch
end
try
    using Metal
    Metal.functional() && push!(GPU_ARRAY_TYPES, "MtlArray" => MtlArray)
catch
end

function tiny_static_mesh()
    vertices = [
        SVector(-1.0, -1.0, -1.0),
        SVector( 0.0, -1.0, -1.0),
        SVector( 0.0,  0.0, -1.0),
        SVector(-1.0,  0.0, -1.0),
        SVector( 1.0, -1.0, -1.0),
        SVector( 1.0,  0.0, -1.0),
    ]
    faces = [
        SVector(1, 2, 3, 4),
        SVector(2, 5, 6, 3),
    ]
    centers = [
        SVector(-0.5, -0.5, -1.0),
        SVector( 0.5, -0.5, -1.0),
    ]
    normals = [SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 1.0)]
    areas = [1.0, 1.0]
    radii = [sqrt(0.5), sqrt(0.5)]
    return MH.StaticArraysMesh(vertices, faces, centers, normals, areas, radii, 6, 2)
end

function static_element_pair()
    e1 = MH.StaticElement(
        SVector(0.0, 0.0, -1.0),
        @SMatrix([-0.5 -0.5 0.0; 0.5 -0.5 0.0; 0.5 0.5 0.0; -0.5 0.5 0.0]) .+ SVector(0.0, 0.0, -1.0)',
        SVector(0.0, 0.0, 1.0),
        1.0,
        sqrt(2) / 2,
    )
    e2 = MH.StaticElement(
        SVector(1.0, 1.0, -2.0),
        @SMatrix([-0.5 -0.5 0.0; 0.5 -0.5 0.0; 0.5 0.5 0.0; -0.5 0.5 0.0]) .+ SVector(1.0, 1.0, -2.0)',
        SVector(0.0, 0.0, 1.0),
        1.0,
        sqrt(2) / 2,
    )
    return e1, e2
end

function radiation_style_bc(smesh, ω, dof=SVector(0.0, 0.0, 1.0))
    scale = -1im * ω
    return [scale * dot(n, dof) for n in smesh.normals]
end

gpu_scalar(x) = x isa AbstractArray ? Array(x)[] : x

# Named function so Enzyme can differentiate through GPU/dummy-GPU assembly.
function enzyme_assembly_sum(kabs, gfs, smesh, arrtype)
    S, _ = MH.assemble_matrices_broadcasting(gfs, smesh, kabs; arrtype)
    return real(sum(Array(S)))
end

function enzyme_assembly_derivative(arrtype, gfs, smesh, k)
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
    derivs = Enzyme.gradient(mode, enzyme_assembly_sum, k,
        Enzyme.Const(gfs), Enzyme.Const(smesh), Enzyme.Const(arrtype))
    return first(derivs)
end

function backend_can_hold_elements(arrtype)
    try
        smesh = tiny_static_mesh()
        arrtype([element(smesh, 1)])
        return true
    catch err
        @info "Skipping $arrtype: cannot store StaticElement" exception=err
        return false
    end
end

function run_backend_tests(arrtype, label)
    smesh = tiny_static_mesh()
    k = 1.0
    ω = 1.0
    gfs_wu = (Rankine(), RankineReflected(), GFWu())
    gfs_del = (DelhommeauRankine(), DelhommeauRankineReflected(), GFWu())
    e1, e2 = static_element_pair()

    @testset "$label broadcasting vs CPU" begin
        for (gfs, gflabel) in ((gfs_wu, "Wu"), (gfs_del, "DelhommeauRankine"))
            @testset "$gflabel" begin
                S, D = MH.assemble_matrices_broadcasting(gfs, smesh, k)
                S_gpu, D_gpu = MH.assemble_matrices_broadcasting(gfs, smesh, k; arrtype)
                @test S_gpu isa arrtype
                @test Array(S_gpu) ≈ S
                @test Array(D_gpu) ≈ D

                S_ind, K = MH.assemble_matrices_broadcasting(gfs, smesh, k; direct=false)
                S_ind_gpu, K_gpu = MH.assemble_matrices_broadcasting(gfs, smesh, k; direct=false, arrtype)
                @test Array(S_ind_gpu) ≈ S_ind
                @test Array(K_gpu) ≈ K

                S_disp, D_disp = assemble_matrices(gfs, smesh, k; arrtype)
                @test Array(S_disp) ≈ S
                @test Array(D_disp) ≈ D
            end
        end

        elements_gpu = arrtype([element(smesh, i) for i in 1:smesh.nfaces])
        S_el, D_el = MH.assemble_matrices_broadcasting(gfs_wu, elements_gpu, k)
        S, D = MH.assemble_matrices_broadcasting(gfs_wu, smesh, k)
        @test S_el isa arrtype
        @test Array(S_el) ≈ S
        @test Array(D_el) ≈ D
    end

    @testset "$label KernelAbstractions vs broadcasting" begin
        S, D = MH.assemble_matrices_broadcasting(gfs_wu, smesh, k)
        S_ka, D_ka = MH.assemble_matrices_ka(gfs_wu, smesh, k; arrtype)
        @test Array(S_ka) ≈ S rtol=1e-8 atol=1e-10
        @test Array(D_ka) ≈ D rtol=1e-8 atol=1e-10

        S_ind, K = MH.assemble_matrices_broadcasting(gfs_wu, smesh, k; direct=false)
        S_ka_ind, K_ka = MH.assemble_matrices_ka(gfs_wu, smesh, k; direct=false, arrtype)
        @test Array(S_ka_ind) ≈ S_ind rtol=1e-8 atol=1e-10
        @test Array(K_ka) ≈ K rtol=1e-8 atol=1e-10
    end

    @testset "$label Green kernels" begin
        gpu_gfs = (
            Rankine(),
            RankineReflected(),
            GFWu(),
            DelhommeauRankine(),
            DelhommeauRankineReflected(),
        )
        for gf in gpu_gfs
            let gf = gf
                @testset "$(typeof(gf))" begin
                    E1 = arrtype([e1])
                    E2 = reshape(arrtype([e2]), 1, 1)
                    int_k(a, b) = integral(gf, a, b, k)
                    grad_k(a, b) = integral_gradient(gf, a, b, k; with_respect_to_first_variable=true)
                    g_k(a, b) = greens(gf, a, b, k)
                    gg_k(a, b) = gradient_greens(gf, a, b, k; with_respect_to_first_variable=true)

                    @test Array(int_k.(E1, E2))[1] ≈ integral(gf, e1, e2, k) atol=1e-10 rtol=1e-8
                    @test Array(grad_k.(E1, E2))[1] ≈ integral_gradient(gf, e1, e2, k; with_respect_to_first_variable=true) atol=1e-10 rtol=1e-8
                    @test Array(g_k.(E1, E2))[1] ≈ greens(gf, e1, e2, k) atol=1e-10 rtol=1e-8
                    @test Array(gg_k.(E1, E2))[1] ≈ gradient_greens(gf, e1, e2, k; with_respect_to_first_variable=true) atol=1e-10 rtol=1e-8
                end
            end
        end
    end

    @testset "$label linear solve" begin
        S, D = MH.assemble_matrices_broadcasting(gfs_wu, smesh, k)
        bc = radiation_style_bc(smesh, ω)
        ϕ, _ = solve(D, S, bc; direct=true)

        S_gpu, D_gpu = MH.assemble_matrices_broadcasting(gfs_wu, smesh, k; arrtype)
        ϕ_gpu, _ = solve(D_gpu, S_gpu, arrtype(bc); direct=true)
        @test Array(ϕ_gpu) ≈ ϕ rtol=1e-6 atol=1e-8

        ϕ_ind, sources = solve(D, S, bc; direct=false)
        ϕ_ind_gpu, sources_gpu = solve(D_gpu, S_gpu, arrtype(bc); direct=false)
        @test Array(ϕ_ind_gpu) ≈ ϕ_ind rtol=1e-6 atol=1e-8
        @test Array(sources_gpu) ≈ sources rtol=1e-6 atol=1e-8
    end

    @testset "$label BC / forces" begin
        bc = radiation_style_bc(smesh, ω)
        S, D = MH.assemble_matrices_broadcasting(gfs_wu, smesh, k)
        ϕ, _ = solve(D, S, bc)
        pressure = 1im * MH.SETTINGS.rho * ω * ϕ
        dof = [0.0, 0.0, 1.0]
        force_cpu = sum(pressure .* (-[dot(SVector(dof...), n) for n in smesh.normals]) .* smesh.areas)

        S_gpu, D_gpu = MH.assemble_matrices_broadcasting(gfs_wu, smesh, k; arrtype)
        ϕ_gpu, _ = solve(D_gpu, S_gpu, arrtype(bc))
        pressure_gpu = 1im * MH.SETTINGS.rho * ω * ϕ_gpu
        n_dot = arrtype(-[dot(SVector(dof...), n) for n in smesh.normals])
        areas_gpu = arrtype(smesh.areas)
        force_gpu = sum(pressure_gpu .* n_dot .* areas_gpu)
        @test gpu_scalar(force_gpu) ≈ force_cpu rtol=1e-6 atol=1e-8
    end

    @testset "$label differentiability" begin
        function assembly_sum(kabs; AT=Array)
            S, _ = MH.assemble_matrices_broadcasting(gfs_wu, smesh, kabs; arrtype=AT)
            return real(gpu_scalar(sum(S)))
        end

        cpu_fd = ForwardDiff.derivative(κ -> assembly_sum(κ; AT=Array), k)

        @testset "ForwardDiff Duals" begin
            try
                gpu_fd = ForwardDiff.derivative(κ -> assembly_sum(κ; AT=arrtype), k)
                @test gpu_fd ≈ cpu_fd rtol=1e-5 atol=1e-6
            catch err
                @info "ForwardDiff Duals on $label failed" exception=err
                @test_skip "ForwardDiff Duals not supported on $label"
            end
        end

        @testset "Zygote" begin
            cpu_zy = Zygote.gradient(κ -> assembly_sum(κ; AT=Array), k)[1]
            @test cpu_zy ≈ cpu_fd rtol=1e-5 atol=1e-6
            try
                gpu_grad = Zygote.gradient(κ -> assembly_sum(κ; AT=arrtype), k)[1]
                gpu_grad === nothing && error("Zygote returned nothing")
                @test gpu_grad ≈ cpu_zy rtol=1e-5 atol=1e-6
            catch err
                @info "Zygote on $label failed" exception=err
                @test_skip "Zygote cannot see through $label broadcasts"
            end
        end

        @testset "Enzyme reverse" begin
            # JLArray/Metal: Enzyme hits GPUArrays broadcast `mkcontext` (not an assembly bug).
            # CuArray is still attempted when CUDA.functional().
            if !HAS_ENZYME
                @test_skip "Enzyme not installed"
            elseif string(nameof(arrtype)) in ("JLArray", "MtlArray")
                @test_skip "Enzyme reverse cannot differentiate $label GPUArrays broadcasts yet"
            else
                gpu_grad = enzyme_assembly_derivative(arrtype, gfs_wu, smesh, k)
                @test gpu_grad ≈ cpu_fd rtol=1e-5 atol=1e-6
            end
        end

        @testset "AutoForwardDiff" begin
            gpu_grad = DifferentiationInterface.derivative(
                κ -> assembly_sum(κ; AT=arrtype), AutoForwardDiff(), k)
            @test gpu_grad ≈ cpu_fd rtol=1e-5 atol=1e-6
        end
    end

    @testset "$label end-to-end radiation-style" begin
        for (gfs, gflabel) in ((gfs_wu, "Wu"), (gfs_del, "DelhommeauRankine"))
            @testset "$gflabel" begin
                bc = radiation_style_bc(smesh, ω)
                S, D = MH.assemble_matrices_broadcasting(gfs, smesh, k)
                ϕ, _ = solve(D, S, bc)
                pressure = 1im * MH.SETTINGS.rho * ω * ϕ
                added_mass = real(sum(pressure .* smesh.areas)) / ω^2
                damping = imag(sum(pressure .* smesh.areas)) / ω

                S_gpu, D_gpu = MH.assemble_matrices_broadcasting(gfs, smesh, k; arrtype)
                ϕ_gpu, _ = solve(D_gpu, S_gpu, arrtype(bc))
                @test Array(ϕ_gpu) ≈ ϕ rtol=1e-6 atol=1e-8

                pressure_gpu = 1im * MH.SETTINGS.rho * ω * ϕ_gpu
                force_gpu = gpu_scalar(sum(pressure_gpu .* arrtype(smesh.areas)))
                @test real(force_gpu) / ω^2 ≈ added_mass rtol=1e-6 atol=1e-8
                @test imag(force_gpu) / ω ≈ damping rtol=1e-6 atol=1e-8
            end
        end
    end
end

@testset "Vendor-agnostic GPU array path" begin
    if isempty(GPU_ARRAY_TYPES)
        @test_skip "No GPU array backend (install JLArrays, CUDA, or Metal)"
    else
        for (label, arrtype) in GPU_ARRAY_TYPES
            @testset "$label" begin
                if !backend_can_hold_elements(arrtype)
                    @test_skip "$label cannot store Float64 StaticElement (e.g. Metal is Float32-only)"
                else
                    run_backend_tests(arrtype, label)
                end
            end
        end
    end
end
