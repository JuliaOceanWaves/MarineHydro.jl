using Test
using Zygote
using ForwardDiff
using MarineHydro
using LinearAlgebra

const HAS_ENZYME = try
    using Enzyme
    true
catch
    false
end

@testset "GeometryAD fd mesh rules" begin
    hull = make_sphere_mesher(resolution=4)
    r0 = 1.0
    m = hull(r0)
    @test m isa Mesh
    @test m.nfaces > 0
    @test size(m.vertices, 2) == 3

    area_sum(r) = sum(hull(r).areas)
    h = 1e-5
    fd = (area_sum(r0 + h) - area_sum(r0 - h)) / (2h)
    zy = Zygote.gradient(area_sum, r0)[1]
    fwd = ForwardDiff.derivative(area_sum, r0)
    @test zy ≈ fd rtol=1e-4 atol=1e-5
    @test fwd ≈ fd rtol=1e-4 atol=1e-5
    @test zy ≈ 2 * area_sum(r0) / r0 rtol=1e-2

    if HAS_ENZYME
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        function enzyme_area_sum(r, mesher)
            return sum(mesher(r).areas)
        end
        ez = first(Enzyme.gradient(mode, enzyme_area_sum, r0, Enzyme.Const(hull)))
        @test ez ≈ zy rtol=1e-6 atol=1e-8
    end

    # Vector parameter: (radius,) through fd_mesh_function.
    hull1 = fd_mesh_function(p -> hull(p[1]))
    area_sum_p(p) = sum(hull1(p).areas)
    p0 = [r0]
    zy_p = Zygote.gradient(area_sum_p, p0)[1]
    fwd_p = ForwardDiff.gradient(area_sum_p, p0)
    @test zy_p[1] ≈ zy rtol=1e-6 atol=1e-8
    @test fwd_p[1] ≈ zy rtol=1e-6 atol=1e-8

    if HAS_ENZYME
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        function enzyme_area_sum_p(p, mesher)
            return sum(mesher(p).areas)
        end
        ez_p = first(Enzyme.gradient(mode, enzyme_area_sum_p, copy(p0), Enzyme.Const(hull1)))
        @test ez_p[1] ≈ zy rtol=1e-6 atol=1e-8
    end

    # Full BEM path: same pattern as paper/MeshGradients_singlebody.jl.
    dof = [0.0, 0.0, 1.0]
    ω = 1.2
    A_of_r(r) = calculate_radiation_forces(hull(r), dof, ω)[1]
    zy_A = Zygote.gradient(A_of_r, r0)[1]
    fd_A = (A_of_r(r0 + h) - A_of_r(r0 - h)) / (2h)
    @test zy_A ≈ fd_A rtol=1e-3 atol=1e-3
    if HAS_ENZYME
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        function enzyme_A_of_r(r, mesher, omega, dof)
            return calculate_radiation_forces(mesher(r), dof, omega)[1]
        end
        ez_A = first(Enzyme.gradient(mode, enzyme_A_of_r, r0,
            Enzyme.Const(hull), Enzyme.Const(ω), Enzyme.Const(dof)))
        @test ez_A ≈ zy_A rtol=1e-5 atol=1e-6
    end

    # smesh + solve: Dual / Enzyme through Capytaine mesher.
    function added_mass_smesh(r, omega, mesher)
        body = FloatingBody(StaticArraysMesh(mesher(r)), [:Heave], "s")
        return added_mass(solve(RadiationProblem(body, omega), DirectBEM())).Heave
    end
    fwd_s = ForwardDiff.derivative(r -> added_mass_smesh(r, ω, hull), r0)
    fd_s = (added_mass_smesh(r0 + h, ω, hull) - added_mass_smesh(r0 - h, ω, hull)) / (2h)
    @test fwd_s ≈ fd_s rtol=1e-3 atol=1e-3
    if HAS_ENZYME
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        ez_s = first(Enzyme.gradient(mode, added_mass_smesh, r0,
            Enzyme.Const(ω), Enzyme.Const(hull)))
        @test ez_s ≈ fwd_s rtol=1e-5 atol=1e-6
    end

    # Generator that returns StaticArraysMesh. Duals go through hull (FDMesher)
    # then StaticArraysMesh(::Mesh).
    smesh_area(r) = sum(StaticArraysMesh(hull(r)).areas)
    fwd_sa = ForwardDiff.derivative(smesh_area, r0)
    @test fwd_sa ≈ fd rtol=1e-4 atol=1e-5
end
