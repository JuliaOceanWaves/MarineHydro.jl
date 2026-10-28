using MarineHydro
using PyCall
using Test
using ForwardDiff

mesh = MarineHydro.Mesh(MarineHydro.example_mesh_from_capytaine())
smesh = MarineHydro.StaticArraysMesh(MarineHydro.example_mesh_from_capytaine())
greens_functions = (Rankine(), RankineReflected(), GFWu())

@testset "Matrix shape" begin
    wavenumber = 1.0
    ω = √(wavenumber*9.8)
    dof = [0,0,1]
    green_functions = (
        Rankine(),
        RankineReflected(),
        #=GFWu()=#
    )
    S, D = assemble_matrices(green_functions, mesh, 1.0);
    S_, K = assemble_matrices(green_functions, mesh, 1.0; direct=false);
    @test size(S) == size(S_) == size(D) == size(K)
    @test S ≈ S_
    @test !(D ≈ K)
end

S, D = MarineHydro.assemble_matrices_comprehension(greens_functions, mesh, 1.0)
S_, D_ = MarineHydro.assemble_matrices_broadcasting(greens_functions, mesh, 1.0)
S__, D__ = MarineHydro.assemble_matrices_broadcasting(greens_functions, smesh, 1.0)
elements = [element(smesh, i) for i in 1:smesh.nfaces]
S_el, D_el = MarineHydro.assemble_matrices_broadcasting(greens_functions, elements, 1.0)
S_sa, D_sa = assemble_matrices(greens_functions, smesh, 1.0)
smesh_from_mesh = MarineHydro.StaticArraysMesh(mesh)
S_conv, D_conv = assemble_matrices(greens_functions, smesh_from_mesh, 1.0)
@test S ≈ S_ ≈ S__ ≈ S_el ≈ S_sa ≈ S_conv
@test D ≈ D_ ≈ D__ ≈ D_el ≈ D_sa ≈ D_conv

@testset "Rankine reused across wavenumbers" begin
    static_gfs, wave_gfs = MarineHydro.partition_greens_functions(greens_functions)
    @test static_gfs == (Rankine(), RankineReflected())
    @test wave_gfs == (GFWu(),)

    Ss1, Ds1 = assemble_matrices(static_gfs, mesh, 1.0)
    Ss2, Ds2 = assemble_matrices(static_gfs, mesh, 2.0)
    @test Ss1 ≈ Ss2
    @test Ds1 ≈ Ds2

    for k in (0.7, 1.5)
        S_full, D_full = assemble_matrices(greens_functions, mesh, k)
        Sw, Dw = assemble_matrices(wave_gfs, mesh, k; include_identity=false)
        @test S_full ≈ Ss1 .+ Sw
        @test D_full ≈ Ds1 .+ Dw
    end
end

@testset "Rankine reused across wavenumbers on StaticArraysMesh" begin
    static_gfs, wave_gfs = MarineHydro.partition_greens_functions(greens_functions)
    Ss1, Ds1 = assemble_matrices(static_gfs, smesh, 1.0)
    Ss2, Ds2 = assemble_matrices(static_gfs, smesh, 2.0)
    @test Ss1 ≈ Ss2
    @test Ds1 ≈ Ds2

    for k in (0.7, 1.5)
        S_full, D_full = assemble_matrices(greens_functions, smesh, k)
        Sw, Dw = assemble_matrices(wave_gfs, smesh, k; include_identity=false)
        @test S_full ≈ Ss1 .+ Sw
        @test D_full ≈ Ds1 .+ Dw
    end
end

    @testset "batch solve matches per-problem solves" begin
    mesh4 = Mesh(MarineHydro.example_mesh_from_capytaine(4))
    body = FloatingBody(mesh4, ["Heave"], [0.0, 0.0, 0.0], "sphere")
    parameters = (wave_frequencies=[1.0, 1.4], radiating_dofs=[:Heave], influenced_dofs=[:Heave])
    problems = problems_from_data(parameters, body)
    @test eltype(problems.radiation) <: RadiationProblem
    @test problems.radiation[1] isa RadiationProblem{Float64}
    @test mesh4.nfaces isa Int
    @test typeof(MarineHydro.partition_greens_functions(greens_functions)) ==
          Tuple{Tuple{Rankine, RankineReflected}, Tuple{GFWu}}
    dual_ω = ForwardDiff.Dual(1.2, 1.0)
    dual_prob = remake(RadiationProblem(body, 1.2); omega=dual_ω)
    @test dual_prob isa RadiationProblem{<:ForwardDiff.Dual}
    @test typeof(dual_prob.omega) === typeof(dual_ω)
    @test typeof(dual_prob.wavenumber) === typeof(dual_ω)
    reused = solve(problems.radiation, DirectBEM())
    separate = [solve(p, DirectBEM()) for p in problems.radiation]
    for (a, b) in zip(reused, separate)
        @test a.forces.Heave ≈ b.forces.Heave rtol=1e-10 atol=1e-12
        @test added_mass(a).Heave ≈ added_mass(b).Heave rtol=1e-10 atol=1e-12
    end
end

@testset "Wu centers assemble matches element broadcast" begin
    k = 1.2
    wave = (GFWu(),)
    for direct in (true, false)
        Sw, Dw = assemble_matrices(wave, smesh, k; direct, include_identity=false)
        Swb, Dwb = MarineHydro.assemble_matrices_broadcasting(wave, smesh, k; direct, include_identity=false)
        @test Sw ≈ Swb rtol=1e-12 atol=1e-12
        @test Dw ≈ Dwb rtol=1e-12 atol=1e-12
    end
    e1 = element(smesh, 1)
    e2 = element(smesh, 2)
    @test MarineHydro._wu_greens(e1.center, e2.center, k) ≈ greens(GFWu(), e1, e2, k)
    @test MarineHydro._wu_integral_centers(e1.center, e2.center, e2.area, k) ≈ integral(GFWu(), e1, e2, k)
    s_ij, d_ij = MarineHydro._wu_sd_centers(e1.center, e2.center, e2.normal, e2.area, k, Val(true))
    @test s_ij ≈ MarineHydro._wu_integral_centers(e1.center, e2.center, e2.area, k)
    @test d_ij ≈ MarineHydro._wu_ndot_gradient_centers(e1.center, e2.center, e2.normal, e2.area, k, Val(true))
end

@testset "Vectorized StaticArraysMesh assemble matches comprehension" begin
    k = 1.3
    for gfs in ( (Rankine(), RankineReflected()), greens_functions )
        for direct in (true, false)
            Sc, Dc = MarineHydro.assemble_matrices_comprehension(gfs, mesh, k; direct)
            Sv, Dv = assemble_matrices(gfs, smesh, k; direct)
            Sb, Db = MarineHydro.assemble_matrices_broadcasting(gfs, smesh, k; direct)
            @test Sv ≈ Sc rtol=1e-10 atol=1e-12
            @test Dv ≈ Dc rtol=1e-10 atol=1e-12
            @test Sv ≈ Sb rtol=1e-10 atol=1e-12
            @test Dv ≈ Db rtol=1e-10 atol=1e-12
        end
    end
end


