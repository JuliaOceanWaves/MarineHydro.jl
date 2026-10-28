using Test
using LinearAlgebra
using StaticArrays
using PyCall
using MarineHydro

const MH = MarineHydro

function birk_setup(corners)
    T, mgc = MH.compute_local_system(corners)
    temp_corners = MH.compute_temporary_panel_corners(corners, mgc, T)
    A, centroid_local = MH.compute_area_and_centroid(temp_corners)
    qgc = MH.compute_qgc(centroid_local, T, mgc)
    local_corners = MH.compute_local_corners(temp_corners, centroid_local)
    return T, qgc, local_corners, A
end

function birk_evaluate(global_field_point, T, qgc, local_corners)
    field_point = T' * (global_field_point - qgc)
    φ = MH.velocity_potential(field_point[1], field_point[2], field_point[3], local_corners)
    v = T * MH.velocity_derivatives(field_point[1], field_point[2], field_point[3], local_corners)
    H = T * MH.velocity_hessian(field_point[1], field_point[2], field_point[3], local_corners) * T'
    return φ, v, field_point, H
end

@testset "Birk 2021 Rankine panel tests" begin

    @testset "Square panel (paper Fig. 5 / Table 1)" begin
        corners = (
            SVector(-0.5, -0.5, 0.0),
            SVector( 0.5, -0.5, 0.0),
            SVector( 0.5,  0.5, 0.0),
            SVector(-0.5,  0.5, 0.0),
        )
        T, qgc, local_corners, A = birk_setup(corners)
        @test A ≈ 1.0 atol=1e-8
        @test T ≈ I atol=1e-8
        @test qgc ≈ [0.0, 0.0, 0.0] atol=1e-8
        @test local_corners[1] ≈ SVector(-0.5, -0.5, 0.0) atol=1e-8
        @test local_corners[2] ≈ SVector( 0.5, -0.5, 0.0) atol=1e-8
        @test local_corners[3] ≈ SVector( 0.5,  0.5, 0.0) atol=1e-8
        @test local_corners[4] ≈ SVector(-0.5,  0.5, 0.0) atol=1e-8

        cases = (
            (SVector(0.0, 0.0, 2.0), -0.03899412, [0.0, 0.0, 0.01873493],
                [0.00882663 0.0 0.0; 0.0 0.00882663 0.0; 0.0 0.0 -0.01765326]),
            (SVector(0.0, 0.0, -0.5), -0.12626703, [0.0, 0.0, -0.16666667],
                [0.18377630 0.0 0.0; 0.0 0.18377630 0.0; 0.0 0.0 -0.36755260]),
            (SVector(0.5, 1.0, 0.0), -0.07396339, [0.03064217, 0.06513339, 0.0],
                [0.03062433 -0.07906868 0.0; -0.07906868 -0.11292475 0.0; 0.0 0.0 0.08230042]),
            (SVector(-0.25, 0.25, 0.5), -0.11549639, [-0.04024089, 0.04024089, 0.13755683],
                [0.14364770 0.02766957 0.10965628; 0.02766957 0.14364770 -0.10965628; 0.10965628 -0.10965628 -0.28729541]),
            (SVector(0.1, 0.4, 8.0), -0.00992118, [0.00001536, 0.00006145, 0.00123369],
                [0.00015354 -0.00000028 -0.00000572; -0.00000028 0.00015248 -0.00002286; -0.00000572 -0.00002286 -0.00030602]),
        )
        for (p, φ_exp, v_exp, H_exp) in cases
            φ, v, _, H = birk_evaluate(p, T, qgc, local_corners)
            @test φ ≈ φ_exp atol=1e-6
            @test collect(v) ≈ v_exp atol=1e-6
            @test collect(H) ≈ H_exp atol=1e-6
            @test H[1, 1] + H[2, 2] + H[3, 3] ≈ 0 atol=1e-8
        end
    end

    @testset "Twisted quadrilateral (paper Fig. 6 / Table 3)" begin
        corners = (
            SVector(-0.7, -1.0, 0.0),
            SVector(-1.0,  1.0, 0.0),
            SVector( 1.0,  0.5, 0.8),
            SVector( 1.0, -1.0, 1.0),
        )
        T, qgc, local_corners, A = birk_setup(corners)
        T_exp = [
            -0.08526306  0.89595433  0.43588536;
             0.99473574  0.10150419 -0.01406082;
            -0.05684204  0.43239188 -0.89989235
        ]
        @test A ≈ 3.55598088 atol=1e-6
        @test T ≈ T_exp atol=1e-6
        @test qgc ≈ [0.03196586, -0.10980295, 0.42891789] atol=1e-6
        @test local_corners[1] ≈ SVector(-0.79872060, -0.93162733, 0.0) atol=1e-6
        @test local_corners[2] ≈ SVector( 1.21632980, -0.99740524, 0.0) atol=1e-6
        @test local_corners[3] ≈ SVector( 0.50296217,  1.08966483, 0.0) atol=1e-6
        @test local_corners[4] ≈ SVector(-1.00050985,  1.02388691, 0.0) atol=1e-6

        cases = (
            (SVector(0.0, 0.0, 2.0), -0.16522813, [-0.01212182, 0.00696597, 0.08803249], SVector(0.02264691, 0.66182865, -1.42928215)),
            (SVector(1.0, -1.0, 0.812), -0.26489432, [0.26364529, -0.27501439, -0.06608018], SVector(-0.98982354, 0.94259724, 0.08973614)),
            (SVector(2.0, -0.110, 0.1), -0.14240410, [0.06825884, 0.00213854, -0.02095174], SVector(-0.14930027, 1.62102729, 1.15383073)),
            (SVector(0.5, -0.5, -0.1), -0.27506722, [0.10713328, -0.05606261, -0.17949664], SVector(-0.39798420, 0.15103078, 0.68546488)),
            (SVector(1.5, -0.5, -9.0), -0.02954344, [0.00048221, -0.00012699, -0.00303477], SVector(0.02264691, -2.80130259, 9.13039218)),
        )
        hessian_cases = (
            (SVector(0.0, 0.0, 2.0),
                [0.03946923 0.00396796 0.02181863; 0.00396796 0.04495038 -0.00996329; 0.02181863 -0.00996329 -0.08441961]),
            (SVector(1.0, -1.0, 0.812),
                [-0.12324138 1.20482974 0.69400251; 1.20482974 0.06958128 -1.03881191; 0.69400251 -1.03881191 0.05366010]),
        )
        for (p, φ_exp, v_exp, p_local_exp) in cases
            φ, v, p_local, H = birk_evaluate(p, T, qgc, local_corners)
            @test collect(p_local) ≈ collect(p_local_exp) atol=1e-6
            @test φ ≈ φ_exp atol=1e-6
            @test collect(v) ≈ v_exp atol=1e-5
            @test H[1, 1] + H[2, 2] + H[3, 3] ≈ 0 atol=1e-8
        end
        for (p, H_exp) in hessian_cases
            _, _, _, H = birk_evaluate(p, T, qgc, local_corners)
            @test collect(H) ≈ H_exp atol=1e-6
        end
    end

    @testset "Triangle panel (paper Fig. 7 / Table 5)" begin
        corners = (
            SVector(0.0, 0.0, 0.0),
            SVector(1.0, 0.0, 0.0),
            SVector(1.0, 1.0, 0.0),
            SVector(0.0, 0.0, 0.0),
        )
        T, qgc, local_corners, A = birk_setup(corners)
        T_exp = [
            0.89442719 -0.44721360 0.0;
            0.44721360  0.89442719 0.0;
            0.0         0.0        1.0
        ]
        @test A ≈ 0.5 atol=1e-8
        @test T ≈ T_exp atol=1e-6
        @test qgc ≈ [0.66666667, 0.33333333, 0.0] atol=1e-6

        cases = (
            (SVector(0.0, 0.0, 2.0), -0.01847387, [-0.00255504, -0.00124116, 0.00801178]),
            (SVector(0.0, 0.0, -0.5), -0.04558955, [-0.03515188, -0.01569148, -0.03689590]),
            (SVector(0.5, 1.0, 0.0), -0.05806854, [-0.03703739, 0.07533101, 0.0]),
            (SVector(-0.25, 0.25, 0.5), -0.03856218, [-0.03156649, 0.00016099, 0.02101846]),
            (SVector(0.1, 0.4, 8.0), -0.00495674, [-0.00004349, 0.00000518, 0.00061541]),
        )
        hessian_cases = (
            (SVector(0.0, 0.0, 2.0),
                [0.00284679 -0.00054992 0.00324874; -0.00054992 0.00365068 0.00155038; 0.00324874 0.00155038 -0.00649747]),
            (SVector(0.5, 1.0, 0.0),
                [-0.00000000 0.14235251 0.0; 0.14235251 -0.15915494 0.0; 0.0 0.0 0.15915494]),
        )
        for (p, φ_exp, v_exp) in cases
            φ, v, _, H = birk_evaluate(p, T, qgc, local_corners)
            @test φ ≈ φ_exp atol=1e-6
            @test collect(v) ≈ v_exp atol=1e-6
            @test H[1, 1] + H[2, 2] + H[3, 3] ≈ 0 atol=1e-8
        end
        for (p, H_exp) in hessian_cases
            _, _, _, H = birk_evaluate(p, T, qgc, local_corners)
            @test collect(H) ≈ H_exp atol=1e-6
        end
    end
end

@testset "Fused Birk φ+v matches separate loops" begin
    lc = (
        SVector(-0.5, -0.5, 0.0),
        SVector( 0.5, -0.5, 0.0),
        SVector( 0.5,  0.5, 0.0),
        SVector(-0.5,  0.5, 0.0),
    )
    for (x, y, z) in ((0.1, 0.2, 0.4), (0.0, 0.0, 0.0), (0.5, 1.0, 0.0), (-0.25, 0.25, 0.5))
        φ1 = MH.velocity_potential(x, y, z, lc)
        v1 = MH.velocity_derivatives(x, y, z, lc)
        φ2, v2 = MH.birk_phi_and_velocity(x, y, z, lc)
        @test φ1 ≈ φ2 atol=1e-14 rtol=1e-14
        @test v1 ≈ v2 atol=1e-14 rtol=1e-14
    end
    lc_tri = (SVector(-0.5, -0.5, 0.0), SVector(0.5, -0.5, 0.0), SVector(0.5, 0.5, 0.0), SVector(0.5, 0.5, 0.0))
    φ1 = MH.velocity_potential(0.1, 0.2, 0.3, lc_tri)
    v1 = MH.velocity_derivatives(0.1, 0.2, 0.3, lc_tri)
    φ2, v2 = MH.birk_phi_and_velocity(0.1, 0.2, 0.3, lc_tri)
    @test φ1 ≈ φ2 atol=1e-14 rtol=1e-14
    @test v1 ≈ v2 atol=1e-14 rtol=1e-14
end

@testset "Rankine matches DelhommeauRankine" begin
    e1 = (center=[0.0, 0.0, -1.0],)
    e2 = (
        center=[1.0, 1.0, -2.0],
        vertices=[-0.5 -0.5 0.0; 0.5 -0.5 0.0; 0.5 0.5 0.0; -0.5 0.5 0.0] .+ [1.0, 1.0, -2.0]',
        normal=[0.0, 0.0, 1.0],
        area=1.0,
        radius=sqrt(2)/2,
    )

    @test greens(Rankine(), e1, e2) ≈ greens(DelhommeauRankine(), e1, e2) atol=1e-12
    @test integral(Rankine(), e1, e2) ≈ integral(DelhommeauRankine(), e1, e2) rtol=1e-4 atol=1e-4
    @test integral_gradient(Rankine(), e1, e2; with_respect_to_first_variable=true) ≈
          integral_gradient(DelhommeauRankine(), e1, e2; with_respect_to_first_variable=true) rtol=1e-4 atol=1e-4
    @test integral_gradient(Rankine(), e1, e2; with_respect_to_first_variable=false) ≈
          integral_gradient(DelhommeauRankine(), e1, e2; with_respect_to_first_variable=false) rtol=1e-4 atol=1e-4

    @test greens(RankineReflected(), e1, e2) ≈ greens(DelhommeauRankineReflected(), e1, e2) atol=1e-12
    @test integral(RankineReflected(), e1, e2) ≈ integral(DelhommeauRankineReflected(), e1, e2) rtol=1e-4 atol=1e-4
    @test integral_gradient(RankineReflected(), e1, e2; with_respect_to_first_variable=true) ≈
          integral_gradient(DelhommeauRankineReflected(), e1, e2; with_respect_to_first_variable=true) rtol=1e-4 atol=1e-4
    @test integral_gradient(RankineReflected(), e1, e2; with_respect_to_first_variable=false) ≈
          integral_gradient(DelhommeauRankineReflected(), e1, e2; with_respect_to_first_variable=false) rtol=1e-4 atol=1e-4
end

@testset "DelhommeauRankine both_integral matches integral and integral_gradient" begin
    mesh = Mesh(MH.example_mesh_from_capytaine(4))
    for (i, j) in ((1, 2), (1, 1), (3, 5))
        e1 = element(mesh, i)
        e2 = element(mesh, j)
        for wrt_first in (true, false)
            S, G = MH.both_integral_and_integral_gradient(DelhommeauRankine(), e1, e2; with_respect_to_first_variable=wrt_first)
            @test S ≈ integral(DelhommeauRankine(), e1, e2) rtol=1e-12 atol=1e-12
            @test G ≈ integral_gradient(DelhommeauRankine(), e1, e2; with_respect_to_first_variable=wrt_first) rtol=1e-12 atol=1e-12
        end
    end
end

@testset "Rankine vs DelhommeauRankine on a sphere mesh" begin
    mesh = Mesh(MH.example_mesh_from_capytaine(4))

    for i in 1:mesh.nfaces
        for j in 1:mesh.nfaces
            e1 = element(mesh, i)
            e2 = element(mesh, j)
            @test integral(Rankine(), e1, e2) ≈ integral(DelhommeauRankine(), e1, e2) rtol=1e-3 atol=1e-4
            @test integral_gradient(Rankine(), e1, e2; with_respect_to_first_variable=true) ≈
                  integral_gradient(DelhommeauRankine(), e1, e2; with_respect_to_first_variable=true) rtol=1e-3 atol=1e-4
            @test integral_gradient(Rankine(), e1, e2; with_respect_to_first_variable=false) ≈
                  integral_gradient(DelhommeauRankine(), e1, e2; with_respect_to_first_variable=false) rtol=1e-3 atol=1e-4
        end
    end
end

@testset "Rankine influence matrices match DelhommeauRankine" begin
    mesh = Mesh(MH.example_mesh_from_capytaine())
    k = 1.0
    old_gfs = (DelhommeauRankine(), DelhommeauRankineReflected())
    new_gfs = (Rankine(), RankineReflected())
    S, D = assemble_matrices(old_gfs, mesh, k; direct=true)
    Sv, Dv = assemble_matrices(new_gfs, mesh, k; direct=true)
    @test Sv ≈ S rtol=1e-3 atol=1e-4
    @test Dv ≈ D rtol=1e-3 atol=1e-4

    S_ind, K = assemble_matrices(old_gfs, mesh, k; direct=false)
    Sv_ind, Kv = assemble_matrices(new_gfs, mesh, k; direct=false)
    @test Sv_ind ≈ S_ind rtol=1e-3 atol=1e-4
    @test Kv ≈ K rtol=1e-3 atol=1e-4
end

@testset "Vectorized StaticArraysMesh Rankine influence matrices match DelhommeauRankine" begin
    mesh = Mesh(MH.example_mesh_from_capytaine())
    smesh = StaticArraysMesh(mesh)
    k = 1.0
    old_gfs = (DelhommeauRankine(), DelhommeauRankineReflected())
    new_gfs = (Rankine(), RankineReflected())
    S, D = assemble_matrices(old_gfs, mesh, k; direct=true)
    Sv, Dv = assemble_matrices(new_gfs, smesh, k; direct=true)
    @test Sv ≈ S rtol=1e-3 atol=1e-4
    @test Dv ≈ D rtol=1e-3 atol=1e-4

    S_ind, K = assemble_matrices(old_gfs, mesh, k; direct=false)
    Sv_ind, Kv = assemble_matrices(new_gfs, smesh, k; direct=false)
    @test Sv_ind ≈ S_ind rtol=1e-3 atol=1e-4
    @test Kv ≈ K rtol=1e-3 atol=1e-4
end

function _compare_rankine_coefficients(floatingbody, parameters; rtol=1e-2, atol=1e-2)
    delhommeau_gfs = (DelhommeauRankine(), DelhommeauRankineReflected(), GFWu())
    rankine_gfs = (Rankine(), RankineReflected(), GFWu())
    data_d = hydrodynamic_coefficients(floatingbody, parameters, DirectBEM(delhommeau_gfs))
    data_r = hydrodynamic_coefficients(floatingbody, parameters, DirectBEM(rankine_gfs))
    @test data_r.added_mass ≈ data_d.added_mass rtol=rtol atol=atol
    @test data_r.radiation_damping ≈ data_d.radiation_damping rtol=rtol atol=atol
    @test data_r.diffraction_force ≈ data_d.diffraction_force rtol=rtol atol=atol
    @test data_r.excitation_force ≈ data_d.excitation_force rtol=rtol atol=atol
    @test data_r.Froude_Krylov_force ≈ data_d.Froude_Krylov_force rtol=1e-12 atol=1e-12
end

@testset "Rankine hydrodynamic coefficients match DelhommeauRankine" begin
    @testset "sphere" begin
        mesh = Mesh(MH.example_mesh_from_capytaine(4))
        floatingbody = FloatingBody(mesh, ["Heave", "Surge"], [0.0, 0.0, 0.0], "sphere")
        parameters = (
            wave_frequencies=[1.0, 2.0],
            wave_directions=[0.0],
            radiating_dofs=[:Heave, :Surge],
            influenced_dofs=[:Heave, :Surge],
        )
        _compare_rankine_coefficients(floatingbody, parameters)
    end

    @testset "horizontal cylinder" begin
        cpt = pyimport("capytaine")
        cptmesh = cpt.meshes.predefined.mesh_horizontal_cylinder(
            radius=1.5,
            center=(0.0, 0.0, 0.0),
            length=2.5,
            faces_max_radius=0.8,
        ).keep_immersed_part(inplace=true)
        floatingbody = FloatingBody(Mesh(cptmesh), ["Heave", "Pitch"], [0.0, 0.0, 0.0], "Horizontal_Cylinder")
        parameters = (
            wave_frequencies=[1.0, 1.5],
            wave_directions=[0.0],
            radiating_dofs=[:Heave, :Pitch],
            influenced_dofs=[:Heave, :Pitch],
        )
        _compare_rankine_coefficients(floatingbody, parameters)
    end
end
