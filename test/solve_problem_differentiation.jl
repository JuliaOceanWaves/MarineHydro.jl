using Test
using MarineHydro
using PyCall
using ForwardDiff
using FiniteDifferences
using StaticArrays
using DifferentiationInterface
import DifferentiationInterface as DI

const MH = MarineHydro

const HAS_ENZYME = try
    using Enzyme
    true
catch
    false
end

cpt = pyimport("capytaine")
cpt_mesh = cpt.mesh_sphere(name="sphere", radius=1.0, center=(0, 0, 0), resolution=(4, 4)).keep_immersed_part()
hull = Mesh(cpt_mesh)
cpt_lid = cpt_mesh.generate_lid(z=-0.05)
lid = Mesh(cpt_lid)

body_nolid = FloatingBody(hull, [:Heave], "nolid")
body_lid = FloatingBody(hull, lid, [:Heave], "withlid")
smesh = StaticArraysMesh(cpt_mesh)
body_smesh = FloatingBody(smesh, [:Heave], "smesh")
const ω = 1.2
const formulation = DirectBEM()

heave_force(body, w) = real(solve(RadiationProblem(body, w), formulation).forces[:Heave])
heave_A(body, w) = added_mass(solve(RadiationProblem(body, w), formulation)).Heave

function smesh_from_radius(r, smesh0)
    s = r
    z = zero(s)
    return StaticArraysMesh(
        [s * v for v in smesh0.vertices],
        smesh0.faces,
        [s * c for c in smesh0.centers],
        [n .+ z for n in smesh0.normals],
        [a * (s * s) for a in smesh0.areas],
        [ρ * s for ρ in smesh0.radii],
        smesh0.nvertices,
        smesh0.nfaces,
    )
end

function heave_A_from_radius(r, smesh0, w)
    body = FloatingBody(smesh_from_radius(r, smesh0), [:Heave], "scaled")
    return heave_A(body, w)
end

enzyme_A_of_omega(w, body) = heave_A(body, w)

@testset "solve lid plumbing" begin
    combined = hull + lid
    @test combined.nfaces == hull.nfaces + lid.nfaces
    mask = eachrow(combined.faces) .∈ Ref(eachrow(hull.faces))
    @test count(mask) == hull.nfaces

    bc_hull = fill(1.0 + 2.0im, hull.nfaces)
    padded = MH._pad_hull_bc(bc_hull, mask)
    @test length(padded) == combined.nfaces
    @test padded[mask] == bc_hull
    @test all(iszero, padded[.!mask])
    @test MH._pad_hull_bc(bc_hull, trues(hull.nfaces)) == bc_hull
end

@testset "solve primal with and without lid" begin
    F0 = heave_force(body_nolid, ω)
    F1 = heave_force(body_lid, ω)
    @test isfinite(F0)
    @test isfinite(F1)
    res = solve(RadiationProblem(body_smesh, ω), formulation)
    @test added_mass(res).Heave ≈ real(res.forces[:Heave]) / ω^2
    @test radiation_damping(res).Heave ≈ imag(res.forces[:Heave]) / ω
end

@testset "remake omega then solve" begin
    prob0 = RadiationProblem(body_smesh, ω)
    dω = ForwardDiff.Dual(ω, 1.0)
    res = solve(remake(prob0; omega=dω), formulation)
    @test res isa RadiationResult
    @test added_mass(res).Heave isa ForwardDiff.Dual
end

@testset "solve dA/dω no lid" begin
    fd = ForwardDiff.derivative(w -> heave_A(body_nolid, w), ω)
    cfd = FiniteDifferences.central_fdm(5, 1)(w -> heave_A(body_nolid, w), ω)
    @test fd ≈ cfd rtol=1e-4 atol=1e-4
end

@testset "solve dA/dω with lid" begin
    fd = ForwardDiff.derivative(w -> heave_A(body_lid, w), ω)
    cfd = FiniteDifferences.central_fdm(5, 1)(w -> heave_A(body_lid, w), ω)
    @test fd ≈ cfd rtol=1e-4 atol=1e-4
end

@testset "hydrodynamic_coefficients Rankine reuse is differentiable in ω" begin
    function A_two_freq(w)
        parameters = (wave_frequencies=[w, w + 0.25], radiating_dofs=[:Heave], influenced_dofs=[:Heave])
        data = hydrodynamic_coefficients(body_nolid, parameters, formulation)
        return sum(data.added_mass)
    end
    fd = ForwardDiff.derivative(A_two_freq, ω)
    cfd = FiniteDifferences.central_fdm(5, 1)(A_two_freq, ω)
    @test fd ≈ cfd rtol=1e-4 atol=1e-4
end

@testset "smesh FloatingBody through solve(prob, formulation)" begin
    @test body_smesh isa FloatingBody{<:StaticArraysMesh}
    @test body_nolid isa FloatingBody{<:Mesh}
    F_dense = heave_force(body_nolid, ω)
    F_smesh = heave_force(body_smesh, ω)
    @test F_smesh ≈ F_dense rtol=1e-8 atol=1e-10

    fd_ω = ForwardDiff.derivative(w -> heave_A(body_smesh, w), ω)
    cfd_ω = FiniteDifferences.central_fdm(5, 1)(w -> heave_A(body_smesh, w), ω)
    @test fd_ω ≈ cfd_ω rtol=1e-4 atol=1e-4

    fd_r = ForwardDiff.derivative(r -> heave_A_from_radius(r, smesh, ω), 1.0)
    @test isfinite(fd_r)
    @test abs(fd_r) > 0

    function A_two_freq_smesh(w)
        parameters = (wave_frequencies=[w, w + 0.25], radiating_dofs=[:Heave], influenced_dofs=[:Heave])
        data = hydrodynamic_coefficients(body_smesh, parameters, formulation)
        return sum(data.added_mass)
    end
    fd_all = ForwardDiff.derivative(A_two_freq_smesh, ω)
    cfd_all = FiniteDifferences.central_fdm(5, 1)(A_two_freq_smesh, ω)
    @test fd_all ≈ cfd_all rtol=1e-4 atol=1e-4

    fwd = AutoForwardDiff()
    dA_di = DI.derivative(w -> heave_A(body_smesh, w), fwd, ω)
    @test dA_di ≈ fd_ω rtol=1e-8 atol=1e-10

    if HAS_ENZYME
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        ez_ω = first(Enzyme.gradient(mode, enzyme_A_of_omega, ω, Enzyme.Const(body_smesh)))
        @test ez_ω ≈ fd_ω rtol=1e-5 atol=1e-6

        ez_r = first(Enzyme.gradient(mode, heave_A_from_radius, 1.0,
            Enzyme.Const(smesh), Enzyme.Const(ω)))
        @test ez_r ≈ fd_r rtol=1e-5 atol=1e-6
    else
        @test_skip "Enzyme not installed"
    end
end

@testset "hydrodynamic_coefficients Enzyme reverse vs ForwardDiff" begin
    if !HAS_ENZYME
        @test_skip "Enzyme not installed"
    else
        fwd = AutoForwardDiff()
        rev = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

        function A_two_freq_smesh(w)
            parameters = (wave_frequencies=[w, w + 0.25], radiating_dofs=[:Heave], influenced_dofs=[:Heave])
            data = hydrodynamic_coefficients(body_smesh, parameters, formulation)
            return sum(data.added_mass)
        end
        dA_fwd = DI.derivative(A_two_freq_smesh, fwd, ω)
        dA_rev = DI.derivative(A_two_freq_smesh, rev, ω)
        @test dA_rev ≈ dA_fwd rtol=1e-5 atol=1e-6

        function Fex_two_freq(w)
            parameters = (wave_frequencies=[w, w + 0.25], radiating_dofs=[:Heave],
                wave_directions=[0.0], influenced_dofs=[:Heave])
            data = hydrodynamic_coefficients(body_smesh, parameters, formulation)
            return sum(real, data.excitation_force)
        end
        dF_fwd = DI.derivative(Fex_two_freq, fwd, ω)
        dF_rev = DI.derivative(Fex_two_freq, rev, ω)
        @test dF_rev ≈ dF_fwd rtol=1e-5 atol=1e-6
    end
end
