using Test
using MarineHydro
using ForwardDiff
using FiniteDifferences
using StaticArrays

const DIF_HAS_ENZYME = try
    using Enzyme
    true
catch
    false
end

const dif_mesher = make_sphere_mesher(resolution=4)
const dif_r0 = 1.0
const dif_ω0 = 1.2
const dif_β0 = 0.0
const dif_formulation = DirectBEM()
const dif_h = 1e-5

function dif_solve(r, ω, mesher)
    body = FloatingBody(StaticArraysMesh(mesher(r)), [:Heave], "s")
    return solve(DiffractionProblem(body, ω; beta=dif_β0), dif_formulation)
end

reD(r, ω, mesher) = real(diffraction_force(dif_solve(r, ω, mesher)).Heave)
imD(r, ω, mesher) = imag(diffraction_force(dif_solve(r, ω, mesher)).Heave)
reK(r, ω, mesher) = real(froude_krylov_force(dif_solve(r, ω, mesher)).Heave)
imK(r, ω, mesher) = imag(froude_krylov_force(dif_solve(r, ω, mesher)).Heave)
reE(r, ω, mesher) = real(excitation_force(dif_solve(r, ω, mesher)).Heave)
imE(r, ω, mesher) = imag(excitation_force(dif_solve(r, ω, mesher)).Heave)

# Enzyme needs the active argument first.
reD_ω(ω, r, mesher) = reD(r, ω, mesher)
imD_ω(ω, r, mesher) = imD(r, ω, mesher)
reK_ω(ω, r, mesher) = reK(r, ω, mesher)
imK_ω(ω, r, mesher) = imK(r, ω, mesher)
reE_ω(ω, r, mesher) = reE(r, ω, mesher)
imE_ω(ω, r, mesher) = imE(r, ω, mesher)

const DIF_OUTPUTS = (
    ("Re(F_D)", reD, reD_ω),
    ("Im(F_D)", imD, imD_ω),
    ("Re(F_FK)", reK, reK_ω),
    ("Im(F_FK)", imK, imK_ω),
    ("Re(F_ex)", reE, reE_ω),
    ("Im(F_ex)", imE, imE_ω),
)

@testset "DiffractionProblem primal Mesh vs smesh" begin
    body_m = FloatingBody(dif_mesher(dif_r0), [:Heave], "m")
    body_s = FloatingBody(StaticArraysMesh(dif_mesher(dif_r0)), [:Heave], "s")
    sol_m = solve(DiffractionProblem(body_m, dif_ω0; beta=dif_β0), dif_formulation)
    sol_s = solve(DiffractionProblem(body_s, dif_ω0; beta=dif_β0), dif_formulation)
    @test sol_m isa DiffractionResult
    @test sol_s isa DiffractionResult
    @test sol_s.forces.Heave ≈ sol_m.forces.Heave rtol=1e-8 atol=1e-10
    @test froude_krylov_force(sol_s).Heave ≈ froude_krylov_force(sol_m).Heave rtol=1e-8 atol=1e-10
    @test excitation_force(sol_s).Heave ≈ excitation_force(sol_m).Heave rtol=1e-8 atol=1e-10
    @test isfinite(real(excitation_force(sol_s).Heave))
end

@testset "DiffractionProblem ForwardDiff vs FD" begin
    for (name, f, _) in DIF_OUTPUTS
        @testset "$name d/dω" begin
            fwd = ForwardDiff.derivative(w -> f(dif_r0, w, dif_mesher), dif_ω0)
            fd = (f(dif_r0, dif_ω0 + dif_h, dif_mesher) - f(dif_r0, dif_ω0 - dif_h, dif_mesher)) / (2dif_h)
            @test isfinite(fwd)
            @test fwd ≈ fd rtol=1e-3 atol=1e-3
        end
        @testset "$name d/dr" begin
            fwd = ForwardDiff.derivative(r -> f(r, dif_ω0, dif_mesher), dif_r0)
            fd = (f(dif_r0 + dif_h, dif_ω0, dif_mesher) - f(dif_r0 - dif_h, dif_ω0, dif_mesher)) / (2dif_h)
            @test isfinite(fwd)
            @test fwd ≈ fd rtol=1e-3 atol=1e-3
        end
    end
end

@testset "DiffractionProblem Enzyme reverse vs ForwardDiff" begin
    if !DIF_HAS_ENZYME
        @test_skip "Enzyme not installed"
    else
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        for (name, f, fω) in DIF_OUTPUTS
            @testset "$name" begin
                dω_fwd = ForwardDiff.derivative(w -> f(dif_r0, w, dif_mesher), dif_ω0)
                dω_ez = first(Enzyme.gradient(mode, fω, dif_ω0, Enzyme.Const(dif_r0), Enzyme.Const(dif_mesher)))
                @test dω_ez ≈ dω_fwd rtol=1e-5 atol=1e-6

                dr_fwd = ForwardDiff.derivative(r -> f(r, dif_ω0, dif_mesher), dif_r0)
                dr_ez = first(Enzyme.gradient(mode, f, dif_r0, Enzyme.Const(dif_ω0), Enzyme.Const(dif_mesher)))
                @test dr_ez ≈ dr_fwd rtol=1e-5 atol=1e-6
            end
        end
    end
end
