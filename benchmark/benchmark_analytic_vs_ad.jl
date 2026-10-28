# Analytic Birk kernels vs assembled AD (ForwardDiff through the formulas,
# Zygote reverse mode via the ChainRules rrules).
#
# The existing `benchmark/` environment does not list Zygote/ForwardDiff, so
# run this from the package project:
#
#   julia --project=. benchmark/benchmark_analytic_vs_ad.jl

using ForwardDiff
using Zygote
using StaticArrays
using LinearAlgebra
using MarineHydro
const MH = MarineHydro

function mintime(f, n=200)
    f()
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

function showb(label, t)
    println(rpad(label, 36), " ", round(t * 1e6; sigdigits=3), " μs")
end

local_corners = (
    SVector(-0.5, -0.5, 0.0),
    SVector( 0.5, -0.5, 0.0),
    SVector( 0.5,  0.5, 0.0),
    SVector(-0.5,  0.5, 0.0),
)
x, y, z = -0.25, 0.25, 0.5
field = [x, y, z]

element_1_center = [0.0, 0.0, -1.0]
element_2 = (
    center=[1.0, 1.0, -2.0],
    vertices=[-0.5 -0.5 0.0; 0.5 -0.5 0.0; 0.5 0.5 0.0; -0.5 0.5 0.0] .+ [1.0, 1.0, -2.0]',
    normal=[0.0, 0.0, 1.0],
    radius=sqrt(2) / 2,
    area=1.0,
)

println("=== kernel, field point (x,y,z) ===")
showb("analytic velocity", mintime(() -> MH.velocity_derivatives(x, y, z, local_corners)))
showb("analytic Hessian", mintime(() -> MH.velocity_hessian(x, y, z, local_corners)))
showb("Zygote ∇φ (rrule = velocity)", mintime(() -> Zygote.gradient((a, b, c) -> MH.velocity_potential(a, b, c, local_corners), x, y, z)))
showb("FD ∇φ (AD through φ)", mintime(() -> ForwardDiff.gradient(q -> MH.velocity_potential(q[1], q[2], q[3], local_corners), field)))
showb("Zygote J(v) (rrule = Hessian)", mintime(() -> Zygote.jacobian(q -> MH.velocity_derivatives(q[1], q[2], q[3], local_corners), field)))
showb("FD J(v) (AD through v)", mintime(() -> ForwardDiff.jacobian(q -> MH.velocity_derivatives(q[1], q[2], q[3], local_corners), field)))
showb("corners adjoint of φ (FD)", mintime(() -> MH._potential_corners_adjoint(1.0, x, y, z, local_corners)))
showb("corners adjoint of v (FD)", mintime(() -> MH._velocity_corners_adjoint(SVector(1.0, 0.0, 0.0), x, y, z, local_corners)))

println()
println("=== integral / gradient wrt field center ===")
showb("analytic integral_gradient", mintime(() -> integral_gradient(Rankine(), (center=element_1_center,), element_2; with_respect_to_first_variable=true)))
showb("Zygote ∇∫ Rankine", mintime(() -> Zygote.gradient(c -> integral(Rankine(), (center=c,), element_2), element_1_center), 80))
showb("FD ∇∫ Rankine", mintime(() -> ForwardDiff.gradient(c -> integral(Rankine(), (center=c,), element_2), element_1_center), 80))
showb("Zygote ∇∫ DelhommeauRankine", mintime(() -> Zygote.gradient(c -> integral(DelhommeauRankine(), (center=c,), element_2), element_1_center), 80))
showb("FD ∇∫ DelhommeauRankine", mintime(() -> ForwardDiff.gradient(c -> integral(DelhommeauRankine(), (center=c,), element_2), element_1_center), 80))
showb("Zygote J(∇∫) Rankine", mintime(() -> Zygote.jacobian(c -> collect(integral_gradient(Rankine(), (center=c,), element_2; with_respect_to_first_variable=true)), element_1_center), 40))
showb("FD J(∇∫) Rankine", mintime(() -> ForwardDiff.jacobian(c -> collect(integral_gradient(Rankine(), (center=c,), element_2; with_respect_to_first_variable=true)), element_1_center), 40))
showb("Zygote J(∇∫) DelhommeauRankine", mintime(() -> Zygote.jacobian(c -> collect(integral_gradient(DelhommeauRankine(), (center=c,), element_2; with_respect_to_first_variable=true)), element_1_center), 40))
showb("FD J(∇∫) DelhommeauRankine", mintime(() -> ForwardDiff.jacobian(c -> collect(integral_gradient(DelhommeauRankine(), (center=c,), element_2; with_respect_to_first_variable=true)), element_1_center), 40))
