using MarineHydro
using Zygote
using FiniteDifferences
using Test
using ColorTypes
using Plots
include("MeshGradients_singlebody.jl")

orange = RGB(230/255,159/255,0/255) 
vermillion = RGB(213/255, 94/255, 0/255) 
bluishgreen = RGB(0/255, 158/255, 115/255) 

function radiation_forces(mesh::Mesh, dof, omega; direct = false)
    k = omega^2 / 9.8
    S, D = assemble_matrix_wu(mesh, k;direct)
    bc = radiation_bc(mesh, dof, omega)
    potential = solve(D, S, bc)
    pressure = 1im * 1023 * omega * potential
    forces = integrate_pressure(mesh, pressure, dof)
    return [real(forces)/omega^2, imag(forces)/omega]
end


function added_mass_program(radius,omega,dof,direct)  
    mesh = differentiableMesh(radius) #fd
    A = radiation_forces(mesh,dof,omega;direct)[1]
    return A
end

function damping_program(radius,omega,dof,direct)  
    mesh = differentiableMesh(radius) #fd
    B = radiation_forces(mesh,dof,omega;direct)[2]
    return B
end

green_functions = (
        Rankine(),
        RankineReflected(),
        GFWu(),
    )

radius_range = [1,2,3]
w = 1.03
r = 1.0
heave = [0,0,1]


indirect_gradients = Array{Any}(undef, length(radius_range))
direct_gradients = Array{Any}(undef, length(radius_range))

for (i, r) in enumerate(radius_range)
        grads_indirect = Zygote.gradient(X -> added_mass_program(X[1], X[2], X[3],false), [r, w, heave])[1][1]
        indirect_gradients[i] = grads_indirect
        grads_direct = Zygote.gradient(X -> added_mass_program(X[1], X[2], X[3],true), [r, w, heave])[1][1]
        direct_gradients[i] = grads_direct
    end

plot(collect(radius_range), indirect_gradients , xlabel="r(radius)", ylabel="∂A_∂r", label="indirect BIE", marker = "*",color = vermillion,linestyle = :dash)
plot!(collect(radius_range), direct_gradients, xlabel="r(radius)", ylabel="∂A_∂r", label="direct BIE",marker = "*",color = bluishgreen,linestyle = :solid)

savefig("/15183/MarineHydro.jl/dev/direct_indirect_ad_added_mass_heave_with_radius.pdf")


for (i, r) in enumerate(radius_range)
    grads_indirect = Zygote.gradient(X -> damping_program(X[1], X[2], X[3],false), [r, w, heave])[1][1]
    indirect_gradients[i] = grads_indirect
    grads_direct = Zygote.gradient(X -> damping_program(X[1], X[2], X[3],true), [r, w, heave])[1][1]
    direct_gradients[i] = grads_direct
end

plot(collect(radius_range), indirect_gradients , xlabel="r(radius)", ylabel="∂B_∂r", label="indirect BIE", marker = "*",color = vermillion,linestyle = :dash)
plot!(collect(radius_range), direct_gradients, xlabel="r(radius)", ylabel="∂B_∂r", label="direct BIE",marker = "*",color = bluishgreen,linestyle = :solid)

savefig("/15183/MarineHydro.jl/dev/direct_indirect_ad_damping_heave_with_radius.pdf")
