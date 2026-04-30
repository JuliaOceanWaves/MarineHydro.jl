using BenchmarkTools
using MarineHydro 
using PyCall
using Zygote

cpt = pyimport("capytaine")
radius = 1.0
resolution = (6, 6) #smaller data just to test
cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution)
cptmesh.keep_immersed_part(inplace=true)
mesh = MarineHydro.Mesh(cptmesh)
npanels = mesh.nfaces
println( "Mesh information $resolution with  panels $npanels")
omega = 1.12
k = omega^2 / 9.8
dof = [0.0,0.0,1.0]

function check_added_mass(ω,mesh,dof;green_functions)
    added_mass_value = MarineHydro.calculate_radiation_forces(mesh, dof, ω)[1]
    return added_mass_value
end

green_functions = (
        Rankine(),
        RankineReflected(),
         GFWu())
A(w) = check_added_mass(w, mesh, dof;green_functions)
benchmark_result_amass = @benchmark check_added_mass(omega, mesh, dof;green_functions)
println(benchmark_result_amass)

A_w_grad, = Zygote.gradient(A,omega)
benchmark_result_gradient = @benchmark Zygote.gradient(A,omega)

println(benchmark_result_gradient)

