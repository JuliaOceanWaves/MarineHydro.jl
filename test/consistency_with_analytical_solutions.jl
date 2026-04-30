using MarineHydro
using Test
using PyCall


cpt = pyimport("capytaine")
radius = 1.0 #fixed
resolution = (10, 10)
cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution)
cptmesh.keep_immersed_part(inplace=true)
mesh = Mesh(cptmesh)

@testset "Comparison with analytical added mass and radiation damping using exact Delhommeau  rtol=1e-1" begin
    K_heave = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    A_heave = [0.8764, 0.8627, 0.7938, 0.7157, 0.6452, 0.5861, 0.5381, 0.4999, 0.4698, 0.4464, 0.4284, 0.4047, 0.3924, 0.3871, 0.3864, 0.3884, 0.3988, 0.4111, 0.4322, 0.4471, 0.4574, 0.4647, 0.4700, 0.4740, 0.4771]

    B_heave = [0.1036, 0.1816, 0.2793, 0.3254, 0.3410, 0.3391, 0.3271, 0.3098, 0.2899, 0.2691, 0.2484, 0.2096, 0.1756, 0.1469, 0.1229, 0.1031, 0.0674, 0.0452, 0.0219, 0.0116, 0.0066, 0.0040, 0.0026, 0.0017, 0.0012]

    rho = 1000
    ω = sqrt.(K_heave .* 9.8)
    heave = [0.0, 0.0, 1.0]
    results_vec = [calculate_radiation_forces(mesh,heave,omega) for omega in ω]
    A_bem = [results[1]./((2/3)*pi*radius*rho)  for results in results_vec]
    @test A_bem ≈ A_heave  rtol=1e-1
    B_bem = [results[2]./((2/3)*pi*radius*rho)  for results in results_vec]
    ##dimensional for heave and damping(requires /wn)
    B_bem = B_bem./ω
    @test B_bem ≈ B_heave  rtol=1e-1
end


@testset "Comparison with analytical added mass and radiation damping using Wu gf  rtol=1e-1" begin
    #doi:10.1017/S0022112082001980
    K_heave = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    A_heave = [0.8764, 0.8627, 0.7938, 0.7157, 0.6452, 0.5861, 0.5381, 0.4999, 0.4698, 0.4464, 0.4284, 0.4047, 0.3924, 0.3871, 0.3864, 0.3884, 0.3988, 0.4111, 0.4322, 0.4471, 0.4574, 0.4647, 0.4700, 0.4740, 0.4771]

    B_heave = [0.1036, 0.1816, 0.2793, 0.3254, 0.3410, 0.3391, 0.3271, 0.3098, 0.2899, 0.2691, 0.2484, 0.2096, 0.1756, 0.1469, 0.1229, 0.1031, 0.0674, 0.0452, 0.0219, 0.0116, 0.0066, 0.0040, 0.0026, 0.0017, 0.0012]

    rho = 1025
    SETTINGS.rho = rho
    ω = sqrt.(K_heave .* 9.8)
    heave = [0.0, 0.0, 1.0]
    results_vec = [calculate_radiation_forces(mesh,heave,omega) for omega in ω]
    A_bem = [results[1]./((2/3)*pi*radius*rho)  for results in results_vec]
    @test A_bem ≈ A_heave  rtol=1e-1
    B_bem = [results[2]./((2/3)*pi*radius*rho)  for results in results_vec]
    ##dimensional for heave and damping(requires /wn)
    B_bem = B_bem./ω
    @test B_bem ≈ B_heave  rtol=1e-1

end

