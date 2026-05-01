using MarineHydro
using PyCall
# import your capytaine mesh
cpt = pyimport("capytaine")
radius = 1.0 #fixed
resolution = (10, 10)
cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution) 
cptmesh.keep_immersed_part(inplace=true)

# declare it Julia mesh
mesh = Mesh(cptmesh)  
ω = 1.03
ζ = [0,0,1] # HEAVE: will be more verbose in future iteration. define it again even if defined in Capytaine.

F = DiffractionForce(mesh,ω,ζ)
println("Excitation Force F: ", F)

A,B = calculate_radiation_forces(mesh,ζ,ω)
println("Addded Mass A: ", A)
println("Damping B: ", B)