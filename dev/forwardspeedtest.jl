using MarineHydro
using DifferentiationInterface 
import ForwardDiff 
using PyCall
using Test
using DimensionalData
cpt = pyimport("capytaine")

# function MH_compute_pressure(problem; direct::Bool=false, gf::String="Wu")
#     bc = compute_bc(problem) 

#     if problem.forward_speed==0
#         omega = problem.omega
#         wavenumber = problem.wavenumber
#     else
#         omega = problem.encountered_omega
#         wavenumber = problem.encountered_wavenumber # use encountered_wavenumber in gfs
#     end

#     if gf=="Wu"  
#         selected_GF = GFWu()
#     elseif gf=="ExactGuevelDelhommeau"
#         selected_GF = ExactGuevelDelhommeau()
#     end

#     S, D = assemble_matrices([Rankine(), RankineReflected(), selected_GF], problem.floatingbody.mesh, wavenumber; direct=direct)
#     potential, sources = solve(D, S, bc; direct=direct)

#     pressure = 1im * SETTINGS.rho * omega * potential # uses encountered_omega
#     uncorrcted_pressure = pressure # must be correct since encountered values and normal MH are correct

#     if problem.forward_speed!=0
#         if direct
#             # change normals to all be unit vector in x direction
#             S, D = assemble_matrices([Rankine(), RankineReflected(), selected_GF], problem.floatingbody.mesh, wavenumber; direct=direct, all_normals=[1, 0, 0])
#             nabla_phi_dot_x = S \ (D * potential)
#             K = D
#         else
#             # change normals to all be unit vector in x direction
#             S, K = assemble_matrices([Rankine(), RankineReflected(), selected_GF], problem.floatingbody.mesh, wavenumber; direct=direct, all_normals=[1, 0, 0])
#             nabla_phi_dot_x = K * sources
#         end
#         pressure .+= SETTINGS.rho * problem.forward_speed * nabla_phi_dot_x
#     else
#         K=D
#     end
#     forces = integrate_pressure(problem.floatingbody, problem.influenced_dofs, pressure)
#     return pressure, forces, K, sources, uncorrcted_pressure
# end


# omega = 0.5
# forward_speed = 0.2
# beta = 2

# t_DOFs = ["Surge","Sway"] # translational DOFs
# r_DOFs = ["Roll","Pitch"] # rotational DOFs
# DOFs = [t_DOFs; r_DOFs] # all DOFs

# cpt = pyimport("capytaine")
# radius = 1.0 #fixed
# resolution = (10, 10)
# cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution)
# cptmesh.keep_immersed_part(inplace=true)
# # cptmesh.translated((0,0,-1))

# # Create FloatingBody object
# cptbody = cpt.FloatingBody(mesh=cptmesh)
# cptbody.center_of_mass = (0.0, 0.0, 0.0)
# cptbody.rotation_center = (1.0, 1.0, 0.0) # off set for nonzero off-diagoinal elements
# foreach(dof -> cptbody.add_translation_dof(name=dof), t_DOFs)
# foreach(dof -> cptbody.add_rotation_dof(name=dof), r_DOFs)
# cptbody.active_dofs = DOFs
# cptbody.name = "Horizontal Cylinder"

# # Get MarineHydro values
# mesh = Mesh(cptmesh)
# rigid_dof_list = DOFs
# rotation_center = collect(cptbody.rotation_center)
# floatingbody = FloatingBody(mesh, rigid_dof_list, rotation_center, "Horizontal_Cylinder")

# cpt_dif_prob = cpt.DiffractionProblem(body=cptbody,water_depth=Inf,omega=omega,forward_speed=forward_speed,wave_direction=beta)
# cpt_dif_bc = cpt_dif_prob.boundary_condition
# mh_dif_prob = DiffractionProblem(floatingbody, omega, beta, compute_wavenumber(omega), forward_speed, Symbol.(DOFs))
# mh_dif_bc = compute_bc(mh_dif_prob)

# cpt_solver = cpt.BEMSolver()
# cpt_dif_result = cpt_solver.solve(problem=cpt_dif_prob,method="indirect")    
# cpt_dif_pressure = cpt_dif_result.pressure

# mh_dif_pressure, mh_dif_forces, mh_K, mh_dif_sources, mh_uncorrcted_pressure = MH_compute_pressure(mh_dif_prob;direct=false)

# gf_params = Dict("free_surface" => cpt_dif_result.free_surface, "water_depth" => cpt_dif_result.water_depth, "wavenumber" => cpt_dif_result.encounter_wavenumber)
# dGF = cpt.Delhommeau()
# S, cpt_K = dGF.evaluate(cptmesh, cptmesh; free_surface = cpt_dif_result.free_surface, water_depth = cpt_dif_result.water_depth, wavenumber = cpt_dif_result.encounter_wavenumber, early_dot_product=false)


# num_vals = 10
# @testset "Forward speed tests" begin
#     for i in 1:num_vals
#         for j in 1:num_vals
#             @testset "K matrix at $i and $j" begin
#                 @test mh_K[i,j] ≈ cpt_K[i,j,1] atol=1e-3 rtol = 1e-3
#             end
#         end
#     end

#     @testset "Diffraction boundary conditions" begin
#         @test cpt_dif_bc ≈ mh_dif_bc atol=1e-4 rtol = 1e-4
#     end

#     @testset "Diffraction sources" begin
#         @test cpt_dif_result.sources ≈ mh_dif_sources atol=1e-4 rtol = 1e-4        
#     end

#     @testset "Diffraction pressure" begin
#         @test cpt_dif_pressure ≈ mh_dif_pressure atol=1e-4 rtol = 1e-4
#     end

#     for rad_dof in DOFs
#         cpt_rad_prob = cpt.RadiationProblem(body=cptbody,water_depth=Inf,omega=omega,forward_speed=forward_speed,radiating_dof=rad_dof,wave_direction=beta)
#         cpt_rad_bc = cpt_rad_prob.boundary_condition
#         mh_rad_prob = RadiationProblem(floatingbody, omega, beta, compute_wavenumber(omega), forward_speed, Symbol(rad_dof), Symbol.(DOFs))
#         mh_rad_bc = compute_bc(mh_rad_prob)
#         @testset "Radiation boundary condition for rad_dof=$rad_dof" begin
#             @test cpt_rad_bc ≈ mh_rad_bc atol=1e-4 rtol = 1e-4
#         end
#         @testset "Encountered properties" begin
#             @test cpt_rad_prob.encounter_omega ≈ mh_rad_prob.encountered_omega atol=1e-4 rtol = 1e-4
#             @test cpt_rad_prob.encounter_wavenumber ≈ mh_rad_prob.encountered_wavenumber atol=1e-4 rtol = 1e-4
#         end
#     end  


# end


forward_speeds = [0.0] # forward speed in x direction [m/s]
h = Inf # sea depth [m]
omegas = 0.5:0.5:2 # frequencies [rad/s]
betas = [pi/6] # incident wave angle [rad]
t_DOFs = ["Surge","Sway"] # translational DOFs
r_DOFs = ["Roll","Pitch"] # rotational DOFs
DOFs = [t_DOFs; r_DOFs] # all DOFs
method = "indirect"
if method == "direct"
    direct = true
elseif method == "indirect"
    direct = false
end
gf = "ExactGuevelDelhommeau"
# gf = "Wu"


# Create Mesh object
radius = 1.5  
center = (0.0,0.0,0.0) 
len = 2.5
faces_max_radius = 0.5
cptmesh = cpt.meshes.predefined.mesh_horizontal_cylinder(
            radius=radius,
            center=center, 
            length=len, 
            faces_max_radius = faces_max_radius
            ).keep_immersed_part(inplace=true)

# Create FloatingBody object
cptbody = cpt.FloatingBody(mesh=cptmesh)
cptbody.center_of_mass = (0.0, 0.0, 0.0)
cptbody.rotation_center = (1.0, 1.0, 0.0) # off set for nonzero off-diagoinal elements
foreach(dof -> cptbody.add_translation_dof(name=dof), t_DOFs)
foreach(dof -> cptbody.add_rotation_dof(name=dof), r_DOFs)
cptbody.active_dofs = DOFs
cptbody.name = "Horizontal Cylinder"

# Setup and solve BEM problems
solver = cpt.BEMSolver()
dof_list = cptbody.active_dofs
xr = pyimport("xarray")
test_matrix = xr.Dataset(coords=Dict("omega" => omegas,
    "wave_direction" => betas,
    "radiating_dof" => DOFs,
    "forward_speed" => forward_speeds[1]))
results = cpt.BEMSolver().fill_dataset(test_matrix, cptbody, method=method)    

# Get Capytaine values
A_cpt = results.added_mass
B_cpt = results.radiation_damping
F_FK_cpt = results.Froude_Krylov_force 
F_D_cpt = results.diffraction_force
F_ex_cpt = results.excitation_force

# Get MarineHydro values
mesh = Mesh(cptmesh)
rigid_dof_list = DOFs
rotation_center = collect(cptbody.rotation_center)
floatingbody = FloatingBody(mesh, rigid_dof_list, rotation_center, "Horizontal_Cylinder")

parameters = (wave_frequencies=omegas, 
    wave_directions=betas,
    radiating_dofs=Symbol.(DOFs),
    influenced_dofs=Symbol.(DOFs),
    forward_speeds=forward_speeds)

mhresults = compute_and_label_hydrodynamic_coefficients(parameters, floatingbody; direct=direct, gf=gf)

A_mh = mhresults.added_mass
B_mh = mhresults.radiation_damping
F_FK_mh = mhresults.Froude_Krylov_force
F_D_mh = mhresults.diffraction_force
F_ex_mh = mhresults.excitation_force

display(A_mh)
