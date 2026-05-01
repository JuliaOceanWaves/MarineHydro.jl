using MarineHydro
using PyCall
cpt = pyimport("capytaine")

# Description of problem
h = Inf # sea depth [m]
omegas = [1.03, 1.7] # frequencies [rad/s]
beta = 0 # incident wave angle [rad]
t_DOFs = ["Heave"] # translational DOFs
r_DOFs = ["Pitch"] # rotational DOFs
DOFs = [t_DOFs; r_DOFs] # all DOFs

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
foreach(dof -> cptbody.add_translation_dof(name=dof), t_DOFs)
foreach(dof -> cptbody.add_rotation_dof(name=dof), r_DOFs)
cptbody.active_dofs = DOFs
cptbody.name = "Body 1"
cptbody.center_of_mass = (0, 0, 0)
cptbody.rotation_center = (0, 0, 0)

# Setup and solve BEM problems
solver = cpt.BEMSolver()
dof_list = cptbody.active_dofs
xr = pyimport("xarray")
test_matrix = xr.Dataset(coords=Dict("omega" => omegas, "wave_direction" => [0.0], "radiating_dof" => DOFs))
results = cpt.BEMSolver().fill_dataset(test_matrix, cptbody, method="direct")
 

# Get Capytaine values
num_dof = length(results.influenced_dof)
num_omega = length(omegas)
A_cpt = reshape(results.added_mass.values,num_dof,num_dof, num_omega)
B_cpt = reshape(results.radiation_damping.values,num_dof,num_dof, num_omega)
F_FK_cpt = reshape(results.Froude_Krylov_force.values,num_dof,1,num_omega)
F_D_cpt = reshape(results.diffraction_force.values,num_dof,1,num_omega)

# Get MarineHydro values
mesh = Mesh(cptmesh)
rigid_dof_list = ["Heave"]
rotation_center = Float64[cptbody.rotation_center]
fb = FloatingBody(mesh, rigid_dof_list, rotation_center)

omega = omegas[1]

A_MH,B_MH = calculate_radiation_forces(fb, omega)
F_FK_MH = FroudeKrylovForce(fb, omega)
F_D_MH = DiffractionForce(fb,omega)

