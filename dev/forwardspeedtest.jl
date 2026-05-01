using MarineHydro
using DifferentiationInterface 
import ForwardDiff 
using PyCall
using Test
using DimensionalData
cpt = pyimport("capytaine")

@testset "Hydrodynamic Coefficient Comparison with Capytaine for MDOF Horizontal Cylinder (atol=1e-4 rtol = 1e-1) " begin
    # Description of problem
    forward_speeds = [0.01] # forward speed in x direction [m/s]
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
    # gf = "ExactGuevelDelhommeau"
    gf = "Wu"
   

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

    for forward_speed in forward_speeds
        for omega in omegas
            for influenced_dof in DOFs
                for radiating_dof in DOFs
                    beta = betas[1]
                    @testset "Omega: $omega, influenced_dof: $influenced_dof, radiating_dof: $radiating_dof" begin
                        # Test added mass
                        a_cpt = A_cpt.sel(omega=omega, radiating_dof=radiating_dof, influenced_dof=influenced_dof).values[]
                        a_mh = A_mh[influenced_dofs = At(Symbol(influenced_dof)),
                            radiating_dofs = At(Symbol(radiating_dof)),
                            wave_frequencies = At(omega),
                            forward_speeds = At(forward_speed),
                            wave_directions = At(beta)]
                        @test  a_cpt ≈ a_mh atol=1e-4 rtol = 1e-1
                        # Test radiation damping
                        b_cpt = B_cpt.sel(omega=omega, radiating_dof=radiating_dof, influenced_dof=influenced_dof).values[]
                        b_mh = B_mh[influenced_dofs = At(Symbol(influenced_dof)),
                            radiating_dofs = At(Symbol(radiating_dof)),
                            wave_frequencies = At(omega),
                            forward_speeds = At(forward_speed),
                            wave_directions = At(beta)]
                        @test  b_cpt ≈ b_mh atol=1e-4 rtol = 1e-1
                    end                          
                end
                for beta in betas
                    @testset "Omega: $omega, influenced_dof: $influenced_dof, beta: $beta" begin
                        # Test FK force
                        f_FK_cpt = F_FK_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
                        f_FK_mh = F_FK_mh[influenced_dofs = At(Symbol(influenced_dof)),
                            wave_frequencies = At(omega),
                            wave_directions = At(beta),
                            forward_speeds = At(forward_speed)]
                        @test real(f_FK_cpt) ≈ real(f_FK_mh) atol=1e-4 rtol = 1e-1
                        @test imag(f_FK_cpt) ≈ imag(f_FK_mh) atol=1e-4 rtol = 1e-1
                        # Test diffraction force
                        f_D_cpt = F_D_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
                        f_D_mh = F_D_mh[influenced_dofs = At(Symbol(influenced_dof)),
                            wave_frequencies = At(omega),
                            wave_directions = At(beta),
                            forward_speeds = At(forward_speed)]
                        @test real(f_D_cpt) ≈ real(f_D_mh) atol=1e-4 rtol = 1e-1
                        @test imag(f_D_cpt) ≈ imag(f_D_mh) atol=1e-4 rtol = 1e-1
                        # Test excitation force
                        f_ex_cpt = F_ex_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
                        f_ex_mh = F_ex_mh[influenced_dofs = At(Symbol(influenced_dof)),
                            wave_frequencies = At(omega),
                            wave_directions = At(beta),
                            forward_speeds = At(forward_speed)]
                        @test real(f_ex_cpt) ≈ real(f_ex_mh) atol=1e-4 rtol = 1e-1
                        @test imag(f_ex_cpt) ≈ imag(f_ex_mh) atol=1e-4 rtol = 1e-1
                    end 
                end           
            end        
        end
    end
end