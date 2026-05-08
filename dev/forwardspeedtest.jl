using MarineHydro
using DifferentiationInterface 
import ForwardDiff 
using PyCall
using Test
using DimensionalData
cpt = pyimport("capytaine")

function MH_compute_pressure(problem; direct::Bool=false, gf::String="Wu")
    bc = compute_bc(problem) 

    if problem.forward_speed==0
        omega = problem.omega
        wavenumber = problem.wavenumber
    else
        @assert direct==false "Forward speed problems are only developed with the indirect method"
        omega = problem.encountered_omega
        wavenumber = problem.encountered_wavenumber # use encountered_wavenumber in gfs
    end


    if gf=="Wu"  
        selected_GF = GFWu()
    elseif gf=="ExactGuevelDelhommeau"
        selected_GF = ExactGuevelDelhommeau()
    end

    S, D = assemble_matrices([Rankine(), RankineReflected(), selected_GF], problem.floatingbody.mesh, wavenumber; direct=direct)
    potential, sources = solve(D, S, bc; direct=direct)

    pressure = 1im * SETTINGS.rho * omega * potential # uses encountered_omega

    if problem.forward_speed!=0
        # change normals to all be unit vector in x direction
        S, K = assemble_matrices([Rankine(), RankineReflected(), selected_GF], problem.floatingbody.mesh, wavenumber; direct=direct, all_normals=[1,0,0])
        
        nabla_phi_dot_x = K * sources
        pressure .+= SETTINGS.rho * problem.forward_speed * nabla_phi_dot_x
    end
    forces = integrate_pressure(problem.floatingbody, problem.influenced_dofs, pressure)
    return pressure, forces, sources, K
end



@testset "Forward speed tests" begin

    omega = 0.5
    forward_speed = 2
    beta = 0.5

    t_DOFs = ["Surge","Sway"] # translational DOFs
    r_DOFs = ["Roll","Pitch"] # rotational DOFs
    DOFs = [t_DOFs; r_DOFs] # all DOFs

    # Create Mesh object
    radius = 1.5  
    center = (0.0,0.0,0.0) 
    len = 2.5
    faces_max_radius = 0.9
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

    # Get MarineHydro values
    mesh = Mesh(cptmesh)
    rigid_dof_list = DOFs
    rotation_center = collect(cptbody.rotation_center)
    floatingbody = FloatingBody(mesh, rigid_dof_list, rotation_center, "Horizontal_Cylinder")


    cpt_dif_prob = cpt.DiffractionProblem(body=cptbody,water_depth=Inf,omega=omega,forward_speed=forward_speed,wave_direction=beta)
    cpt_dif_bc = cpt_dif_prob.boundary_condition
    mh_dif_prob = DiffractionProblem(floatingbody, omega, beta, compute_wavenumber(omega), forward_speed, Symbol.(DOFs))
    mh_dif_bc = compute_bc(mh_dif_prob)

    @testset "Encountered properties" begin
        @test cpt_rad_prob.encounter_omega ≈ mh_rad_prob.encountered_omega atol=1e-4 rtol = 1e-4
        @test cpt_rad_prob.encounter_wavenumber ≈ mh_rad_prob.encountered_wavenumber atol=1e-4 rtol = 1e-4
    end

    @testset "Diffraction boundary conditions" begin
        @test cpt_dif_bc ≈ mh_dif_bc atol=1e-4 rtol = 1e-4
    end

    cpt_solver = cpt.BEMSolver()
    cpt_dif_result = cpt_solver.solve(problem=cpt_dif_prob,method="indirect")    
    cpt_dif_pressure = cpt_dif_result.pressure
    # println("cpt forces: $(cpt_dif_result.forces)")
    mh_dif_pressure, mh_dif_forces, mh_dif_sources, mh_K = MH_compute_pressure(mh_dif_prob)
    # println("mh forces: $(mh_dif_forces)")




    @testset "Diffraction sources" begin
        @test cpt_dif_result.sources ≈ mh_dif_sources atol=1e-4 rtol = 1e-4        
    end

    @testset "Diffraction pressure" begin
        @test cpt_dif_pressure ≈ mh_dif_pressure atol=1e-4 rtol = 1e-4
        println("length of cpt pressure: $(length(cpt_dif_pressure))")
        println("length of mh pressure: $(length(mh_dif_pressure))")
    end

    # THE ISSUE IS THE K MATRIX!!!
    @testset "Diffraction K" begin
        problem = cpt_dif_prob
        result = cpt_dif_result
        result = problem.make_results_container(sources=result.sources)
        cpt_solver = cpt.BEMSolver()
        engine = cpt_solver.engine
        println(propertynames(engine))
        gradG = cpt_solver.engine.build_fullK_matrix(problem.body.mesh_including_lid,
        result.body.mesh_including_lid,free_surface=result.free_surface,
        water_depth=result.water_depth,
        wavenumber=result.encounter_wavenumber)
        @test gradG[1] ≈ mh_K atol=1e-4 rtol = 1e-4
    end

    

    for rad_dof in DOFs
        cpt_rad_prob = cpt.RadiationProblem(body=cptbody,water_depth=Inf,omega=omega,forward_speed=forward_speed,radiating_dof=rad_dof,wave_direction=beta)
        cpt_rad_bc = cpt_rad_prob.boundary_condition
        mh_rad_prob = RadiationProblem(floatingbody, omega, beta, compute_wavenumber(omega), forward_speed, Symbol(rad_dof), Symbol.(DOFs))
        mh_rad_bc = compute_bc(mh_rad_prob)
        @testset "Radiation boundary condition for rad_dof=$rad_dof" begin
            @test cpt_rad_bc ≈ mh_rad_bc atol=1e-4 rtol = 1e-4
        end
    end
    


    
    


end







# @testset "Hydrodynamic Coefficient Comparison with Capytaine for MDOF Horizontal Cylinder (atol=1e-4 rtol = 1e-1) " begin
#     # Description of problem
#     forward_speeds = [2.0] # forward speed in x direction [m/s]
#     h = Inf # sea depth [m]
#     omegas = 0.5:0.5:2 # frequencies [rad/s]
#     betas = [pi/6] # incident wave angle [rad]
#     t_DOFs = ["Surge","Sway"] # translational DOFs
#     r_DOFs = ["Roll","Pitch"] # rotational DOFs
#     DOFs = [t_DOFs; r_DOFs] # all DOFs
#     method = "indirect"
#     if method == "direct"
#         direct = true
#     elseif method == "indirect"
#         direct = false
#     end
#     gf = "ExactGuevelDelhommeau"
#     # gf = "Wu"
   

#     # Create Mesh object
#     radius = 1.5  
#     center = (0.0,0.0,0.0) 
#     len = 2.5
#     faces_max_radius = 0.5
#     cptmesh = cpt.meshes.predefined.mesh_horizontal_cylinder(
#                 radius=radius,
#                 center=center, 
#                 length=len, 
#                 faces_max_radius = faces_max_radius
#                 ).keep_immersed_part(inplace=true)

#     # Create FloatingBody object
#     cptbody = cpt.FloatingBody(mesh=cptmesh)
#     cptbody.center_of_mass = (0.0, 0.0, 0.0)
#     cptbody.rotation_center = (1.0, 1.0, 0.0) # off set for nonzero off-diagoinal elements
#     foreach(dof -> cptbody.add_translation_dof(name=dof), t_DOFs)
#     foreach(dof -> cptbody.add_rotation_dof(name=dof), r_DOFs)
#     cptbody.active_dofs = DOFs
#     cptbody.name = "Horizontal Cylinder"

#     # Setup and solve BEM problems
#     solver = cpt.BEMSolver()
#     dof_list = cptbody.active_dofs
#     xr = pyimport("xarray")
#     test_matrix = xr.Dataset(coords=Dict("omega" => omegas,
#         "wave_direction" => betas,
#         "radiating_dof" => DOFs,
#         "forward_speed" => forward_speeds[1]))
#     results = cpt.BEMSolver().fill_dataset(test_matrix, cptbody, method=method)    

#     # Get Capytaine values
#     A_cpt = results.added_mass
#     B_cpt = results.radiation_damping
#     F_FK_cpt = results.Froude_Krylov_force 
#     F_D_cpt = results.diffraction_force
#     F_ex_cpt = results.excitation_force

#     # Get MarineHydro values
#     mesh = Mesh(cptmesh)
#     rigid_dof_list = DOFs
#     rotation_center = collect(cptbody.rotation_center)
#     floatingbody = FloatingBody(mesh, rigid_dof_list, rotation_center, "Horizontal_Cylinder")

#     parameters = (wave_frequencies=omegas, 
#         wave_directions=betas,
#         radiating_dofs=Symbol.(DOFs),
#         influenced_dofs=Symbol.(DOFs),
#         forward_speeds=forward_speeds)

#     mhresults = compute_and_label_hydrodynamic_coefficients(parameters, floatingbody; direct=direct, gf=gf)

#     A_mh = mhresults.added_mass
#     B_mh = mhresults.radiation_damping
#     F_FK_mh = mhresults.Froude_Krylov_force
#     F_D_mh = mhresults.diffraction_force
#     F_ex_mh = mhresults.excitation_force

#     for forward_speed in forward_speeds
#         for omega in omegas
#             for influenced_dof in DOFs
#                 for radiating_dof in DOFs
#                     beta = betas[1]
#                     @testset "Omega: $omega, influenced_dof: $influenced_dof, radiating_dof: $radiating_dof" begin
#                         # Test added mass
#                         a_cpt = A_cpt.sel(omega=omega, radiating_dof=radiating_dof, influenced_dof=influenced_dof).values[]
#                         a_mh = A_mh[influenced_dofs = At(Symbol(influenced_dof)),
#                             radiating_dofs = At(Symbol(radiating_dof)),
#                             wave_frequencies = At(omega),
#                             forward_speeds = At(forward_speed),
#                             wave_directions = At(beta)]
#                         @test  a_cpt ≈ a_mh atol=1e-4 rtol = 1e-1
#                         # Test radiation damping
#                         b_cpt = B_cpt.sel(omega=omega, radiating_dof=radiating_dof, influenced_dof=influenced_dof).values[]
#                         b_mh = B_mh[influenced_dofs = At(Symbol(influenced_dof)),
#                             radiating_dofs = At(Symbol(radiating_dof)),
#                             wave_frequencies = At(omega),
#                             forward_speeds = At(forward_speed),
#                             wave_directions = At(beta)]
#                         @test  b_cpt ≈ b_mh atol=1e-4 rtol = 1e-1
#                     end                          
#                 end
#                 for beta in betas
#                     @testset "Omega: $omega, influenced_dof: $influenced_dof, beta: $beta" begin
#                         # Test FK force
#                         f_FK_cpt = F_FK_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
#                         f_FK_mh = F_FK_mh[influenced_dofs = At(Symbol(influenced_dof)),
#                             wave_frequencies = At(omega),
#                             wave_directions = At(beta),
#                             forward_speeds = At(forward_speed)]
#                         @test real(f_FK_cpt) ≈ real(f_FK_mh) atol=1e-4 rtol = 1e-1
#                         @test imag(f_FK_cpt) ≈ imag(f_FK_mh) atol=1e-4 rtol = 1e-1
#                         # Test diffraction force
#                         f_D_cpt = F_D_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
#                         f_D_mh = F_D_mh[influenced_dofs = At(Symbol(influenced_dof)),
#                             wave_frequencies = At(omega),
#                             wave_directions = At(beta),
#                             forward_speeds = At(forward_speed)]
#                         @test real(f_D_cpt) ≈ real(f_D_mh) atol=1e-4 rtol = 1e-1
#                         @test imag(f_D_cpt) ≈ imag(f_D_mh) atol=1e-4 rtol = 1e-1
#                         # Test excitation force
#                         f_ex_cpt = F_ex_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
#                         f_ex_mh = F_ex_mh[influenced_dofs = At(Symbol(influenced_dof)),
#                             wave_frequencies = At(omega),
#                             wave_directions = At(beta),
#                             forward_speeds = At(forward_speed)]
#                         @test real(f_ex_cpt) ≈ real(f_ex_mh) atol=1e-4 rtol = 1e-1
#                         @test imag(f_ex_cpt) ≈ imag(f_ex_mh) atol=1e-4 rtol = 1e-1
#                     end 
#                 end           
#             end        
#         end
#     end
# end