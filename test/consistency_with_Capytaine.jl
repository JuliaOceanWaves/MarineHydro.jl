using MarineHydro
using Test
using PyCall
using DimensionalData

cpt = pyimport("capytaine")

cpt_mesh_sphere = MarineHydro.example_mesh_from_capytaine()
cpt_mesh_two_spheres = (cpt_mesh_sphere + cpt_mesh_sphere.translated_x(5.0)).copy(name="floating two spheres")

@testset "Comparison with Capytaine for $(cptmesh.name)" for cptmesh in [cpt_mesh_sphere, cpt_mesh_two_spheres]

    mesh = Mesh(cptmesh)

    # As long as we use Capytaine for the mesh, we might as well use it also for validation.
    # When we get rid of the dependency to Capytaine, this test might need to be restructured.

    @testset "Pure Rankine, direct=$direct" for direct in [true, false]
        green_functions = (Rankine(),)
        S, D = assemble_matrices(green_functions, mesh, 1.0, direct=direct)
        capy_S, capy_D = cpt.Delhommeau().evaluate(cptmesh, cptmesh, free_surface=Inf, water_depth=Inf, wavenumber=1.0, adjoint_double_layer=!direct)
        @test S ≈ capy_S  atol=1e-6
        @test D ≈ capy_D  atol=1e-6
    end

    @testset "Rankine + reflected, direct=$direct" for direct in [true, false]
        green_functions = (Rankine(), RankineReflected())
        S, D = assemble_matrices(green_functions, mesh, 1.0, direct=direct)
        capy_S, capy_D = cpt.Delhommeau().evaluate(cptmesh, cptmesh, free_surface=0.0, water_depth=Inf, wavenumber=0.0, adjoint_double_layer=!direct)
        @test S ≈ capy_S  atol=1e-6
        @test D ≈ capy_D  atol=1e-6
    end

    @testset "Rankine - reflected, direct=$direct" for direct in [true, false]
        green_functions = (Rankine(), RankineReflectedNegative())
        S, D = assemble_matrices(green_functions, mesh, 1.0, direct=direct)
        capy_S, capy_D = cpt.Delhommeau().evaluate(cptmesh, cptmesh, free_surface=0.0, water_depth=Inf, wavenumber=Inf, adjoint_double_layer=!direct)
        @test S ≈ capy_S  atol=1e-6
        @test D ≈ capy_D  atol=1e-6
    end

    @testset "Full Green function with Guével-Delhommeau, direct=$direct, k=$k" for direct in [true, false], k in [1.0, 2.0]
        green_functions = (Rankine(), RankineReflected(), ExactGuevelDelhommeau(),)
        S, D = assemble_matrices(green_functions, mesh, k, direct=direct)
        capy_S, capy_D = cpt.Delhommeau(tabulation_nr=0).evaluate(cptmesh, cptmesh, free_surface=0.0, water_depth=Inf, wavenumber=k, adjoint_double_layer=!direct)
        @test real.(S) ≈ real.(capy_S)  atol=1e-3
        @test imag.(S) ≈ imag.(capy_S)  atol=1e-6
        @test real.(D) ≈ real.(capy_D)  atol=1e-4
        @test imag.(D) ≈ imag.(capy_D)  atol=1e-6
    end

    @testset "Full Green function with Wu, direct=$direct, k=$k" for direct in [true, false], k in [1.0, 2.0]
        green_functions = (Rankine(), RankineReflected(), GFWu())
        S, D = assemble_matrices(green_functions, mesh, k, direct=direct)
        capy_S, capy_D = cpt.Delhommeau(tabulation_nr=0).evaluate(cptmesh, cptmesh, free_surface=0.0, water_depth=Inf, wavenumber=k, adjoint_double_layer=!direct)
        @test real.(S) ≈ real.(capy_S)  atol=1e-2
        @test imag.(S) ≈ imag.(capy_S)  atol=1e-6
        @test real.(D) ≈ real.(capy_D)  atol=1e-2
        @test imag.(D) ≈ imag.(capy_D)  atol=1e-6
    end

    @testset "Diffraction and Excitation methods, boundary condition, airywaves test" begin 
        # cpt = pyimport("capytaine")
        # r = 1.0
        ω = 1.03
        # Create the mesh and body
        # cptmesh = cpt.mesh_sphere(radius=r, center=(0, 0, 0), resolution=(14, 14)).immersed_part()
        cptbody = cpt.FloatingBody(cptmesh, name="sphere")
        cptbody.add_translation_dof(name="Heave")
        pb = cpt.DiffractionProblem(body = cptbody,omega = ω,wave_direction= 0 )
        diffProbBC = pb.boundary_condition
        # mesh  = Mesh(cptmesh)
        juliaBC = AiryBC(mesh,ω)
        @test diffProbBC ≈ juliaBC atol=1e-3 rtol = 1e-3

        capyairy = cpt.bem.airy_waves.airy_waves_pressure(cptmesh.faces_centers, pb)
        juliaairy = airy_waves_pressure(mesh.centers, ω)
        @test abs.(capyairy) ≈ abs.(juliaairy) atol=1e-3 rtol = 1e-3

        capyairyV =cpt.bem.airy_waves.airy_waves_velocity(cptmesh.faces_centers,pb)
        juliaairyV = airy_waves_velocity(mesh.centers,ω)
        @test capyairyV ≈ juliaairyV atol=1e-4 rtol = 1e-4

        capyairyPot =cpt.bem.airy_waves.airy_waves_potential(cptmesh.faces_centers,pb)
        juliaairyPot = airy_waves_potential(mesh.centers, ω)
        @test capyairyPot ≈ juliaairyPot atol=1e-4 rtol = 1e-4
    end
end


@testset "Comparison of Excitation Forces with Capytaine (rtol = 1e-1) " begin
    ### diffraction forces analytical by https://www.sciencedirect.com/science/article/pii/S002980182202813X
    #buut that did not match 
    #from capytaine 
    cpt = pyimport("capytaine")
    r = 1.0
    # Create the mesh and body
    cptmesh = cpt.mesh_sphere(radius=r, center=(0, 0, 0), resolution=(14, 14)).immersed_part()
    cptbody = cpt.FloatingBody(cptmesh, name="sphere")
    cptbody.add_translation_dof(name="Heave")

    # Define the wave numbers and corresponding frequencies
    K_heave_diff = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    omegas = sqrt.(K_heave_diff .* 9.81)
    non_dimensional_const = 2/3 * pi * r^3 * 1000

    # Create the test matrix
    xr = pyimport("xarray")
    test_matrix = xr.Dataset(coords=Dict("omega" => omegas, "wave_direction" => [0.0]))
    results = cpt.BEMSolver().fill_dataset(test_matrix, cptbody, method="direct")
    Froude_heave =  vec(results.Froude_Krylov_force.values)./ non_dimensional_const
    Diff_heave =   vec(results.diffraction_force.values) ./ non_dimensional_const
    mesh = Mesh(cptmesh)
    dof = [0,0,1]
    julia_diff_omega = [DiffractionForce(mesh,w,dof) for w in omegas] ./ non_dimensional_const
    julia_fr_force = [FroudeKrylovForce(mesh,w,dof) for w in omegas] ./ non_dimensional_const
    @test real.(julia_diff_omega) ≈ real.(Diff_heave) rtol=1e-1
    @test imag.(julia_diff_omega) ≈ imag.(Diff_heave) rtol=1e-1
    @test real.(julia_fr_force) ≈ real.(Froude_heave) rtol=1e-1
    @test imag.(julia_fr_force) ≈ imag.(Froude_heave) atol=1e-1

end 




@testset "Hydrodynamic Coefficient Comparison with Capytaine for MDOF Horizontal Cylinder (atol=1e-4 rtol = 1e-1) " begin
    # Description of problem
    h = Inf # sea depth [m]
    omegas = 0.5:0.5:2 # frequencies [rad/s]
    betas = 0.0:pi/4:pi/2 # incident wave angle [rad]
    t_DOFs = ["Surge","Sway","Heave"] # translational DOFs
    r_DOFs = ["Roll","Pitch","Yaw"] # rotational DOFs
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
    test_matrix = xr.Dataset(coords=Dict("omega" => omegas, "wave_direction" => betas, "radiating_dof" => DOFs))
    results = cpt.BEMSolver().fill_dataset(test_matrix, cptbody, method="direct")    

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
        influenced_dofs=Symbol.(DOFs))

    mhresults = compute_and_label_hydrodynamic_coefficients(parameters, floatingbody)

    A_mh = mhresults.added_mass
    B_mh = mhresults.radiation_damping
    F_FK_mh = mhresults.Froude_Krylov_force
    F_D_mh = mhresults.diffraction_force
    F_ex_mh = mhresults.excitation_force

    for omega in omegas
        for influenced_dof in DOFs
            for radiating_dof in DOFs
                @testset "Omega: $omega, influenced_dof: $influenced_dof, radiating_dof: $radiating_dof" begin
                    # Test added mass
                    a_cpt = A_cpt.sel(omega=omega, radiating_dof=radiating_dof, influenced_dof=influenced_dof).values[]
                    a_mh = A_mh[influenced_dofs = At(Symbol(influenced_dof)),radiating_dofs = At(Symbol(radiating_dof)), wave_frequencies = At(omega)]
                    @test  a_cpt ≈ a_mh atol=1e-4 rtol = 1e-1
                    # Test radiation damping
                    b_cpt = B_cpt.sel(omega=omega, radiating_dof=radiating_dof, influenced_dof=influenced_dof).values[]
                    b_mh = B_mh[influenced_dofs = At(Symbol(influenced_dof)),radiating_dofs = At(Symbol(radiating_dof)), wave_frequencies = At(omega)]
                    @test  b_cpt ≈ b_mh atol=1e-4 rtol = 1e-1
                end                          
            end
            for beta in betas
                @testset "Omega: $omega, influenced_dof: $influenced_dof, beta: $beta" begin
                    # Test FK force
                    f_FK_cpt = F_FK_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
                    f_FK_mh = F_FK_mh[influenced_dofs = At(Symbol(influenced_dof)), wave_frequencies = At(omega), wave_directions = At(beta)]
                    @test real(f_FK_cpt) ≈ real(f_FK_mh) atol=1e-4 rtol = 1e-1
                    @test imag(f_FK_cpt) ≈ imag(f_FK_mh) atol=1e-4 rtol = 1e-1
                    # Test diffraction force
                    f_D_cpt = F_D_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
                    f_D_mh = F_D_mh[influenced_dofs = At(Symbol(influenced_dof)), wave_frequencies = At(omega), wave_directions = At(beta)]
                    @test real(f_D_cpt) ≈ real(f_D_mh) atol=1e-4 rtol = 1e-1
                    @test imag(f_D_cpt) ≈ imag(f_D_mh) atol=1e-4 rtol = 1e-1
                    # Test excitation force
                    f_ex_cpt = F_ex_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
                    f_ex_mh = F_ex_mh[influenced_dofs = At(Symbol(influenced_dof)), wave_frequencies = At(omega), wave_directions = At(beta)]
                    @test real(f_ex_cpt) ≈ real(f_ex_mh) atol=1e-4 rtol = 1e-1
                    @test imag(f_ex_cpt) ≈ imag(f_ex_mh) atol=1e-4 rtol = 1e-1
                end 
            end           
        end        
    end
end

@testset "Hydrodynamic Coefficient Comparison with Capytaine for Array of MDOF Horizontal Cylinders  (atol=1e-4 rtol = 1e-1) " begin
    # Description of problem
    h = Inf # sea depth [m]
    omegas = 0.5:0.5:2 # frequencies [rad/s]
    betas = 0.0:pi/4:pi/2 # incident wave angle [rad]
    t_DOFs = ["Heave"] # translational DOFs
    r_DOFs = ["Roll"] # rotational DOFs
    DOFs = [t_DOFs; r_DOFs] # all DOFs
    sep_dis = 50.0 # separation distance

    # Create Mesh objects
    # Mesh 1
    radius = 1.5  
    center1 = (0.0,0.0,0.0) 
    len = 2.5
    faces_max_radius = 0.5
    cptmesh1 = cpt.meshes.predefined.mesh_horizontal_cylinder(
                radius=radius,
                center=center1, 
                length=len, 
                faces_max_radius = faces_max_radius
                ).keep_immersed_part(inplace=true)
    # Mesh 2
    center2 = (sep_dis,0.0,0.0) 
    cptmesh2 = cpt.meshes.predefined.mesh_horizontal_cylinder(
                radius=radius,
                center=center2, 
                length=len, 
                faces_max_radius = faces_max_radius
                ).keep_immersed_part(inplace=true)

    # Create FloatingBody object

    # FloatingBody 1
    cptbody1 = cpt.FloatingBody(mesh=cptmesh1)
    cptbody1.center_of_mass = center1
    cptbody1.rotation_center = (1.0, 1.0, 0.0) # off set for nonzero off-diagoinal elements
    foreach(dof -> cptbody1.add_translation_dof(name=dof), t_DOFs)
    foreach(dof -> cptbody1.add_rotation_dof(name=dof), r_DOFs)
    cptbody1.active_dofs = DOFs
    cptbody1.name = "Horizontal_Cylinder_1"

    # FloatingBody 2
    cptbody2 = cpt.FloatingBody(mesh=cptmesh2)
    cptbody2.center_of_mass = center2
    cptbody2.rotation_center = (sep_dis+1.0, 1.0, 0.0) # off set for nonzero off-diagoinal elements
    foreach(dof -> cptbody2.add_translation_dof(name=dof), t_DOFs)
    foreach(dof -> cptbody2.add_rotation_dof(name=dof), r_DOFs)
    cptbody2.active_dofs = DOFs
    cptbody2.name = "Horizontal_Cylinder_2"

    cptbody = cptbody1 + cptbody2

    # Setup and solve BEM problems
    solver = cpt.BEMSolver()
    dof_list = cptbody.dofs
    all_dofs = string.(collect(keys(dof_list)))
    xr = pyimport("xarray")
    test_matrix = xr.Dataset(coords=Dict("omega" => omegas, "wave_direction" => betas, "radiating_dof" => all_dofs))
    results = cpt.BEMSolver().fill_dataset(test_matrix, cptbody, method="direct")    

    # Get Capytaine values
    A_cpt = results.added_mass
    B_cpt = results.radiation_damping
    F_FK_cpt = results.Froude_Krylov_force 
    F_D_cpt = results.diffraction_force
    F_ex_cpt = results.excitation_force

    # Get MarineHydro values
    # FloatingBody 1
    mesh1 = Mesh(cptmesh1)
    rigid_dof_list = DOFs
    rotation_center1 = collect(cptbody1.rotation_center)
    floatingbody1 = FloatingBody(mesh1, rigid_dof_list, rotation_center1, "Horizontal_Cylinder_1")

    # FloatingBody 2
    mesh2 = Mesh(cptmesh2)
    rotation_center2 = collect(cptbody2.rotation_center)
    floatingbody2 = FloatingBody(mesh2, rigid_dof_list, rotation_center2, "Horizontal_Cylinder_2")

    floatingbody = combine_floatingbodies([floatingbody1, floatingbody2])


    parameters = (wave_frequencies=omegas, 
        wave_directions=betas,
        radiating_dofs=collect(keys(floatingbody.dofs)),
        influenced_dofs=collect(keys(floatingbody.dofs)))

    mhresults = compute_and_label_hydrodynamic_coefficients(parameters, floatingbody)

    A_mh = mhresults.added_mass
    B_mh = mhresults.radiation_damping
    F_FK_mh = mhresults.Froude_Krylov_force
    F_D_mh = mhresults.diffraction_force
    F_ex_mh = mhresults.excitation_force

    for omega in omegas
        for influenced_dof in all_dofs
            for radiating_dof in all_dofs
                @testset "Omega: $omega, influenced_dof: $influenced_dof, radiating_dof: $radiating_dof" begin
                    # Test added mass
                    a_cpt = A_cpt.sel(omega=omega, radiating_dof=radiating_dof, influenced_dof=influenced_dof).values[]
                    a_mh = A_mh[influenced_dofs = At(Symbol(influenced_dof)),radiating_dofs = At(Symbol(radiating_dof)), wave_frequencies = At(omega)]
                    @test  a_cpt ≈ a_mh atol=1e-4 rtol = 1e-1
                    # Test radiation damping
                    b_cpt = B_cpt.sel(omega=omega, radiating_dof=radiating_dof, influenced_dof=influenced_dof).values[]
                    b_mh = B_mh[influenced_dofs = At(Symbol(influenced_dof)),radiating_dofs = At(Symbol(radiating_dof)), wave_frequencies = At(omega)]
                    @test  b_cpt ≈ b_mh atol=1e-4 rtol = 1e-1
                end                          
            end
            for beta in betas
                @testset "Omega: $omega, influenced_dof: $influenced_dof, beta: $beta" begin
                    # Test FK force
                    f_FK_cpt = F_FK_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
                    f_FK_mh = F_FK_mh[influenced_dofs = At(Symbol(influenced_dof)), wave_frequencies = At(omega), wave_directions = At(beta)]
                    @test real(f_FK_cpt) ≈ real(f_FK_mh) atol=1e-4 rtol = 1e-1
                    @test imag(f_FK_cpt) ≈ imag(f_FK_mh) atol=1e-4 rtol = 1e-1
                    # Test diffraction force
                    f_D_cpt = F_D_cpt.sel(omega=omega, influenced_dof=influenced_dof, wave_direction=beta).values[]
                    f_D_mh = F_D_mh[influenced_dofs = At(Symbol(influenced_dof)), wave_frequencies = At(omega), wave_directions = At(beta)]
                    @test real(f_D_cpt) ≈ real(f_D_mh) atol=1e-4 rtol = 1e-1
                    @test imag(f_D_cpt) ≈ imag(f_D_mh) atol=1e-4 rtol = 1e-1
                end 
            end           
        end        
    end
end