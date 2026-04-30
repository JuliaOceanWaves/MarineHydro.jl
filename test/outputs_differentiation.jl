using Test
using MarineHydro
using PyCall
using LinearAlgebra
using DifferentiationInterface 
import ForwardDiff 
import Zygote


@testset "Integrate Pressure " begin
    cpt = pyimport("capytaine")
    radius = 1.0
    resolution = (4, 4) #smaller data just to test
    cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution)
    cptmesh.keep_immersed_part(inplace=true)
    mesh = Mesh(cptmesh)
    @testset "Re(Forces) and Im(Forces) test" begin
        pressure = rand(size(mesh.areas))   # Random pressure values
        dof = [0.0,0.0,1.0]
        # Differentiability check
        real_grad_zygote, = Zygote.gradient(mesh -> real.(integrate_pressure(mesh, pressure, dof)), mesh) #with respect to mesh
        @test real_grad_zygote !== nothing

        imag_grad_zygote, = Zygote.gradient(mesh -> imag.(integrate_pressure(mesh, pressure, dof)), mesh) #with respect to mesh
        @test imag_grad_zygote !== nothing

        grad_areas = real_grad_zygote.areas #integration depends on normal and area only
        @test grad_areas !== nothing
        grad_normals = real_grad_zygote.normals
        @test typeof(grad_normals) == Matrix{Float64}
        @test grad_normals !== nothing

        # Test the gradient shape (should be the same shape as mesh panels)
        @test size(grad_areas) == size(mesh.areas)
        @test size(grad_normals) == size(mesh.normals)
    end
end

@testset "Gradient accuracy check with Finite diff [w.r.t omega]" begin
    using FiniteDifferences
    cpt = pyimport("capytaine")
    radius = 1.0
    resolution = (4, 4) #smaller data just to test
    cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution).keep_immersed_part()
    mesh = Mesh(cptmesh)
    omega = 1.12
    dof = [0.0,0.0,1.0]
    green_functions = (
        Rankine(),
        RankineReflected(),
        GFWu(),
    )

    function check_added_mass(ω,mesh,dof)
        A = calculate_radiation_forces(mesh,dof,ω)[1]
        return A
    end

    @testset "A(w) partials with Wu " begin
        A(w) = check_added_mass(w, mesh, dof)
        A_w_grad, = Zygote.gradient(A,omega)
        #checking accuracy with finite FiniteDifferences
    # Central difference with order 5
        fd_grad = FiniteDifferences.central_fdm(5, 1)(A, omega)
        @test A_w_grad !== nothing
        @test typeof(A_w_grad) == Float64
        @test A_w_grad ≈  fd_grad atol=1e-6 rtol=1e-6
    end

    @testset "A(w) with w using exact Delhommeau" begin
        green_functions = (
            Rankine(),
            RankineReflected(),
            ExactGuevelDelhommeau(),
        )
        A(w) = check_added_mass(w, mesh, dof)
        A_w_grad1, = Zygote.gradient(A,omega)
        #checking accuracy with FiniteDifferences
    # Central difference with order 5
        fd_grad1 = FiniteDifferences.central_fdm(5, 1)(A, omega)
        @test A_w_grad1 !== nothing
        @test typeof(A_w_grad1) == Float64
        @test A_w_grad1 ≈  fd_grad1 atol=1e-6 rtol=1e-6
    end
end

@testset "Gradient accuracy check with Finite diff [w.r.t omega] for MDOF cylnder" begin
    
    # Description of problem
    omegas = [1.0, 1.5] # frequencies [rad/s]
    beta = 0 # incident wave angle [rad]
    t_DOFs = ["Heave"] # translational DOFs
    r_DOFs = ["Pitch"] # rotational DOFs
    DOFs = [t_DOFs; r_DOFs] # all DOFs

    # Create Mesh object
    radius = 1.5  
    center = (0.0,0.0,0.0) 
    len = 2.5
    faces_max_radius = 0.7
    cpt = pyimport("capytaine")
    cptmesh = cpt.meshes.predefined.mesh_horizontal_cylinder(
                radius=radius,
                center=center, 
                length=len, 
                faces_max_radius = faces_max_radius
                ).keep_immersed_part(inplace=true)

    # Get MarineHydro values
    mesh = Mesh(cptmesh)
    rigid_dof_list = DOFs
    rotation_center = [1.0, 1.0, 0.0] # off set for nonzero off-diagoinal elements
    floatingbody = FloatingBody(mesh, rigid_dof_list, rotation_center, "Horizontal_Cylinder")

    # Radiation solve functions
    function A_and_B_vec(w)
        parameters = (wave_frequencies=w,
            radiating_dofs=collect(keys(floatingbody.dofs)),
            influenced_dofs=collect(keys(floatingbody.dofs)))
        data = compute_hydrodynamic_coefficients(parameters, floatingbody)
        return vcat(vec(data.added_mass), vec(data.radiation_damping)) 
    end

    # Incident + diffraction solve function
    function F_ex_vec(w)
        parameters = (wave_frequencies=[w],
            wave_directions=[beta],
            influenced_dofs=collect(keys(floatingbody.dofs)))
        data = compute_hydrodynamic_coefficients(parameters, floatingbody)
        return vcat(real.(vec(data.excitation_force)),imag.(vec(data.excitation_force))) 
    end

    backend = AutoForwardDiff()



    for omega in omegas
        @testset "Verify sensitivity of added mass and damping wrt Omega: $omega" begin
            A_and_B_FD = FiniteDifferences.central_fdm(5, 1)(A_and_B_vec, omega)
            A_and_B_AD = derivative(A_and_B_vec, backend, omega)
            @test A_and_B_AD !== nothing
            @test typeof(A_and_B_AD) == Vector{Float64}
            @test A_and_B_AD ≈ A_and_B_FD atol=1e-6 rtol=1e-6
        end
        @testset "Verify sensitivity of excitation force wrt Omega: $omega" begin
            F_ex_FD = FiniteDifferences.central_fdm(5, 1)(F_ex_vec, omega)
            F_ex_AD = derivative(F_ex_vec, backend, omega)
            @test F_ex_AD !== nothing
            @test typeof(F_ex_AD) == Vector{Float64}
            @test F_ex_AD ≈ F_ex_FD atol=1e-6 rtol=1e-6
        end
    end  
end
