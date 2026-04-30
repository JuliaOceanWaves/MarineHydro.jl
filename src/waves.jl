
# Old functions

function radiation_bc(mesh::Mesh, dof, omega)
    """
        radiation_bc(mesh::Mesh, dof, omega)

    Calculates the radiation boundary conditions for floating bodies at each panel.

    # Arguments
    - `mesh::Mesh`: The mesh of the floating body.
    - `dof`: The degrees of freedom (assumed same for each panel).
    - `omega`: The frequency of the incident ocean wave ~~~.

    # Returns
    - The (Neumann) radiation boundary condition values for each panel.
"""
    return -1im .* omega .* sum(mesh.normals .* dof', dims=2)
end

function integrate_pressure(mesh::Mesh, pressure, dof)
    normal_dof_amp = -sum(transpose(dof) .* mesh.normals, dims=2)
    forces = sum(pressure .* normal_dof_amp .* mesh.areas)
  return forces
  end


  
function calculate_radiation_forces(mesh::Mesh, dof, omega)
    k = omega^2 / SETTINGS.g
    S, D = assemble_matrix_wu(mesh, k)
    bc = radiation_bc(mesh, dof, omega)
    potential = solve(D, S, bc)
    pressure = 1im * SETTINGS.rho * omega * potential
    forces = integrate_pressure(mesh, pressure, dof)
    return [real(forces)/omega^2, imag(forces)/omega]
end


function DiffractionForce(mesh::Mesh,ω,dof)
    green_functions = (
        Rankine(),
        RankineReflected(),
        GFWu(),
    )
    k = ω^2 / SETTINGS.g
    S, D = assemble_matrices(green_functions, mesh, k)
    bc = AiryBC(mesh, ω)
    potential = solve(D, S, bc)
    forces = diffraction_force(potential,mesh, ω,dof)
    return forces
end


function diffraction_force(potential,mesh, omega,dof)
    pressure = 1im*SETTINGS.rho* potential * omega 
    forces = integrate_pressure(mesh,pressure,dof) 
    return forces  
  end


function FroudeKrylovForce(mesh::Mesh, ω,dof)
    """Compute the Froude-Krylov force."""
    pressure =  airy_waves_pressure(mesh.centers,  ω)
    return  integrate_pressure(mesh::Mesh, pressure, dof) 
end



# New function

function integrate_pressure(floatingbody::FloatingBody, influenced_dofs::Vector{Symbol}, pressure)
    mesh = floatingbody.mesh

    force_values = [
        let
            dof_mat = floatingbody.dofs[dof_symbol]
            normal_dof_amp_on_face = -sum(dof_mat .* mesh.normals, dims=2)
            sum(pressure .* normal_dof_amp_on_face .* mesh.areas)
        end 
        for dof_symbol in influenced_dofs
    ]
    forces = NamedTuple{Tuple(influenced_dofs)}(Tuple(force_values))
    return forces
end

################################ Radiation methods #########################################

function radiation_bc(mesh::Mesh, dof_mat::Matrix{Float64}, omega::Real)
    """
    Calculates the radiation boundary conditions for floating bodies at each panel.

    # Arguments
    - `floatingbody::FloatingBody`: The floating body
    - `omega`: The frequency of the incident ocean wave ~~~.

    # Returns
    - The (Neumann) radiation boundary condition values for each panel.
"""
    bc =  -1im .* omega .* sum(mesh.normals .* dof_mat, dims=2)
    return bc
end

# Version of compute_bc for a radiation problem
function compute_bc(problem::RadiationProblem)
    return radiation_bc(problem.floatingbody.mesh,
    problem.floatingbody.dofs[problem.radiating_dof],
    problem.omega)
end

################################ Diffraction and Excitation methods #########################################
function airy_waves_potential(points, omega, beta=0)
    wavenumber = omega^2/SETTINGS.g
    x, y, z = points[:, 1], points[:, 2], points[:, 3]
    wbar = x .* cos(beta) .+ y .* sin(beta)
    cih = exp.(wavenumber .* z)
    phi = -1im*SETTINGS.g/omega .* cih .* exp.(1im * wavenumber * wbar)
    return phi
end

function airy_waves_velocity(points, omega, beta=0, water_depth = Inf)
    """Compute the fluid velocity for Airy waves at a given point (or array of points)."""
    k = omega^2/SETTINGS.g

    x, y, z = points[:, 1], points[:, 2], points[:, 3]

    wbar = x .* cos(beta) .+ y .* sin(beta)
    cih = exp.(k .* z)
    sih = exp.(k .* z)

    v = SETTINGS.g * k / omega .* exp.(1im * k .* wbar) .* 
        hcat(cos(beta) .* cih, sin(beta) .* cih, -1im .* sih)
    return v
end


#boundary conditions from airy wave for solving diffraction problem
function AiryBC(mesh,omega,beta=0)
    """Boundary condition for diffraction problem : the velocity on the floating body is the velocity of Airy wave field."""
    bcs = -sum(airy_waves_velocity(mesh.centers,omega,beta) .* mesh.normals, dims = 2)
    return bcs
end

# Version of compute_bc for a diffraction problem
function compute_bc(problem::DiffractionProblem)
    return AiryBC(problem.floatingbody.mesh, problem.omega, problem.beta)
end


function airy_waves_pressure(points, omega, beta=0)
    """Compute the pressure for Airy waves."""  

    return 1im .* omega .* SETTINGS.rho .* airy_waves_potential(points, omega, beta)
end


function FroudeKrylovForce(floatingbody::FloatingBody, influenced_dofs::Vector{Symbol}, ω, beta=0)
    """Compute the Froude-Krylov force."""

    mesh = floatingbody.mesh
    pressure =  airy_waves_pressure(mesh.centers,  ω, beta)
    forces = integrate_pressure(floatingbody, influenced_dofs, pressure) 
    return forces 
end
