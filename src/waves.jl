
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


# function DiffractionForce(mesh::Mesh,ω,dof)
#     green_functions = (
#         Rankine(),
#         RankineReflected(),
#         GFWu(),
#     )
#     k = ω^2 / SETTINGS.g
#     S, D = assemble_matrices(green_functions, mesh, k)
#     bc = AiryBC(mesh, ω)
#     potential = solve(D, S, bc)
#     forces = diffraction_force(potential,mesh, ω,dof)
#     return forces
# end


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

function compute_wavenumber(omega::Real)
    return omega^2 / SETTINGS.g
end

function compute_encountered_values(omega::Real, beta::Real, forward_speed::Real)
    k = compute_wavenumber(omega)    
    if forward_speed==0
        return omega, k, beta
    else
        doppler_omega = omega - k * forward_speed * cos(beta)
        encountered_omega = abs(doppler_omega)
        encountered_wavenumber = compute_wavenumber(encountered_omega)
        if doppler_omega >= 0
            encountered_wave_direction = beta
        else
            encountered_wave_direction = beta + pi
        end
        return encountered_omega, encountered_wavenumber, encountered_wave_direction
    end
end

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

# function radiation_bc(mesh::Mesh, dof_mat::Matrix{Float64}, omega::Real)
#     """
#     Calculates the radiation boundary conditions for floating bodies at each panel.

#     # Arguments
#     - `floatingbody::FloatingBody`: The floating body
#     - `omega`: The frequency of the incident ocean wave ~~~.

#     # Returns
#     - The (Neumann) radiation boundary condition values for each panel.
# """
#     bc =  -1im .* omega .* sum(mesh.normals .* dof_mat, dims=2)
#     return bc
# end

function radiation_bc(problem::RadiationProblem)
    """
    Calculates the radiation boundary conditions for floating bodies at each panel.

    # Returns
    - The (Neumann) radiation boundary condition values for each panel.
"""
    dof_mat = problem.floatingbody.dofs[problem.radiating_dof]
    displacement_on_face = sum(problem.floatingbody.mesh.normals .* dof_mat, dims=2)
    bc =  -1im .* problem.encountered_omega .* displacement_on_face
    if problem.forward_speed!=0
        ddofdx = evaluate_gradient_of_motion(problem.floatingbody.mesh, string(problem.radiating_dof))
        ddofdx_dot_n = sum(ddofdx .* problem.floatingbody.mesh.normals, dims=2)
        bc .+= problem.forward_speed .* ddofdx_dot_n
    end
    return bc
end

# Version of compute_bc for a radiation problem
# function compute_bc(problem::RadiationProblem)
#     return radiation_bc(problem.floatingbody.mesh,
#     problem.floatingbody.dofs[problem.radiating_dof],
#     problem.omega)
# end

function compute_bc(problem::RadiationProblem)
    return radiation_bc(problem)
end

################################ Diffraction and Excitation methods #########################################
# function airy_waves_potential(points, omega, beta=0)
#     wavenumber = compute_wavenumber(omega)
#     x, y, z = points[:, 1], points[:, 2], points[:, 3]
#     wbar = x .* cos(beta) .+ y .* sin(beta)
#     cih = exp.(wavenumber .* z)
#     phi = -1im*SETTINGS.g/omega .* cih .* exp.(1im * wavenumber * wbar)
#     return phi
# end

function airy_waves_potential(points, problem::DiffractionProblem)
    x, y, z = points[:, 1], points[:, 2], points[:, 3]

    wavenumber = problem.wavenumber
    enc_beta = problem.encountered_beta
    wbar = x .* cos(enc_beta) .+ y .* sin(enc_beta)    
    
    cih = exp.(wavenumber .* z)
    phi = -1im*SETTINGS.g/problem.omega .* cih .* exp.(1im * wavenumber * wbar)
    return phi
end

# function airy_waves_velocity(points, omega, beta=0, water_depth = Inf)
#     """Compute the fluid velocity for Airy waves at a given point (or array of points)."""
#     k = compute_wavenumber(omega)

#     x, y, z = points[:, 1], points[:, 2], points[:, 3]

#     wbar = x .* cos(beta) .+ y .* sin(beta)
#     cih = exp.(k .* z)
#     sih = exp.(k .* z)

#     v = SETTINGS.g * k / omega .* exp.(1im * k .* wbar) .* 
#         hcat(cos(beta) .* cih, sin(beta) .* cih, -1im .* sih)
#     return v
# end

function airy_waves_velocity(points, problem::DiffractionProblem)
    """Compute the fluid velocity for Airy waves at a given point (or array of points)."""

    x, y, z = points[:, 1], points[:, 2], points[:, 3]
    k = problem.wavenumber
    enc_beta = problem.encountered_beta

    wbar = x .* cos(enc_beta) .+ y .* sin(enc_beta)

    cih = exp.(k .* z)
    sih = exp.(k .* z)

    v = SETTINGS.g * k / problem.omega .* 
        exp.(1im * k .* wbar) .* 
        hcat(cos(problem.beta) .* cih, sin(problem.beta) .* cih, -1im .* sih)
    return v
end


#boundary conditions from airy wave for solving diffraction problem
# Old
# function AiryBC(mesh,omega,beta=0)
#     """Boundary condition for diffraction problem : the velocity on the floating body is the velocity of Airy wave field."""
#     bcs = -sum(airy_waves_velocity(mesh.centers,omega,beta) .* mesh.normals, dims = 2)
#     return bcs
# end

function AiryBC(problem::DiffractionProblem)
    """Boundary condition for diffraction problem : the velocity on the floating body is the velocity of Airy wave field."""
    bcs = -sum(airy_waves_velocity(problem.floatingbody.mesh.centers, problem) .* problem.floatingbody.mesh.normals, dims = 2)
    return bcs
end

# Version of compute_bc for a diffraction problem
# function compute_bc(problem::DiffractionProblem)
#     return AiryBC(problem.floatingbody.mesh, problem.omega, problem.beta)
# end

function compute_bc(problem::DiffractionProblem)
    return AiryBC(problem)
end


# function airy_waves_pressure(points, omega, beta=0)
#     """Compute the pressure for Airy waves."""  

#     return 1im .* omega .* SETTINGS.rho .* airy_waves_potential(points, omega, beta)
# end

function airy_waves_pressure(points, problem::DiffractionProblem)
    """Compute the pressure for Airy waves."""  

    return 1im .* problem.omega .* SETTINGS.rho .* airy_waves_potential(points, problem)
end


# function FroudeKrylovForce(floatingbody::FloatingBody, influenced_dofs::Vector{Symbol}, ω, beta=0)
#     """Compute the Froude-Krylov force."""

#     mesh = floatingbody.mesh
#     pressure =  airy_waves_pressure(mesh.centers,  ω, beta)
#     forces = integrate_pressure(floatingbody, influenced_dofs, pressure) 
#     return forces 
# end

function FroudeKrylovForce(problem::DiffractionProblem, influenced_dofs::Vector{Symbol})
    """Compute the Froude-Krylov force."""

    mesh = problem.floatingbody.mesh
    pressure =  airy_waves_pressure(mesh.centers,  problem)
    forces = integrate_pressure(problem.floatingbody, influenced_dofs, pressure) 
    return forces 
end
