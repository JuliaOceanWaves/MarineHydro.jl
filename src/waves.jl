# Old functions

"""
    radiation_bc(mesh, dof, omega)

Calculates the radiation boundary conditions for floating bodies at each panel.

# Arguments
- `mesh`: The mesh of the floating body.
- `dof`: The degrees of freedom (assumed same for each panel).
- `omega`: The frequency of the incident ocean wave.

# Returns
- The (Neumann) radiation boundary condition values for each panel.
"""
function radiation_bc(mesh::Mesh, dof, omega)
    nfaces = mesh.nfaces
    d1 = dof[1]
    d2 = dof[2]
    d3 = dof[3]
    scale = -1im * omega
    return [scale * (mesh.normals[i, 1] * d1 + mesh.normals[i, 2] * d2 + mesh.normals[i, 3] * d3) for i in 1:nfaces]
end

function radiation_bc(mesh::StaticArraysMesh, dof, omega)
    d1, d2, d3 = dof[1], dof[2], dof[3]
    scale = -1im * omega
    return [scale * (n[1] * d1 + n[2] * d2 + n[3] * d3) for n in mesh.normals]
end

function integrate_pressure(mesh::Mesh, pressure, dof)
    normal_dof_amp = -sum(transpose(dof) .* mesh.normals, dims=2)
    forces = sum(pressure .* normal_dof_amp .* mesh.areas)
    return forces
end

function integrate_pressure(mesh::StaticArraysMesh, pressure, dof)
    d1, d2, d3 = dof[1], dof[2], dof[3]
    s = zero(promote_type(eltype(pressure), eltype(mesh.areas)))
    @inbounds for i in 1:mesh.nfaces
        n = mesh.normals[i]
        s += pressure[i] * -(d1 * n[1] + d2 * n[2] + d3 * n[3]) * mesh.areas[i]
    end
    return s
end

function calculate_radiation_forces(mesh, dof, omega)
    k = omega^2 / SETTINGS.g
    S, D = assemble_matrix_wu(mesh, k)
    bc = radiation_bc(mesh, dof, omega)
    potential, sources = solve(D, S, bc)
    pressure = 1im * SETTINGS.rho * omega * potential
    forces = integrate_pressure(mesh, pressure, dof)
    return [real(forces)/omega^2, imag(forces)/omega]
end


function diffraction_force(mesh::Mesh, ω, dof)
    green_functions = (
        Rankine(),
        RankineReflected(),
        GFWu(),
    )
    k = ω^2 / SETTINGS.g
    S, D = assemble_matrices(green_functions, mesh, k)
    bc = airy_bc(mesh, ω)
    potential, sources = solve(D, S, bc)
    forces = diffraction_force(potential, mesh, ω, dof)
    return forces
end

function airy_waves_potential(points, omega, beta=0)
    wavenumber = compute_wavenumber(omega)
    x, y, z = points[:, 1], points[:, 2], points[:, 3]
    wbar = x .* cos(beta) .+ y .* sin(beta)
    cih = exp.(wavenumber .* z)
    phi = -1im*SETTINGS.g/omega .* cih .* exp.(1im * wavenumber * wbar)
    return phi
end

"""Compute the fluid velocity for Airy waves at a given point (or array of points)."""
function airy_waves_velocity(points, omega, beta=0, water_depth=Inf)
    k = compute_wavenumber(omega)

    x, y, z = points[:, 1], points[:, 2], points[:, 3]

    wbar = x .* cos(beta) .+ y .* sin(beta)
    cih = exp.(k .* z)
    sih = exp.(k .* z)

    v = SETTINGS.g * k / omega .* exp.(1im * k .* wbar) .* 
        hcat(cos(beta) .* cih, sin(beta) .* cih, -1im .* sih)
    return v
end

"""
Boundary condition for the diffraction problem: the velocity on the floating body
is the velocity of the Airy wave field.
"""
function airy_bc(mesh, omega, beta=0)
    velocity = airy_waves_velocity(mesh.centers, omega, beta)
    nfaces = Int(mesh.nfaces)
    return [-(velocity[i, 1] * mesh.normals[i, 1] + velocity[i, 2] * mesh.normals[i, 2] + velocity[i, 3] * mesh.normals[i, 3]) for i in 1:nfaces]
end

"""Compute the pressure for Airy waves."""
function airy_waves_pressure(points, omega, beta=0)
    return 1im .* omega .* SETTINGS.rho .* airy_waves_potential(points, omega, beta)
end


function diffraction_force(potential, mesh, omega, dof)
    pressure = 1im * SETTINGS.rho * potential * omega
    forces = integrate_pressure(mesh, pressure, dof)
    return forces
end

"""Compute the Froude-Krylov force."""
function froude_krylov_force(mesh::Mesh, ω, dof)
    pressure = airy_waves_pressure(mesh.centers, ω)
    return integrate_pressure(mesh, pressure, dof)
end



# New functions

"""
    compute_wavenumber(omega, g = SETTINGS.g)

Deep-water dispersion relation `k = omega^2 / g`.
"""
compute_wavenumber(omega) = compute_wavenumber(omega, SETTINGS.g)
compute_wavenumber(omega, g) = omega^2 / g

function compute_encountered_values(omega, beta, forward_speed, g=SETTINGS.g)
    k = compute_wavenumber(omega, g)
    if forward_speed == 0
        return omega, k, beta
    else
        doppler_omega = omega - k * forward_speed * cos(beta)
        encountered_omega = abs(doppler_omega)
        encountered_wavenumber = compute_wavenumber(encountered_omega, g)
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
        _integrate_dof_pressure(mesh, floatingbody.dofs[dof_symbol], pressure)
        for dof_symbol in influenced_dofs
    ]
    forces = NamedTuple{Tuple(influenced_dofs)}(Tuple(force_values))
    return forces
end

function _integrate_dof_pressure(mesh::Mesh, dof_mat, pressure)
    normal_dof_amp_on_face = -sum(dof_mat .* mesh.normals, dims=2)
    return sum(pressure .* normal_dof_amp_on_face .* mesh.areas)
end

function _integrate_dof_pressure(mesh::StaticArraysMesh, dof_mat, pressure)
    s = zero(promote_type(eltype(pressure), eltype(mesh.areas), eltype(dof_mat)))
    @inbounds for i in 1:mesh.nfaces
        n = mesh.normals[i]
        disp = dof_mat[i, 1] * n[1] + dof_mat[i, 2] * n[2] + dof_mat[i, 3] * n[3]
        s += pressure[i] * -disp * mesh.areas[i]
    end
    return s
end

################################ Radiation methods #########################################


function radiation_bc(problem::RadiationProblem)
    """
    Calculates the radiation boundary conditions for floating bodies at each panel.

    # Returns
    - The (Neumann) radiation boundary condition values for each panel.
"""
    mesh = problem.floatingbody.mesh
    dof_mat = problem.floatingbody.dofs[problem.radiating_dof]
    displacement_on_face = _normal_displacement(mesh, dof_mat)
    bc =  -1im .* problem.encountered_omega .* displacement_on_face
    if problem.forward_speed!=0
        ddofdx = evaluate_gradient_of_motion(mesh, string(problem.radiating_dof))
        bc = bc .+ problem.forward_speed .* _normal_displacement(mesh, ddofdx)
    end
    return bc
end

_normal_displacement(mesh::Mesh, dof_mat) = sum(mesh.normals .* dof_mat, dims=2)

function _normal_displacement(mesh::StaticArraysMesh, dof_mat)
    T = promote_type(eltype(mesh.areas), eltype(dof_mat))
    out = Vector{T}(undef, mesh.nfaces)
    @inbounds for i in 1:mesh.nfaces
        n = mesh.normals[i]
        out[i] = n[1] * dof_mat[i, 1] + n[2] * dof_mat[i, 2] + n[3] * dof_mat[i, 3]
    end
    return out
end

function compute_bc(problem::RadiationProblem)
    return radiation_bc(problem)
end

################################ Diffraction and Excitation methods #########################################


function airy_waves_potential(points, problem::DiffractionProblem)
    x, y, z = points[:, 1], points[:, 2], points[:, 3]

    wavenumber = problem.wavenumber
    enc_beta = problem.encountered_beta
    wbar = x .* cos(enc_beta) .+ y .* sin(enc_beta)

    cih = exp.(wavenumber .* z)
    phi = -1im * problem.g / problem.omega .* cih .* exp.(1im * wavenumber * wbar)
    return phi
end

"""Compute the fluid velocity for Airy waves at a given point (or array of points)."""
function airy_waves_velocity(points, problem::DiffractionProblem)
    x, y, z = points[:, 1], points[:, 2], points[:, 3]
    k = problem.wavenumber
    enc_beta = problem.encountered_beta

    wbar = x .* cos(enc_beta) .+ y .* sin(enc_beta)

    cih = exp.(k .* z)
    sih = exp.(k .* z)

    v = problem.g * k / problem.omega .*
        exp.(1im * k .* wbar) .*
        hcat(cos(problem.beta) .* cih, sin(problem.beta) .* cih, -1im .* sih)
    return v
end


# Boundary conditions from the Airy wave for solving the diffraction problem.
function airy_bc(problem::DiffractionProblem)
    return _airy_bc(problem.floatingbody.mesh, problem)
end

function _airy_bc(mesh::Mesh, problem::DiffractionProblem)
    return -sum(airy_waves_velocity(mesh.centers, problem) .* mesh.normals, dims=2)
end

function _airy_phi(p::SVector{3}, problem::DiffractionProblem)
    k = problem.wavenumber
    enc_beta = problem.encountered_beta
    wbar = p[1] * cos(enc_beta) + p[2] * sin(enc_beta)
    return -1im * problem.g / problem.omega * exp(k * p[3]) * exp(1im * k * wbar)
end

function _airy_vel(p::SVector{3}, problem::DiffractionProblem)
    k = problem.wavenumber
    enc_beta = problem.encountered_beta
    wbar = p[1] * cos(enc_beta) + p[2] * sin(enc_beta)
    cih = exp(k * p[3])
    amp = problem.g * k / problem.omega * exp(1im * k * wbar)
    return SVector(amp * cos(problem.beta) * cih, amp * sin(problem.beta) * cih, -1im * amp * cih)
end

function _airy_bc(mesh::StaticArraysMesh, problem::DiffractionProblem)
    # Do not use `dot` — it conjugates Complex velocities and flips the force.
    return [begin
        v = _airy_vel(mesh.centers[i], problem)
        n = mesh.normals[i]
        -(v[1] * n[1] + v[2] * n[2] + v[3] * n[3])
    end for i in eachindex(mesh.centers)]
end

function compute_bc(problem::DiffractionProblem)
    return airy_bc(problem)
end


"""Compute the pressure for Airy waves."""
function airy_waves_pressure(points, problem::DiffractionProblem)
    return 1im .* problem.omega .* problem.rho .* airy_waves_potential(points, problem)
end

function froude_krylov_force(problem::DiffractionProblem, influenced_dofs::Vector{Symbol})
    mesh = problem.floatingbody.mesh
    pressure = _airy_pressure(mesh, problem)
    return integrate_pressure(problem.floatingbody, influenced_dofs, pressure)
end

_airy_pressure(mesh::Mesh, problem::DiffractionProblem) = airy_waves_pressure(mesh.centers, problem)
function _airy_pressure(mesh::StaticArraysMesh, problem::DiffractionProblem)
    return [1im * problem.omega * problem.rho * _airy_phi(p, problem) for p in mesh.centers]
end

# Deprecated CamelCase-era names; prefer the snake_case functions.
const AiryBC = airy_bc
const FroudeKrylovForce = froude_krylov_force
const DiffractionForce = diffraction_force
