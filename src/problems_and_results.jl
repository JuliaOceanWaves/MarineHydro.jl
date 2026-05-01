using DimensionalData

########################## Problems #########################################

# Abstract type named LinearPotentialFlowProblem. The diffraction and radiation problems will be subtypes of this type. 
abstract type LinearPotentialFlowProblem end

# Define DiffractionProblem struct as a subtype of LinearPotentialFlowProblem
struct DiffractionProblem <: LinearPotentialFlowProblem
    floatingbody::FloatingBody
    omega::Real
    beta::Real
    forward_speed::Real
    influenced_dofs::Vector{Symbol}
    function DiffractionProblem(floatingbody::FloatingBody,
        omega::Real,
        beta::Real,
        forward_speed::Real,
        influenced_dofs::Vector{Symbol})
        @assert influenced_dofs ⊆ keys(floatingbody.dofs) "the influenced_dofs Symbols must be a key of floatingbody.dof"
        return new(floatingbody, omega, beta, forward_speed, influenced_dofs)
    end
end

# Define RadiationProblem struct as a subtype of LinearPotentialFlowProblem
struct RadiationProblem <: LinearPotentialFlowProblem
    floatingbody::FloatingBody
    omega::Real
    forward_speed::Real
    radiating_dof::Symbol
    influenced_dofs::Vector{Symbol}
    function RadiationProblem(floatingbody::FloatingBody,
        omega::Real,
        forward_speed::Real,
        radiating_dof::Symbol,
        influenced_dofs::Vector{Symbol})
        @assert (radiating_dof in keys(floatingbody.dofs)) "the radiating_dof Symbol must be a key of floatingbody.dof"
        @assert influenced_dofs ⊆ keys(floatingbody.dofs) "the influenced_dofs Symbols must be a key of floatingbody.dof"
        return new(floatingbody, omega, forward_speed, radiating_dof, influenced_dofs)
    end
end

########################## Results #########################################

abstract type LinearPotentialFlowResult end

struct DiffractionResult <: LinearPotentialFlowResult
    problem::DiffractionProblem
    forces::NamedTuple
    function DiffractionResult(problem::DiffractionProblem,
        forces::NamedTuple)
        return new(problem, forces)
    end
end
struct RadiationResult <: LinearPotentialFlowResult
    problem::RadiationProblem
    forces::NamedTuple
    function RadiationResult(problem::RadiationProblem,
        forces::NamedTuple)
        return new(problem, forces)
    end
end

# Convert problem and forces for that problem into a results struct
function make_result(problem::RadiationProblem, forces::NamedTuple)
    return RadiationResult(problem,forces)
end
function make_result(problem::DiffractionProblem, forces::NamedTuple)
    return DiffractionResult(problem,forces)
end


# Convert parameters and problem into a Vector of problems
function problems_from_data(parameters::NamedTuple, floatingbody::FloatingBody)

    # if influenced_dofs not specified, assume all floatingbody dofs are influenced
    if :influenced_dofs in keys(parameters)
        inf_dofs = parameters.influenced_dofs
    else
        inf_dofs = collect(keys(floatingbody.dofs))
    end

    # Forward speed corrections
    if :forward_speeds in keys(parameters)
        forward_speeds = parameters.forward_speeds
    else
        forward_speeds = [0] # assume zero forward speed in not specified
    end

    # There is at least one diffraction problem to solve
    if :wave_directions in keys(parameters)
        diffraction_problems = vec([DiffractionProblem(floatingbody, omega, beta, forward_speed, inf_dofs) 
            for beta in parameters[:wave_directions], 
                omega in parameters[:wave_frequencies],
                forward_speed in forward_speeds])
    else
        diffraction_problems = LinearPotentialFlowProblem[]
    end

    # There is at least one radiation problem to solve
    if :radiating_dofs in keys(parameters)
        radiation_problems = vec([RadiationProblem(floatingbody, omega, forward_speed, rad_dof, inf_dofs)  
            for rad_dof in parameters[:radiating_dofs], 
                omega in parameters[:wave_frequencies],
                forward_speed in forward_speeds])
    else
        radiation_problems = LinearPotentialFlowProblem[]

    end

    return vcat(diffraction_problems, radiation_problems)
end


# Convert Vector of results into NameTuple of hydrodynamic coefficients
# assemble_hydrodynamic_coefficients automatically determines what outputs to compute based on what parameters are specified. 
function assemble_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody, results::Vector{<:LinearPotentialFlowResult})

    omegas = parameters.wave_frequencies


    # Forward speed corrections
    if :forward_speeds in keys(parameters)
        forward_speeds = parameters.forward_speeds
    else
        forward_speeds = [0] # assume zero forward speed in not specified
    end

    if :influenced_dofs in keys(parameters)
        inf_dofs = parameters.influenced_dofs
    else
        inf_dofs = collect(keys(floatingbody.dofs))
    end

    # Diffraction or incident problem 
    if :wave_directions in keys(parameters)
        betas = parameters.wave_directions
        dif_lookup = Dict(
            (omega = r.problem.omega,
            beta = r.problem.beta,
            forward_speed = r.problem.forward_speed) => r.forces 
            for r in results if r isa DiffractionResult
        )
        inc_lookup = Dict(
            (omega = r.problem.omega,
            beta = r.problem.beta,
            forward_speed = r.problem.forward_speed) => FroudeKrylovForce(floatingbody,inf_dofs,r.problem.omega,r.problem.beta)
            for r in results if r isa DiffractionResult
        )
        diffraction_force_data = [
        dif_lookup[(omega=omega,beta=beta,forward_speed=forward_speed)][i] 
        for i in 1:length(inf_dofs), omega in omegas, beta in betas, forward_speed in forward_speeds
        ]
        Froude_Krylov_force_data = [
            inc_lookup[(omega=omega,beta=beta,forward_speed=forward_speed)][i] 
            for i in 1:length(inf_dofs), omega in omegas, beta in betas, forward_speed in forward_speeds
        ]
        excitation_force_data = diffraction_force_data .+ Froude_Krylov_force_data
    else
        diffraction_force_data = []
        Froude_Krylov_force_data = []
        excitation_force_data = []
    end

    if :radiating_dofs in keys(parameters)
        rad_dofs = parameters.radiating_dofs
        rad_lookup = Dict(
            (radiating_dof = r.problem.radiating_dof,
            omega = r.problem.omega,
            forward_speed = r.problem.forward_speed) => r.forces 
            for r in results if r isa RadiationResult
        )
        added_mass_data = [
            real(rad_lookup[(radiating_dof=radiating_dof,
            omega=omega,
            forward_speed=forward_speed)][i]) / omega^2
            for i in 1:length(inf_dofs), radiating_dof in rad_dofs, omega in omegas, forward_speed in forward_speeds
        ]
        radiation_damping_data = [
            imag(rad_lookup[(radiating_dof=radiating_dof,
            omega=omega,
            forward_speed=forward_speed)][i]) / omega
            for i in 1:length(inf_dofs), radiating_dof in rad_dofs, omega in omegas, forward_speed in forward_speeds
        ]
    else
        added_mass_data = []
        radiation_damping_data = []
    end

    
    data = (added_mass=added_mass_data,
    radiation_damping=radiation_damping_data,
    diffraction_force=diffraction_force_data,
    Froude_Krylov_force=Froude_Krylov_force_data,
    excitation_force=excitation_force_data)
    
    return data 
end



# Convert NameTuple of hydrodynamic coefficients into DimStack
function create_DimStack(data::NamedTuple, parameters::NamedTuple, floatingbody::FloatingBody)

    added_mass_data = data.added_mass
    radiation_damping_data = data.radiation_damping
    diffraction_force_data = data.diffraction_force
    Froude_Krylov_force_data = data.Froude_Krylov_force
    excitation_force_data = data.excitation_force    
    
    omegas = parameters.wave_frequencies
    betas = parameters.wave_directions
    rad_dofs = parameters.radiating_dofs
    if :influenced_dofs in keys(parameters)
        inf_dofs = parameters.influenced_dofs
    else
        inf_dofs = collect(keys(floatingbody.dofs))
    end
    # Forward speed corrections
    if :forward_speeds in keys(parameters)
        forward_speeds = parameters.forward_speeds
    else
        forward_speeds = [0] # assume zero forward speed in not specified
    end
     

    radiation_dims = (Dim{:influenced_dofs}(collect(inf_dofs)), 
        Dim{:radiating_dofs}(collect(rad_dofs)),
        Dim{:wave_frequencies}(omegas),
        Dim{:forward_speeds}(forward_speeds))

    diffraction_dims = (Dim{:influenced_dofs}(collect(inf_dofs)),
        Dim{:wave_frequencies}(collect(omegas)),
        Dim{:wave_directions}(betas),
        Dim{:forward_speeds}(forward_speeds))


    added_mass_array = DimArray(added_mass_data, radiation_dims)
    radiation_damping_array = DimArray(radiation_damping_data, radiation_dims)
    excitation_force_array = DimArray(excitation_force_data, diffraction_dims)
    diffraction_force_array = DimArray(diffraction_force_data, diffraction_dims)
    Froude_Krylov_force_array = DimArray(Froude_Krylov_force_data, diffraction_dims)


    DimStack_of_data = DimStack((
        added_mass = added_mass_array,
        radiation_damping = radiation_damping_array,
        excitation_force = excitation_force_array,
        diffraction_force = diffraction_force_array,
        Froude_Krylov_force = Froude_Krylov_force_array))
    return DimStack_of_data 
end



# Compute NamedTuple of of results (with keys added_mass, ...)
# This is differentiable
function compute_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody; direct::Bool=true, gf::String="Wu")
    problems = problems_from_data(parameters, floatingbody)
    results = solve_all_problems(problems; direct=direct, gf=gf)
    data = assemble_hydrodynamic_coefficients(parameters, floatingbody, results)
    return data
end

# This is NOT differentiable (as is) due to DimStack 
function compute_and_label_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody; direct::Bool=true, gf::String="Wu")
    data = compute_hydrodynamic_coefficients(parameters, floatingbody; direct=direct, gf=gf)
    DimStack_of_data = create_DimStack(data, parameters, floatingbody)
    return DimStack_of_data
end