using DimensionalData

########################## Problems #########################################

# Abstract type named LinearPotentialFlowProblem. The diffraction and radiation problems will be subtypes of this type. 
abstract type LinearPotentialFlowProblem end

# Define DiffractionProblem struct as a subtype of LinearPotentialFlowProblem
struct DiffractionProblem <: LinearPotentialFlowProblem
    floatingbody::FloatingBody
    omega::Real
    beta::Real
    influenced_dofs::Vector{Symbol}
    function DiffractionProblem(floatingbody::FloatingBody,
        omega::Real,
        beta::Real,
        influenced_dofs::Vector{Symbol})
        @assert influenced_dofs ⊆ keys(floatingbody.dofs) "the influenced_dofs Symbols must be a key of floatingbody.dof"
        return new(floatingbody, omega, beta, influenced_dofs)
    end
end

# Define RadiationProblem struct as a subtype of LinearPotentialFlowProblem
struct RadiationProblem <: LinearPotentialFlowProblem
    floatingbody::FloatingBody
    omega::Real
    radiating_dof::Symbol
    influenced_dofs::Vector{Symbol}
    function RadiationProblem(floatingbody::FloatingBody,
        omega::Real,
        radiating_dof::Symbol,
        influenced_dofs::Vector{Symbol})
        @assert (radiating_dof in keys(floatingbody.dofs)) "the radiating_dof Symbol must be a key of floatingbody.dof"
        @assert influenced_dofs ⊆ keys(floatingbody.dofs) "the influenced_dofs Symbols must be a key of floatingbody.dof"
        return new(floatingbody, omega, radiating_dof, influenced_dofs)
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


    # There is at least one diffraction problem to solve
    if :wave_directions in keys(parameters)
        diffraction_problems = vec([DiffractionProblem(floatingbody, omega, beta, inf_dofs) 
            for beta in parameters[:wave_directions], 
                omega in parameters[:wave_frequencies]])
    else
        diffraction_problems = LinearPotentialFlowProblem[]
    end

    # There is at least one radiation problem to solve
    if :radiating_dofs in keys(parameters)
        radiation_problems = vec([RadiationProblem(floatingbody, omega, dof, inf_dofs)  
            for dof in parameters[:radiating_dofs], 
                omega in parameters[:wave_frequencies]])
    else
        radiation_problems = LinearPotentialFlowProblem[]

    end

    return vcat(diffraction_problems, radiation_problems)
end


# Convert Vector of results into NameTuple of hydrodynamic coefficients
# assemble_hydrodynamic_coefficients automatically determines what outputs to compute based on what parameters are specified. 
function assemble_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody, results::Vector{<:LinearPotentialFlowResult})

    omegas = parameters.wave_frequencies

    # if :wave_directions in keys(parameters)
    #     betas = parameters.wave_directions
    # else
    #     betas = [0.0]        
    # end
    # if :radiating_dofs in keys(parameters)
    #     rad_dofs = parameters.radiating_dofs
    # else
    #     rad_dofs = [:no_radiating_dofs]         
    # end

    if :influenced_dofs in keys(parameters)
        inf_dofs = parameters.influenced_dofs
    else
        inf_dofs = collect(keys(floatingbody.dofs))
    end


    if :wave_directions in keys(parameters)
        betas = parameters.wave_directions
        dif_lookup = Dict(
            (omega = r.problem.omega, beta = r.problem.beta) => r.forces 
            for r in results if r isa DiffractionResult
        )
        inc_lookup = Dict(
            (omega = r.problem.omega, beta = r.problem.beta) => FroudeKrylovForce(floatingbody,inf_dofs,r.problem.omega,r.problem.beta)
            for r in results if r isa DiffractionResult
        )
        diffraction_force_data = [
        dif_lookup[(omega=omega,beta=beta)][i] 
        for i in 1:length(inf_dofs), omega in omegas, beta in betas
        ]
        Froude_Krylov_force_data = [
            inc_lookup[(omega=omega,beta=beta)][i] 
            for i in 1:length(inf_dofs), omega in omegas, beta in betas
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
            (radiating_dof = r.problem.radiating_dof, omega = r.problem.omega) => r.forces 
            for r in results if r isa RadiationResult
        )
        added_mass_data = [
            real(rad_lookup[(radiating_dof=radiating_dof, omega=omega)][i]) / omega^2
            for i in 1:length(inf_dofs), radiating_dof in rad_dofs, omega in omegas
        ]
        radiation_damping_data = [
            imag(rad_lookup[(radiating_dof=radiating_dof, omega=omega)][i]) / omega
            for i in 1:length(inf_dofs), radiating_dof in rad_dofs, omega in omegas
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
     

    radiation_dims = (Dim{:influenced_dofs}(collect(inf_dofs)), 
        Dim{:radiating_dofs}(collect(rad_dofs)),
        Dim{:wave_frequencies}(omegas))

    diffraction_dims = (Dim{:influenced_dofs}(collect(inf_dofs)),
        Dim{:wave_frequencies}(collect(omegas)),
        Dim{:wave_directions}(betas))


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
function compute_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody)
    problems = problems_from_data(parameters, floatingbody)
    results = solve_all_problems(problems)
    data = assemble_hydrodynamic_coefficients(parameters, floatingbody, results)
    return data
end

# This is NOT differentiable (as is) due to DimStack 
function compute_and_label_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody)
    data = compute_hydrodynamic_coefficients(parameters, floatingbody)
    DimStack_of_data = create_DimStack(data, parameters, floatingbody)
    return DimStack_of_data
end