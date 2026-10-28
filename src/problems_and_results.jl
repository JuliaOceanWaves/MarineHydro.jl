using DimensionalData

########################## Problems #########################################

# Abstract type named LinearPotentialFlowProblem. The diffraction and radiation problems will be subtypes of this type. 
abstract type LinearPotentialFlowProblem end

_problem_scalar_type(vals...) = promote_type(map(typeof, vals)...)

# Define DiffractionProblem struct as a subtype of LinearPotentialFlowProblem.
# `rho` and `g` live on the problem (typed `T`, defaulting from `SETTINGS`) so
# they promote with Duals for ForwardDiff and are Enzyme-active struct fields
# instead of reads from a mutable global.
struct DiffractionProblem{T, B} <: LinearPotentialFlowProblem
    floatingbody::B
    omega::T
    beta::T
    wavenumber::T
    encountered_omega::T
    encountered_beta::T
    encountered_wavenumber::T
    forward_speed::T
    rho::T
    g::T
    influenced_dofs::Vector{Symbol}
    function DiffractionProblem(floatingbody::B,
        omega, beta, wavenumber,
        encountered_omega, encountered_beta, encountered_wavenumber,
        forward_speed, influenced_dofs::Vector{Symbol};
        rho = SETTINGS.rho, g = SETTINGS.g) where B
        influenced_dofs ⊆ keys(floatingbody.dofs) || throw(ArgumentError(
            "each influenced_dofs Symbol must be a key of floatingbody.dofs"))
        T = _problem_scalar_type(omega, beta, wavenumber, encountered_omega,
            encountered_beta, encountered_wavenumber, forward_speed, rho, g)
        return new{T, B}(floatingbody, T(omega), T(beta), T(wavenumber),
            T(encountered_omega), T(encountered_beta), T(encountered_wavenumber),
            T(forward_speed), T(rho), T(g), influenced_dofs)
    end
end

function DiffractionProblem(floatingbody::FloatingBody,
        omega, beta, wavenumber, forward_speed, influenced_dofs::Vector{Symbol};
        rho = SETTINGS.rho, g = SETTINGS.g)
    if forward_speed == 0
        return DiffractionProblem(floatingbody, omega, beta, wavenumber,
            omega, beta, wavenumber, forward_speed, influenced_dofs; rho, g)
    else
        encountered_omega, encountered_wavenumber, encountered_beta =
            compute_encountered_values(omega, beta, forward_speed, g)
        return DiffractionProblem(floatingbody, omega, beta, wavenumber,
            encountered_omega, encountered_beta, encountered_wavenumber,
            forward_speed, influenced_dofs; rho, g)
    end
end

function DiffractionProblem(floatingbody::FloatingBody, omega, beta;
        influenced_dofs::Union{AbstractVector,Nothing}=nothing,
        forward_speed=0,
        wavenumber=nothing,
        rho=SETTINGS.rho,
        g=SETTINGS.g)
    inf = isnothing(influenced_dofs) ?
        collect(Symbol, keys(floatingbody.dofs)) :
        collect(Symbol, influenced_dofs)
    k = isnothing(wavenumber) ? compute_wavenumber(omega, g) : wavenumber
    return DiffractionProblem(floatingbody, omega, beta, k, forward_speed, inf; rho, g)
end

function DiffractionProblem(floatingbody::FloatingBody, omega;
        beta=0.0,
        influenced_dofs::Union{AbstractVector,Nothing}=nothing,
        forward_speed=0,
        wavenumber=nothing,
        rho=SETTINGS.rho,
        g=SETTINGS.g)
    return DiffractionProblem(floatingbody, omega, beta;
        influenced_dofs, forward_speed, wavenumber, rho, g)
end


# `B` is the concrete `FloatingBody{...}` type. Enzyme reverse through an
# abstract `floatingbody::FloatingBody` field (and `Union{T,Nothing}` betas)
# crashed with "illegal insertion" when the mesh was active.
struct RadiationProblem{T, B} <: LinearPotentialFlowProblem
    floatingbody::B
    omega::T
    beta::T
    wavenumber::T
    encountered_omega::T
    encountered_beta::T
    encountered_wavenumber::T
    forward_speed::T
    rho::T
    g::T
    radiating_dof::Symbol
    influenced_dofs::Vector{Symbol}
    function RadiationProblem(floatingbody::B,
        omega, beta, wavenumber,
        encountered_omega, encountered_beta, encountered_wavenumber,
        forward_speed, radiating_dof::Symbol, influenced_dofs::Vector{Symbol};
        rho = SETTINGS.rho, g = SETTINGS.g) where B
        radiating_dof in keys(floatingbody.dofs) || throw(ArgumentError(
            "the radiating_dof Symbol must be a key of floatingbody.dofs"))
        influenced_dofs ⊆ keys(floatingbody.dofs) || throw(ArgumentError(
            "each influenced_dofs Symbol must be a key of floatingbody.dofs"))
        scalars = (omega, wavenumber, encountered_omega, encountered_wavenumber,
            forward_speed, rho, g)
        beta_vals = filter(!isnothing, (beta, encountered_beta))
        T = _problem_scalar_type(scalars..., beta_vals...)
        β = isnothing(beta) ? zero(T) : T(beta)
        βe = isnothing(encountered_beta) ? zero(T) : T(encountered_beta)
        return new{T, B}(floatingbody, T(omega), β, T(wavenumber),
            T(encountered_omega), βe, T(encountered_wavenumber), T(forward_speed),
            T(rho), T(g), radiating_dof, influenced_dofs)
    end
end


function RadiationProblem(floatingbody::FloatingBody,
        omega, beta, wavenumber, forward_speed, radiating_dof::Symbol,
        influenced_dofs::Vector{Symbol};
        rho = SETTINGS.rho, g = SETTINGS.g)
    if forward_speed == 0
        return RadiationProblem(floatingbody, omega, beta, wavenumber,
            omega, beta, wavenumber, forward_speed, radiating_dof, influenced_dofs;
            rho, g)
    else
        encountered_omega, encountered_wavenumber, encountered_beta =
            compute_encountered_values(omega, beta, forward_speed, g)
        return RadiationProblem(floatingbody, omega, beta, wavenumber,
            encountered_omega, encountered_beta, encountered_wavenumber,
            forward_speed, radiating_dof, influenced_dofs; rho, g)
    end
end

function RadiationProblem(floatingbody::FloatingBody, omega;
        radiating_dof::Union{Symbol,Nothing}=nothing,
        influenced_dofs::Union{AbstractVector,Nothing}=nothing,
        beta=nothing,
        forward_speed=0,
        wavenumber=nothing,
        rho=SETTINGS.rho,
        g=SETTINGS.g)
    inf = isnothing(influenced_dofs) ?
        collect(Symbol, keys(floatingbody.dofs)) :
        collect(Symbol, influenced_dofs)
    rad = isnothing(radiating_dof) ? only(inf) : radiating_dof
    k = isnothing(wavenumber) ? compute_wavenumber(omega, g) : wavenumber
    return RadiationProblem(floatingbody, omega, beta, k, forward_speed, rad, inf; rho, g)
end

function remake(prob::RadiationProblem; kwargs...)
    floatingbody = get(kwargs, :floatingbody, prob.floatingbody)
    omega = get(kwargs, :omega, prob.omega)
    forward_speed = get(kwargs, :forward_speed, prob.forward_speed)
    radiating_dof = get(kwargs, :radiating_dof, prob.radiating_dof)
    influenced_dofs = collect(Symbol, get(kwargs, :influenced_dofs, prob.influenced_dofs))
    rho = get(kwargs, :rho, prob.rho)
    g = get(kwargs, :g, prob.g)
    wavenumber = haskey(kwargs, :wavenumber) ? kwargs[:wavenumber] :
        (haskey(kwargs, :omega) ? compute_wavenumber(omega, g) : prob.wavenumber)
    beta = if haskey(kwargs, :beta)
        kwargs[:beta]
    elseif forward_speed == 0
        nothing
    else
        prob.beta
    end
    return RadiationProblem(floatingbody, omega, beta, wavenumber, forward_speed,
        radiating_dof, influenced_dofs; rho, g)
end

function remake(prob::DiffractionProblem; kwargs...)
    floatingbody = get(kwargs, :floatingbody, prob.floatingbody)
    omega = get(kwargs, :omega, prob.omega)
    forward_speed = get(kwargs, :forward_speed, prob.forward_speed)
    influenced_dofs = collect(Symbol, get(kwargs, :influenced_dofs, prob.influenced_dofs))
    rho = get(kwargs, :rho, prob.rho)
    g = get(kwargs, :g, prob.g)
    wavenumber = haskey(kwargs, :wavenumber) ? kwargs[:wavenumber] :
        (haskey(kwargs, :omega) ? compute_wavenumber(omega, g) : prob.wavenumber)
    beta = get(kwargs, :beta, prob.beta)
    return DiffractionProblem(floatingbody, omega, beta, wavenumber, forward_speed,
        influenced_dofs; rho, g)
end

########################## Results #########################################

abstract type LinearPotentialFlowResult end

# `forces` is concretely typed (NamedTuple keys and value types in the type
# parameter) so field access is inferred for ForwardDiff and Enzyme.
struct DiffractionResult{P<:DiffractionProblem, F<:NamedTuple} <: LinearPotentialFlowResult
    problem::P
    forces::F
end
struct RadiationResult{P<:RadiationProblem, F<:NamedTuple} <: LinearPotentialFlowResult
    problem::P
    forces::F
end

# Added mass and damping (`A = Re(F)/ω²`, `B = Im(F)/ω`). Dual `ω` / Dual forces
# stay Dual. Reverse-mode engines need a real scalar: `added_mass(sol).Heave`.
_result_omega(res::RadiationResult) =
    res.problem.forward_speed == 0 ? res.problem.omega : res.problem.encountered_omega

added_mass(res::RadiationResult) = map(F -> real(F) / _result_omega(res)^2, res.forces)
radiation_damping(res::RadiationResult) = map(F -> imag(F) / _result_omega(res), res.forces)

radiation_coefficients(res::RadiationResult) = (
    added_mass = added_mass(res),
    radiation_damping = radiation_damping(res),
)

diffraction_force(res::DiffractionResult) = res.forces
froude_krylov_force(res::DiffractionResult) =
    froude_krylov_force(res.problem, res.problem.influenced_dofs)
excitation_force(res::DiffractionResult) =
    map(+, diffraction_force(res), froude_krylov_force(res))

# Convert problem and forces for that problem into a results struct
function make_result(problem::RadiationProblem, forces::NamedTuple)
    return RadiationResult(problem,forces)
end
function make_result(problem::DiffractionProblem, forces::NamedTuple)
    return DiffractionResult(problem,forces)
end


# NamedTuple keys are part of the type. `Val{(:foo in K)}()` picks a method at
# compile time so Enzyme never sees the other branch (no Union, no `nothing`).
_inf_dofs(p::NamedTuple{K}, body) where K = _inf_dofs(Val{(:influenced_dofs in K)}(), p, body)
_inf_dofs(::Val{true}, p, body) = collect(Symbol, p.influenced_dofs)
_inf_dofs(::Val{false}, p, body) = collect(Symbol, keys(body.dofs))

_speeds(p::NamedTuple{K}, ωs) where K = _speeds(Val{(:forward_speeds in K)}(), p, ωs)
_speeds(::Val{true}, p, ωs) = p.forward_speeds
_speeds(::Val{false}, p, ωs) = [zero(eltype(ωs))]

_empty_problems(::Type{P}, ::Type{T}, body) where {P,T} = P{T, typeof(body)}[]

function problems_from_data(parameters::NamedTuple, floatingbody::FloatingBody)
    ωs = parameters.wave_frequencies
    inf = _inf_dofs(parameters, floatingbody)
    Us = _speeds(parameters, ωs)
    return (
        radiation = _make_radiation(parameters, floatingbody, ωs, Us, inf),
        diffraction = _make_diffraction(parameters, floatingbody, ωs, Us, inf),
    )
end

_make_radiation(p::NamedTuple{K}, body, ωs, Us, inf) where K =
    _make_radiation(Val{(:radiating_dofs in K)}(), p, body, ωs, Us, inf)

function _make_radiation(::Val{false}, p, body, ωs, Us, inf)
    return _empty_problems(RadiationProblem, promote_type(eltype(ωs), eltype(Us)), body)
end

_make_radiation(::Val{true}, p::NamedTuple{K}, body, ωs, Us, inf) where K =
    _make_radiation_grid(Val{(:forward_speeds in K)}(), p, body, ωs, Us, inf)

function _make_radiation_grid(::Val{false}, p, body, ωs, Us, inf)
    β = zero(promote_type(eltype(ωs), eltype(Us)))
    return vec([RadiationProblem(body, ω, β, compute_wavenumber(ω), U, dof, inf)
        for dof in p.radiating_dofs, ω in ωs, U in Us])
end

function _make_radiation_grid(::Val{true}, p, body, ωs, Us, inf)
    return vec([RadiationProblem(body, ω, β, compute_wavenumber(ω), U, dof, inf)
        for β in p.wave_directions, dof in p.radiating_dofs, ω in ωs, U in Us])
end

_make_diffraction(p::NamedTuple{K}, body, ωs, Us, inf) where K =
    _make_diffraction(Val{(:wave_directions in K)}(), p, body, ωs, Us, inf)

function _make_diffraction(::Val{false}, p, body, ωs, Us, inf)
    return _empty_problems(DiffractionProblem, promote_type(eltype(ωs), eltype(Us)), body)
end

function _make_diffraction(::Val{true}, p, body, ωs, Us, inf)
    return vec([DiffractionProblem(body, ω, β, compute_wavenumber(ω), U, inf)
        for β in p.wave_directions, ω in ωs, U in Us])
end

# Column-major index of `vec(A)` when A has size (n1, n2, n3) or (n1, n2, n3, n4).
_lin(i1, i2, i3, n1, n2) = i1 + n1 * ((i2 - 1) + n2 * (i3 - 1))
_lin(i1, i2, i3, i4, n1, n2, n3) = i1 + n1 * ((i2 - 1) + n2 * ((i3 - 1) + n3 * (i4 - 1)))

_empty_coeff(ωs) = zeros(eltype(ωs), 0)

function assemble_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody,
        radiation::AbstractVector, diffraction::AbstractVector)
    ωs = parameters.wave_frequencies
    inf = _inf_dofs(parameters, floatingbody)
    Us = _speeds(parameters, ωs)
    A, B = _pack_radiation(parameters, radiation, inf, ωs, Us)
    Fd, Fk, Fe = _pack_diffraction(parameters, diffraction, inf, ωs, Us)
    return (added_mass=A, radiation_damping=B,
        diffraction_force=Fd, Froude_Krylov_force=Fk, excitation_force=Fe)
end

_pack_radiation(p::NamedTuple{K}, results, inf, ωs, Us) where K =
    _pack_radiation(Val{(:radiating_dofs in K)}(), p, results, inf, ωs, Us)

function _pack_radiation(::Val{false}, p, results, inf, ωs, Us)
    z = _empty_coeff(ωs)
    return z, z
end

_pack_radiation(::Val{true}, p::NamedTuple{K}, results, inf, ωs, Us) where K =
    _pack_radiation_grid(Val{(:forward_speeds in K)}(), p, results, inf, ωs, Us)

# Explicit typed loops, not N-d comprehensions: `ProductIterator` states are
# Union-typed, which boxes every element and trips Enzyme's strict type
# analysis (IllegalTypeAnalysisException). Mutating fills are Enzyme-safe;
# reverse engines only need to differentiate through the stored values.
_radiation_coeff_type(results, ωs) =
    isempty(results) ? float(eltype(ωs)) : typeof(values(added_mass(results[1]))[1])

_diffraction_coeff_type(results, ωs) =
    isempty(results) ? complex(float(eltype(ωs))) :
    promote_type(typeof(values(diffraction_force(results[1]))[1]),
        typeof(values(froude_krylov_force(results[1]))[1]))

function _pack_radiation_grid(::Val{false}, p, results, inf, ωs, Us)
    n_inf, n_rad, n_ω, n_U = length(inf), length(p.radiating_dofs), length(ωs), length(Us)
    A = Array{_radiation_coeff_type(results, ωs)}(undef, n_inf, n_rad, n_ω, n_U)
    B = similar(A)
    for iU in 1:n_U, iω in 1:n_ω, ir in 1:n_rad
        res = results[_lin(ir, iω, iU, n_rad, n_ω)]
        a = values(added_mass(res))
        b = values(radiation_damping(res))
        for i in 1:n_inf
            A[i, ir, iω, iU] = a[i]
            B[i, ir, iω, iU] = b[i]
        end
    end
    return A, B
end

function _pack_radiation_grid(::Val{true}, p, results, inf, ωs, Us)
    n_inf = length(inf)
    n_β, n_rad, n_ω, n_U = length(p.wave_directions), length(p.radiating_dofs), length(ωs), length(Us)
    A = Array{_radiation_coeff_type(results, ωs)}(undef, n_inf, n_rad, n_ω, n_U, n_β)
    B = similar(A)
    for iβ in 1:n_β, iU in 1:n_U, iω in 1:n_ω, ir in 1:n_rad
        res = results[_lin(iβ, ir, iω, iU, n_β, n_rad, n_ω)]
        a = values(added_mass(res))
        b = values(radiation_damping(res))
        for i in 1:n_inf
            A[i, ir, iω, iU, iβ] = a[i]
            B[i, ir, iω, iU, iβ] = b[i]
        end
    end
    return A, B
end

_pack_diffraction(p::NamedTuple{K}, results, inf, ωs, Us) where K =
    _pack_diffraction(Val{(:wave_directions in K)}(), p, results, inf, ωs, Us)

function _pack_diffraction(::Val{false}, p, results, inf, ωs, Us)
    z = _empty_coeff(ωs)
    return z, z, z
end

function _pack_diffraction(::Val{true}, p, results, inf, ωs, Us)
    n_inf, n_β, n_ω, n_U = length(inf), length(p.wave_directions), length(ωs), length(Us)
    Fd = Array{_diffraction_coeff_type(results, ωs)}(undef, n_inf, n_ω, n_β, n_U)
    Fk = similar(Fd)
    for iU in 1:n_U, iβ in 1:n_β, iω in 1:n_ω
        res = results[_lin(iβ, iω, iU, n_β, n_ω)]
        fd = values(diffraction_force(res))
        fk = values(froude_krylov_force(res))
        for i in 1:n_inf
            Fd[i, iω, iβ, iU] = fd[i]
            Fk[i, iω, iβ, iU] = fk[i]
        end
    end
    return Fd, Fk, Fd .+ Fk
end



# Convert NameTuple of hydrodynamic coefficients into DimStack.
# Dimension ticks are metadata (not differentiated). Coefficient arrays can be
# Duals (ForwardDiff) or already-computed gradient arrays; wrap those the same way.
_axis_val(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_axis_val(x::AbstractArray) = map(_axis_val, x)
_axis_val(x) = x

"""
    label_hydrodynamic_coefficients(data, parameters, floatingbody)

Attach `DimensionalData` dimension labels to a NamedTuple of hydrodynamic
coefficient arrays, returning a `DimStack`. Label after AD, not through it:
reverse-mode engines should differentiate `hydrodynamic_coefficients` and the
primal and gradient arrays can each be labeled here.
"""
function label_hydrodynamic_coefficients(data::NamedTuple, parameters::NamedTuple,
        floatingbody::FloatingBody)

    added_mass_data = data.added_mass
    radiation_damping_data = data.radiation_damping
    diffraction_force_data = data.diffraction_force
    Froude_Krylov_force_data = data.Froude_Krylov_force
    excitation_force_data = data.excitation_force    
    
    omegas = _axis_val(parameters.wave_frequencies)
    inf_dofs = :influenced_dofs in keys(parameters) ?
        collect(parameters.influenced_dofs) : collect(keys(floatingbody.dofs))
    forward_speeds = _axis_val(get(parameters, :forward_speeds, [0]))

    layers = NamedTuple()

    if !isempty(added_mass_data)
        rad_dofs = collect(parameters.radiating_dofs)
        if forward_speeds == [0] || forward_speeds == [0.0]
            radiation_dims = (Dim{:influenced_dofs}(inf_dofs),
                Dim{:radiating_dofs}(rad_dofs),
                Dim{:wave_frequencies}(omegas),
                Dim{:forward_speeds}(forward_speeds))
        else
            betas = _axis_val(parameters.wave_directions)
            radiation_dims = (Dim{:influenced_dofs}(inf_dofs),
                Dim{:radiating_dofs}(rad_dofs),
                Dim{:wave_frequencies}(omegas),
                Dim{:forward_speeds}(forward_speeds),
                Dim{:wave_directions}(betas))
        end
        layers = merge(layers, (
            added_mass = DimArray(added_mass_data, radiation_dims),
            radiation_damping = DimArray(radiation_damping_data, radiation_dims),
        ))
    end

    if !isempty(diffraction_force_data)
        betas = _axis_val(parameters.wave_directions)
        diffraction_dims = (Dim{:influenced_dofs}(inf_dofs),
            Dim{:wave_frequencies}(collect(omegas)),
            Dim{:wave_directions}(betas),
            Dim{:forward_speeds}(forward_speeds))
        layers = merge(layers, (
            excitation_force = DimArray(excitation_force_data, diffraction_dims),
            diffraction_force = DimArray(diffraction_force_data, diffraction_dims),
            Froude_Krylov_force = DimArray(Froude_Krylov_force_data, diffraction_dims),
        ))
    end

    return DimStack(layers)
end

# Deprecated CamelCase-era name; prefer `label_hydrodynamic_coefficients`.
const create_DimStack = label_hydrodynamic_coefficients



# NamedTuple of real arrays (keys added_mass, radiation_damping, …). This is the
# differentiable payload. Label with `create_DimStack` after AD, not through it.
# Map over frequencies: Enzyme reverse of a Vector of problems only adjoints ω₁.
# `wave_frequencies` may be a scalar (`ForwardDiff.derivative` / FD on one ω).
_freq_vec(ωs::AbstractArray) = ωs
_freq_vec(ωs) = [ωs]

function hydrodynamic_coefficients(floatingbody::FloatingBody, parameters::NamedTuple,
        formulation::BEMFormulation = DirectBEM())
    chunks = map(_freq_vec(parameters.wave_frequencies)) do ω
        _hydro_one_frequency(floatingbody, parameters, ω, formulation)
    end
    return _stack_frequency_chunks(chunks)
end

function _hydro_one_frequency(floatingbody, parameters, ω, formulation)
    params1 = merge(parameters, (wave_frequencies = [ω],))
    problems = problems_from_data(params1, floatingbody)
    radiation = solve(problems.radiation, formulation)
    diffraction = solve(problems.diffraction, formulation)
    return assemble_hydrodynamic_coefficients(params1, floatingbody, radiation, diffraction)
end

function _stack_or_empty(xs, dim)
    x1 = xs[1]
    ndims(x1) <= 1 && isempty(x1) && return x1
    length(xs) == 1 && return x1
    return cat(xs...; dims=dim)
end

function _stack_frequency_chunks(chunks::AbstractVector)
    return (
        added_mass = _stack_or_empty(map(c -> c.added_mass, chunks), 3),
        radiation_damping = _stack_or_empty(map(c -> c.radiation_damping, chunks), 3),
        diffraction_force = _stack_or_empty(map(c -> c.diffraction_force, chunks), 2),
        Froude_Krylov_force = _stack_or_empty(map(c -> c.Froude_Krylov_force, chunks), 2),
        excitation_force = _stack_or_empty(map(c -> c.excitation_force, chunks), 2),
    )
end

function compute_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody;
        formulation=nothing, direct::Bool=true, gf::String="Wu", greens_functions=nothing)
    return hydrodynamic_coefficients(floatingbody, parameters,
        bem_formulation(; formulation, direct, gf, greens_functions))
end

function compute_and_label_hydrodynamic_coefficients(parameters::NamedTuple, floatingbody::FloatingBody;
        formulation=nothing, direct::Bool=true, gf::String="Wu", greens_functions=nothing)
    data = compute_hydrodynamic_coefficients(parameters, floatingbody; formulation, direct, gf, greens_functions)
    return label_hydrodynamic_coefficients(data, parameters, floatingbody)
end