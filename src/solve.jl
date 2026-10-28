
# Place hull BCs onto hull+lid without setindex! (Zygote) and without BLAS
# `ger` (Enzyme reverse cannot load `zger_64_` on some platforms).
function _pad_hull_bc(bc_on_hull, hull_mask)
    v = vec(bc_on_hull)
    n = length(hull_mask)
    length(v) == n && return v
    T = eltype(v)
    idx = findall(hull_mask)
    return T[sum(i == idx[k] ? v[k] : zero(T) for k in eachindex(idx)) for i in 1:n]
end

function _problem_omega_k(problem)
    if problem.forward_speed == 0
        return problem.omega, problem.wavenumber
    else
        return problem.encountered_omega, problem.encountered_wavenumber
    end
end

# Lid type is a parameter (`Nothing` vs mesh). Dispatch, do not `isnothing`.
_mesh_including_lid(problem) = _body_mesh_lid(problem.floatingbody)
_body_mesh_lid(fb::FloatingBody{<:Any, Nothing}) = (fb.mesh, trues(fb.mesh.nfaces))
function _body_mesh_lid(fb::FloatingBody)
    mesh = fb.mesh + fb.lid_mesh
    hull_mask = eachrow(mesh.faces) .∈ Ref(eachrow(fb.mesh.faces))
    return mesh, hull_mask
end

# Use the mesh already on the body. `FloatingBody{<:Mesh}` stays on the dense
# comprehension path; `FloatingBody{<:StaticArraysMesh}` uses the vectorized
# kernels. Convert once when you build the body — not inside the solve.
_assembly_mesh(mesh) = mesh

function _maybe_forward_speed_pressure(problem, wavenumber, mesh, hull_mask, sources,
        pressure_on_hull, formulation)
    iszero(problem.forward_speed) && return pressure_on_hull
    is_direct(formulation) && throw(ArgumentError(
        "nonzero forward speed is not supported with the direct method; use IndirectBEM (direct = false)"))
    _, K = assemble_matrices(formulation.greens, mesh, wavenumber; direct=false, all_normals=[1, 0, 0])
    # `_mulvec`, not BLAS `*`: Enzyme reverse of zgemv needs `zger_64_`, which
    # fails to load on some platforms (same workaround as the D·bc matvec).
    partial_phi_partial_x = _mulvec(K, sources)
    return pressure_on_hull + problem.rho * problem.forward_speed * partial_phi_partial_x[hull_mask]
end

function _finish_solve(problem, omega, wavenumber, S, D, mesh, hull_mask, bc, formulation)
    direct = is_direct(formulation)
    potential, sources = solve(D, S, bc; direct)
    pressure = 1im * problem.rho * omega * potential
    pressure_on_hull = _maybe_forward_speed_pressure(problem, wavenumber, mesh, hull_mask,
        sources, pressure[hull_mask], formulation)
    forces = integrate_pressure(problem.floatingbody, problem.influenced_dofs, pressure_on_hull)
    return make_result(problem, forces)
end

# `solve(prob)` / `solve(prob, DirectBEM())` — same name as the linear-system
# `solve(D, S, bc)`, dispatched on the first argument.
function solve(problem::LinearPotentialFlowProblem, formulation::BEMFormulation=DirectBEM())
    omega, wavenumber = _problem_omega_k(problem)
    mesh, hull_mask = _mesh_including_lid(problem)
    bc = _pad_hull_bc(vec(compute_bc(problem)), hull_mask)
    S, D = assemble_matrices(formulation.greens, _assembly_mesh(mesh), wavenumber; direct=is_direct(formulation))
    return _finish_solve(problem, omega, wavenumber, S, D, mesh, hull_mask, bc, formulation)
end

function solve_problem(problem::LinearPotentialFlowProblem;
        formulation=nothing, direct::Bool=true, gf::String="Wu", greens_functions=nothing)
    return solve(problem, bem_formulation(; formulation, direct, gf, greens_functions))
end

_empty_results(::Type{P}) where {P<:RadiationProblem} = RadiationResult{P}[]
_empty_results(::Type{P}) where {P<:DiffractionProblem} = DiffractionResult{P}[]

function solve(problems::AbstractVector{P}, formulation::BEMFormulation=DirectBEM()) where {P<:LinearPotentialFlowProblem}
    isempty(problems) && return _empty_results(P)
    return map(p -> solve(p, formulation), problems)
end

function solve_all_problems(problems::AbstractVector{<:LinearPotentialFlowProblem};
        formulation=nothing, direct::Bool=true, gf::String="Wu", greens_functions=nothing)
    return solve(problems, bem_formulation(; formulation, direct, gf, greens_functions))
end
