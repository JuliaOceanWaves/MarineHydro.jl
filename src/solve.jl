

# Solve single problem (one frequency and one radiating dof or wave direction)
function solve_problem(problem::LinearPotentialFlowProblem; direct::Bool=true, gf::String="Wu")
    bc = compute_bc(problem) # computed based on omega, not encoutered_omega

    k = compute_wavenumber(problem.omega)


    if gf=="Wu"  
        S, D = assemble_matrices([Rankine(), RankineReflected(), GFWu()], problem.floatingbody.mesh, k; direct)
    elseif gf=="ExactGuevelDelhommeau"
        S, D = assemble_matrices([Rankine(), RankineReflected(), ExactGuevelDelhommeau()], problem.floatingbody.mesh, k; direct)
    end


    potential = solve(D, S, bc; direct=direct)


    pressure = 1im * SETTINGS.rho * problem.omega * potential


    forces = integrate_pressure(problem.floatingbody, problem.influenced_dofs, pressure) # NamedTuple of complex forces, where each element corresponds to a dof 
    
    
    result = make_result(problem, forces)
    return result 
end

# Solve multiple problems (multiple frequencies, radiating dofs, and/or wave directions)
# Equivalent to Capytaine's solve_all() function. Eventually add parallelization  settings here.
function solve_all_problems(problems::Vector{LinearPotentialFlowProblem}; direct::Bool=true, gf::String="Wu")
    
    results = [solve_problem(problem; direct=direct, gf=gf) for problem in problems]
    
    return results
end