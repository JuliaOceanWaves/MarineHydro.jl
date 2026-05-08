

# Solve single problem (one frequency and one radiating dof or wave direction)
function solve_problem(problem::LinearPotentialFlowProblem; direct::Bool=true, gf::String="Wu")

    bc = compute_bc(problem) 

    if problem.forward_speed==0
        omega = problem.omega
        wavenumber = problem.wavenumber
    else
        @assert direct==false "Forward speed problems are only developed with the indirect method"
        omega = problem.encountered_omega
        wavenumber = problem.encountered_wavenumber # use encountered_wavenumber in gfs
    end


    if gf=="Wu"  
        selected_GF = GFWu()
    elseif gf=="ExactGuevelDelhommeau"
        selected_GF = ExactGuevelDelhommeau()
    end

    S, D = assemble_matrices([Rankine(), RankineReflected(), selected_GF], problem.floatingbody.mesh, wavenumber; direct=direct)

    potential, sources = solve(D, S, bc; direct=direct)

    pressure = 1im * SETTINGS.rho * omega * potential # uses encountered_omega

    if problem.forward_speed!=0
        # change normals to all be unit vector in x direction
        S, K = assemble_matrices([Rankine(), RankineReflected(), selected_GF], problem.floatingbody.mesh, wavenumber; direct=direct, all_normals=[1,0,0])
        nabla_phi_dot_x = K * sources
        pressure .+= SETTINGS.rho * problem.forward_speed * nabla_phi_dot_x
    end

    forces = integrate_pressure(problem.floatingbody, problem.influenced_dofs, pressure) # NamedTuple of complex forces, where each element corresponds to an influenced dof 
    
    result = make_result(problem, forces)
    return result 
end

# Solve multiple problems (multiple frequencies, radiating dofs, and/or wave directions)
# Equivalent to Capytaine's solve_all() function. Eventually add parallelization  settings here.
function solve_all_problems(problems::Vector{LinearPotentialFlowProblem}; direct::Bool=true, gf::String="Wu")
    
    results = [solve_problem(problem; direct=direct, gf=gf) for problem in problems]
    
    return results
end