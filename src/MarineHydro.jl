
module MarineHydro

using ForwardDiff
using StaticArrays
using LinearAlgebra
using LinearAlgebra: cross, dot, norm
using DimensionalData
using ChainRulesCore
using KernelAbstractions
import LinearSolve

const τ̅ = 2π

include("constants.jl")
export PhysicalConstants, SETTINGS, set_g!, set_rho!

include("green_functions/abstract_greens_function.jl")
export greens, gradient_greens, integral, integral_gradient, with_reduced_coordinates
include("green_functions/rankine.jl")
export DelhommeauRankine
include("green_functions/rankine_vectorized.jl")
export Rankine
include("green_functions/rankine_reflected.jl")
export RankineReflected, DelhommeauRankineReflected
include("green_functions/rankine_reflected_negative.jl")
export RankineReflectedNegative, DelhommeauRankineReflectedNegative


include("green_functions/wu.jl")
export GFWu
include("green_functions/exact_Guevel_Delhommeau.jl")
export ExactGuevelDelhommeau

include("meshes.jl")
export Mesh, StaticArraysMesh, smesh, element, combine_meshes, wavebot_mesh

include("bodies.jl")
export FloatingBody, combine_floatingbodies

include("formulations.jl")
export BEMFormulation, DirectBEM, IndirectBEM

include("problems_and_results.jl")
export LinearPotentialFlowProblem, DiffractionProblem, RadiationProblem, remake
export LinearPotentialFlowResult, DiffractionResult, RadiationResult
export added_mass, radiation_damping, radiation_coefficients
export diffraction_force, froude_krylov_force, excitation_force
export make_result, problems_from_data, assemble_hydrodynamic_coefficients
export create_DimStack, label_hydrodynamic_coefficients, hydrodynamic_coefficients
export compute_hydrodynamic_coefficients, compute_and_label_hydrodynamic_coefficients

include("matrix_assembly.jl")
export assemble_matrices, assemble_matrices_broadcasting, assemble_matrices_ka, assemble_matrix_wu, solve

include("waves.jl")
export airy_bc, airy_waves_pressure, airy_waves_velocity, airy_waves_potential
export radiation_bc, integrate_pressure, compute_bc, compute_wavenumber, compute_encountered_values
export calculate_radiation_forces
# Deprecated CamelCase-era names, kept for compatibility.
export AiryBC, FroudeKrylovForce, DiffractionForce

include("solve.jl")
export solve, solve_problem, solve_all_problems

include("ad_rules.jl")
include("ad_rules_geometry.jl")
using .GeometryAD: fd_mesh_rules!, fd_mesh_function, make_sphere_mesher, sphere_mesh
export fd_mesh_rules!, fd_mesh_function, make_sphere_mesher, sphere_mesh

end
