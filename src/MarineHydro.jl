
module MarineHydro

using ForwardDiff
using StaticArrays
using LinearAlgebra
using LinearAlgebra: cross, dot, norm
using ImplicitAD: implicit_linear
using DimensionalData

const τ̅ = 2π

include("constants.jl")
export SETTINGS, set_g!, set_rho!

include("green_functions/abstract_greens_function.jl")
export greens, gradient_greens, integral, integral_gradient, with_reduced_coordinates
include("green_functions/rankine.jl")
export Rankine
include("green_functions/rankine_reflected.jl")
export RankineReflected
include("green_functions/rankine_reflected_negative.jl")
export RankineReflectedNegative
include("green_functions/wu.jl")
export GFWu
include("green_functions/exact_Guevel_Delhommeau.jl")
export ExactGuevelDelhommeau

include("meshes.jl")
export Mesh, element, combine_meshes, +, wavebot_mesh

include("bodies.jl")
export FloatingBody, combine_floatingbodies, +

include("problems_and_results.jl")
export LinearPotentialFlowProblem, DiffractionProblem, RadiationProblem
export LinearPotentialFlowResult, DiffractionResult, RadiationResult
export make_result, problems_from_data, assemble_hydrodynamic_coefficients
export create_DimStack, compute_hydrodynamic_coefficients, compute_and_label_hydrodynamic_coefficients

include("matrix_assembly.jl")
export assemble_matrices, assemble_matrix_wu, solve

include("waves.jl")
export FroudeKrylovForce, AiryBC, airy_waves_pressure, airy_waves_velocity,airy_waves_potential
export radiation_bc, integrate_pressure, compute_bc
export calculate_radiation_forces, DiffractionForce, diffraction_force

include("solve.jl")
export solve_problem, solve_all_problems

end
