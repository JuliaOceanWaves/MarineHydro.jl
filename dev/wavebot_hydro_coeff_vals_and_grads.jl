using MarineHydro
using DifferentiationInterface 
import ForwardDiff 


# Hydrostatics
function hydrostatic_program(r1, r2, d1, d2)
    rho_w = 1000.0 # density of fluid [kg/m^3]
    g = 9.81 # acceleration due to gravity [m/s^2]
    K = rho_w * g * pi * r1^2 # hydrostatic stiffness [kg/s^2]
    M = rho_w * (pi * r1^2 * d1 + (pi/3) * (d2-d1) * (r1^2 + r1*r2 + r2^2)) # mass of body [kg]
    return [M, K]
end

# Radiation
function radiation_program(mesh, omega, dof) 
    A, B = calculate_radiation_forces(mesh,dof,omega)
    return [A, B]
end

# Diffraction
function diffraction_program(mesh, omega, dof) 
    F_D = DiffractionForce(mesh,omega,dof)
    F_FK = FroudeKrylovForce(mesh,omega,dof)
    F_ex = F_D + F_FK
    return [real(F_ex),imag(F_ex)]
end

# Everything
dof = [0.0,0.0,1.0]

function all_programs(mesh, x, omega)
    r1 = x[1]
    r2 = x[2]
    d1 = x[3]
    d2 = x[4]

    M, K = hydrostatic_program(r1, r2, d1, d2)
    A, B = radiation_program(mesh, omega, dof)
    F_ex_real, F_ex_imag = diffraction_program(mesh, omega, dof) 

    return [M, K, A, B, F_ex_real, F_ex_imag]
end

function compute_for_omega(x, omega)
    mesh = wavebot_mesh(x[1],x[2],x[3],x[4],(3,3,3),10)
    val = all_programs(mesh, x, omega)
    return val
end

backend = AutoForwardDiff()

function compute_all(x_val, omegas)
    AD_grads = [value_and_jacobian(x -> compute_for_omega(x, omega), backend, x_val) for omega in omegas]
    return AD_grads
end


x_val = [0.88, 0.35, 0.16, 0.37]
omegas = [0.2, 0.3]
AD_grads = compute_all(x_val, omegas)
display(AD_grads)