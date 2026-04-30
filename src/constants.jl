
mutable struct constants
    rho::Float64  # Fluid density [kg/m^3]
    g::Float64    # Acceleration due to gravity [m/s^2]
end

# default settings for constants
const SETTINGS = constants(1000.0, 9.81)

# functions to change default settings for constants
set_g!(val)   = (SETTINGS.g = val)
set_rho!(val) = (SETTINGS.rho = val)

