"""
    PhysicalConstants(rho, g)

Default physical constants used when constructing problems.

# Fields
- `rho::Float64`: fluid density [kg/m^3]
- `g::Float64`: acceleration due to gravity [m/s^2]
"""
mutable struct PhysicalConstants
    rho::Float64
    g::Float64
end

const SETTINGS = PhysicalConstants(1000.0, 9.81)

"""
    set_g!(val)

Set the default gravitational acceleration used by newly constructed problems.
"""
set_g!(val) = (SETTINGS.g = val)

"""
    set_rho!(val)

Set the default fluid density used by newly constructed problems.
"""
set_rho!(val) = (SETTINGS.rho = val)
