
struct RankineReflected <: GreensFunction end

"""
    DelhommeauRankineReflected()

Free-surface image of [`DelhommeauRankine`](@ref). Prefer [`RankineReflected`](@ref),
which images the default Birk [`Rankine`](@ref).
"""
struct DelhommeauRankineReflected <: GreensFunction end

const _ReflectedRankine = Union{RankineReflected, DelhommeauRankineReflected}
wavenumber_independent(::_ReflectedRankine) = true

_unreflected(::RankineReflected) = Rankine()
_unreflected(::DelhommeauRankineReflected) = DelhommeauRankine()

function greens(gf::_ReflectedRankine, element_1, element_2, wavenumber=nothing)
    return greens(_unreflected(gf), free_surface_symmetry(element_1), element_2)
    # = greens(_unreflected(gf), element_1, free_surface_symmetry(element_2))
end

function gradient_greens(gf::_ReflectedRankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    if with_respect_to_first_variable
        return gradient_greens(_unreflected(gf), element_1, free_surface_symmetry(element_2); with_respect_to_first_variable)
        # = vertical_reflection(gradient_greens(_unreflected(gf), free_surface_symmetry(element_1), element_2; with_respect_to_first_variable))
    else
        return gradient_greens(_unreflected(gf), free_surface_symmetry(element_1), element_2; with_respect_to_first_variable)
        # = vertical_reflection(gradient_greens(_unreflected(gf), element_1, free_surface_symmetry(element_2); with_respect_to_first_variable))
    end
end

function integral(gf::_ReflectedRankine, element_1, element_2, wavenumber=nothing)
    return integral(_unreflected(gf), free_surface_symmetry(element_1), element_2)
    # = integral(_unreflected(gf), element_1, free_surface_symmetry(element_2))
end

vertical_reflection(x::T) where T <: Tuple = (x[1], x[2], -x[3])
vertical_reflection(x::T) where T <: StaticVector = T((x[1], x[2], -x[3]))
vertical_reflection(x::AbstractVector) = [x[1], x[2], -x[3]]

function integral_gradient(gf::_ReflectedRankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    ng = integral_gradient(_unreflected(gf), free_surface_symmetry(element_1), element_2; with_respect_to_first_variable)
    if with_respect_to_first_variable
        return vertical_reflection(ng)
    else
        return ng
    end
end

function both_integral_and_integral_gradient(gf::_ReflectedRankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    g, ng = both_integral_and_integral_gradient(_unreflected(gf), free_surface_symmetry(element_1), element_2; with_respect_to_first_variable)
    if with_respect_to_first_variable
        return g, vertical_reflection(ng)
    else
        return g, ng
    end
end
