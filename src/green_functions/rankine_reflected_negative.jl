struct RankineReflectedNegative <: GreensFunction end
struct DelhommeauRankineReflectedNegative <: GreensFunction end

const _NegativeReflected = Union{RankineReflectedNegative, DelhommeauRankineReflectedNegative}
wavenumber_independent(::_NegativeReflected) = true

_positive_reflected(::RankineReflectedNegative) = RankineReflected()
_positive_reflected(::DelhommeauRankineReflectedNegative) = DelhommeauRankineReflected()

function greens(gf::_NegativeReflected, element_1, element_2, wavenumber)
    return -greens(_positive_reflected(gf), element_1, element_2, wavenumber)
end

function gradient_greens(gf::_NegativeReflected, element_1, element_2, wavenumber; with_respect_to_first_variable=false)
    return -gradient_greens(_positive_reflected(gf), element_1, element_2, wavenumber; with_respect_to_first_variable)
end

function integral(gf::_NegativeReflected, element_1, element_2, wavenumber)
    return -integral(_positive_reflected(gf), element_1, element_2, wavenumber)
end

function integral_gradient(gf::_NegativeReflected, element_1, element_2, wavenumber; with_respect_to_first_variable=false)
    return -integral_gradient(_positive_reflected(gf), element_1, element_2, wavenumber; with_respect_to_first_variable)
end
