# Direct vs indirect and the Green functions live on the formulation, not kwargs.
# `solve(prob, DirectBEM())` has the same shape as SciML `solve(prob, Tsit5())`,
# but the second argument is a BEM formulation, not a time-stepping algorithm.

abstract type BEMFormulation end

struct DirectBEM{G} <: BEMFormulation
    greens::G
end
DirectBEM() = DirectBEM((Rankine(), RankineReflected(), GFWu()))
DirectBEM(gf::AbstractString) = DirectBEM(default_greens_functions(String(gf)))

struct IndirectBEM{G} <: BEMFormulation
    greens::G
end
IndirectBEM() = IndirectBEM((Rankine(), RankineReflected(), GFWu()))
IndirectBEM(gf::AbstractString) = IndirectBEM(default_greens_functions(String(gf)))

is_direct(::DirectBEM) = true
is_direct(::IndirectBEM) = false

function default_greens_functions(gf::String)
    if gf == "Wu"
        wave_gf = GFWu()
    elseif gf == "ExactGuevelDelhommeau"
        wave_gf = ExactGuevelDelhommeau()
    else
        throw(ArgumentError("unknown Green's function \"$gf\"; use \"Wu\" or \"ExactGuevelDelhommeau\""))
    end
    return (Rankine(), RankineReflected(), wave_gf)
end

function bem_formulation(; formulation=nothing, direct::Bool=true, gf::String="Wu", greens_functions=nothing)
    formulation !== nothing && return formulation
    gfs = greens_functions === nothing ? default_greens_functions(gf) : greens_functions
    return direct ? DirectBEM(gfs) : IndirectBEM(gfs)
end
