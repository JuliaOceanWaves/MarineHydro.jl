"""
    MarineHydro.ReverseAD

Reverse-mode rules for Zygote (ChainRules) and Enzyme. Primals stay in
`green_functions/` and `matrix_assembly.jl`. Both engines share the same VJPs:

- Birk `velocity_potential` / `velocity_derivatives` / `birk_phi_and_velocity`:
  analytic field-point VJP; panel-corner VJP via ForwardDiff through the primal
- `_linearsolve` / `gpu_linsolve`: implicit-function theorem (`z = A'\\ȳ`,
  `dA -= z y'`)
- `_mulvec`: analytic matvec VJP (`dS = ȳ bc'`, `dbc = S' ȳ`)
"""
module ReverseAD

using ..MarineHydro: velocity_potential, velocity_derivatives, velocity_hessian,
    birk_phi_and_velocity, _linearsolve, _mulvec, gpu_linsolve, _backend_ldiv
using ChainRulesCore
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Enzyme: EnzymeRules, Const, Active, Annotation

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

function _flatten_corners(local_corners)
    return SVector{12}(
        local_corners[1][1], local_corners[1][2], local_corners[1][3],
        local_corners[2][1], local_corners[2][2], local_corners[2][3],
        local_corners[3][1], local_corners[3][2], local_corners[3][3],
        local_corners[4][1], local_corners[4][2], local_corners[4][3],
    )
end

function _unflatten_corners(v)
    return (
        SVector(v[1], v[2], v[3]),
        SVector(v[4], v[5], v[6]),
        SVector(v[7], v[8], v[9]),
        SVector(v[10], v[11], v[12]),
    )
end

# Panel corners have no closed-form Birk derivative; fill those cotangents
# with ForwardDiff through the primal. Duals take the primal path, so these
# pullbacks do not recurse into the rrules / EnzymeRules below.
function _potential_corners_adjoint(ȳ, x, y, z, local_corners)
    g = ForwardDiff.gradient(_flatten_corners(local_corners)) do v
        return velocity_potential(x, y, z, _unflatten_corners(v))
    end
    return _unflatten_corners(ȳ .* g)
end

function _velocity_corners_adjoint(ȳ, x, y, z, local_corners)
    J = ForwardDiff.jacobian(_flatten_corners(local_corners)) do v
        return velocity_derivatives(x, y, z, _unflatten_corners(v))
    end
    return _unflatten_corners(J' * ȳ)
end

function _cotangent_svector(ȳ)
    ȳu = unthunk(ȳ)
    return SVector{3}(ȳu[1], ȳu[2], ȳu[3])
end

@inline _enz_arg_tangent(::Const, _) = nothing
@inline _enz_arg_tangent(::Active, Δ) = Δ

_enzyme_zeros_like(x, config) =
    EnzymeRules.width(config) == 1 ? zero(x) :
    ntuple(_ -> zero(x), Val(EnzymeRules.width(config)))

# ---------------------------------------------------------------------------
# Birk velocity_potential  (eq. 46)
# dφ/d(x,y,z) = velocity; dφ/d(corners) via ForwardDiff
# ---------------------------------------------------------------------------

function ChainRulesCore.rrule(::typeof(velocity_potential), x, y, z, local_corners)
    φ = velocity_potential(x, y, z, local_corners)
    v = velocity_derivatives(x, y, z, local_corners)
    function velocity_potential_pullback(ȳ)
        if ȳ isa AbstractZero
            return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
        end
        ȳu = unthunk(ȳ)
        Δcorners = _potential_corners_adjoint(ȳu, x, y, z, local_corners)
        return (NoTangent(), ȳu * v[1], ȳu * v[2], ȳu * v[3], Δcorners)
    end
    return φ, velocity_potential_pullback
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(velocity_potential)},
    ::Type{RT},
    x::Annotation,
    y::Annotation,
    z::Annotation,
    local_corners::Annotation,
) where {RT}
    tape = (x.val, y.val, z.val, local_corners.val)
    res = velocity_potential(tape...)
    retres = EnzymeRules.needs_primal(config) ? res : nothing
    dres = EnzymeRules.needs_shadow(config) ? zero(res) : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, tape)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(velocity_potential)},
    dret::Active,
    tape,
    x::Annotation,
    y::Annotation,
    z::Annotation,
    local_corners::Annotation,
)
    xv, yv, zv, cv = tape
    ȳ = dret.val
    v = velocity_derivatives(xv, yv, zv, cv)
    Δc = local_corners isa Const ? nothing : _potential_corners_adjoint(ȳ, xv, yv, zv, cv)
    return (
        _enz_arg_tangent(x, ȳ * v[1]),
        _enz_arg_tangent(y, ȳ * v[2]),
        _enz_arg_tangent(z, ȳ * v[3]),
        _enz_arg_tangent(local_corners, Δc),
    )
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(velocity_potential)},
    ::Type{<:Const},
    ::Any,
    ::Annotation, ::Annotation, ::Annotation, ::Annotation,
)
    return (nothing, nothing, nothing, nothing)
end

# ---------------------------------------------------------------------------
# Birk velocity_derivatives  (eqs. 47–49)
# dv/d(x,y,z) = Hessian; dv/d(corners) via ForwardDiff
# ---------------------------------------------------------------------------

function ChainRulesCore.rrule(::typeof(velocity_derivatives), x, y, z, local_corners)
    v = velocity_derivatives(x, y, z, local_corners)
    H = velocity_hessian(x, y, z, local_corners)
    function velocity_derivatives_pullback(ȳ)
        if ȳ isa AbstractZero
            return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
        end
        ȳv = _cotangent_svector(ȳ)
        gx = H' * ȳv
        Δcorners = _velocity_corners_adjoint(ȳv, x, y, z, local_corners)
        return (NoTangent(), gx[1], gx[2], gx[3], Δcorners)
    end
    return v, velocity_derivatives_pullback
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(velocity_derivatives)},
    ::Type{RT},
    x::Annotation,
    y::Annotation,
    z::Annotation,
    local_corners::Annotation,
) where {RT}
    tape = (x.val, y.val, z.val, local_corners.val)
    res = velocity_derivatives(tape...)
    retres = EnzymeRules.needs_primal(config) ? res : nothing
    dres = EnzymeRules.needs_shadow(config) ? zero(res) : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, tape)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(velocity_derivatives)},
    dret::Active,
    tape,
    x::Annotation,
    y::Annotation,
    z::Annotation,
    local_corners::Annotation,
)
    xv, yv, zv, cv = tape
    ȳ = dret.val
    g = velocity_hessian(xv, yv, zv, cv)' * ȳ
    Δc = local_corners isa Const ? nothing : _velocity_corners_adjoint(ȳ, xv, yv, zv, cv)
    return (
        _enz_arg_tangent(x, g[1]),
        _enz_arg_tangent(y, g[2]),
        _enz_arg_tangent(z, g[3]),
        _enz_arg_tangent(local_corners, Δc),
    )
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(velocity_derivatives)},
    ::Type{<:Const},
    ::Any,
    ::Annotation, ::Annotation, ::Annotation, ::Annotation,
)
    return (nothing, nothing, nothing, nothing)
end

# ---------------------------------------------------------------------------
# Fused Birk (φ, v) kernel used by `_vrankine_from_geom`.
# dφ/d(x,y,z) = v (the second primal output); dv/d(x,y,z) = Hessian.
# Corner cotangents via ForwardDiff through the fused primal, as above.
# ---------------------------------------------------------------------------

function _phi_v_corners_adjoint(ȳφ, ȳv, x, y, z, local_corners)
    J = ForwardDiff.jacobian(_flatten_corners(local_corners)) do c
        φ, v = birk_phi_and_velocity(x, y, z, _unflatten_corners(c))
        return SVector(φ, v[1], v[2], v[3])
    end
    w = SVector(ȳφ, ȳv[1], ȳv[2], ȳv[3])
    return _unflatten_corners(J' * w)
end

@inline _phi_cotangent(ȳφ, φ) =
    (ȳφ === nothing || ȳφ isa AbstractZero) ? zero(φ) : ȳφ
@inline _v_cotangent(ȳv, v) =
    (ȳv === nothing || ȳv isa AbstractZero) ? zero(v) : _cotangent_svector(ȳv)

function ChainRulesCore.rrule(::typeof(birk_phi_and_velocity), x, y, z, local_corners)
    φ, v = birk_phi_and_velocity(x, y, z, local_corners)
    function birk_phi_and_velocity_pullback(ȳ)
        if ȳ isa AbstractZero
            return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
        end
        ȳu = unthunk(ȳ)
        ȳφ = _phi_cotangent(ȳu[1], φ)
        ȳv = _v_cotangent(ȳu[2], v)
        H = velocity_hessian(x, y, z, local_corners)
        g = ȳφ * v + H' * ȳv
        Δcorners = _phi_v_corners_adjoint(ȳφ, ȳv, x, y, z, local_corners)
        return (NoTangent(), g[1], g[2], g[3], Δcorners)
    end
    return (φ, v), birk_phi_and_velocity_pullback
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(birk_phi_and_velocity)},
    ::Type{RT},
    x::Annotation,
    y::Annotation,
    z::Annotation,
    local_corners::Annotation,
) where {RT}
    res = birk_phi_and_velocity(x.val, y.val, z.val, local_corners.val)
    tape = (x.val, y.val, z.val, local_corners.val, res[2])
    retres = EnzymeRules.needs_primal(config) ? res : nothing
    dres = EnzymeRules.needs_shadow(config) ? map(zero, res) : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, tape)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(birk_phi_and_velocity)},
    dret::Active,
    tape,
    x::Annotation,
    y::Annotation,
    z::Annotation,
    local_corners::Annotation,
)
    xv, yv, zv, cv, v = tape
    ȳφ = _phi_cotangent(dret.val[1], v[1])
    ȳv = _v_cotangent(dret.val[2], v)
    H = velocity_hessian(xv, yv, zv, cv)
    g = ȳφ * v + H' * ȳv
    Δc = local_corners isa Const ? nothing : _phi_v_corners_adjoint(ȳφ, ȳv, xv, yv, zv, cv)
    return (
        _enz_arg_tangent(x, g[1]),
        _enz_arg_tangent(y, g[2]),
        _enz_arg_tangent(z, g[3]),
        _enz_arg_tangent(local_corners, Δc),
    )
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(birk_phi_and_velocity)},
    ::Type{<:Const},
    ::Any,
    ::Annotation, ::Annotation, ::Annotation, ::Annotation,
)
    return (nothing, nothing, nothing, nothing)
end

# ---------------------------------------------------------------------------
# Matvec: y = S bc.  dS = ȳ bc', dbc = S' ȳ  (adjoint, not transpose)
# ---------------------------------------------------------------------------

function ChainRulesCore.rrule(::typeof(_mulvec), S, bc)
    y = S * bc
    function _mulvec_pullback(ȳ)
        ȳu = unthunk(ȳ)
        if ȳu isa AbstractZero
            return (NoTangent(), ZeroTangent(), ZeroTangent())
        end
        return (NoTangent(), ȳu * bc', S' * ȳu)
    end
    return y, _mulvec_pullback
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(_mulvec)},
    ::Type{RT},
    S::Annotation{<:Array},
    bc::Annotation{<:AbstractVecOrMat},
) where {RT}
    cache_S = EnzymeRules.overwritten(config)[2] ? copy(S.val) : S.val
    cache_bc = EnzymeRules.overwritten(config)[3] ? copy(bc.val) : bc.val
    res = _mulvec(cache_S, cache_bc)
    dres = EnzymeRules.needs_shadow(config) ? _enzyme_zeros_like(res, config) : nothing
    retres = EnzymeRules.needs_primal(config) ? res : nothing
    cache = (cache_S, cache_bc, dres)
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, cache)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(_mulvec)},
    ::Type{RT},
    cache,
    S::Annotation{<:Array},
    bc::Annotation{<:AbstractVecOrMat},
) where {RT}
    cache_S, cache_bc, dys = cache
    width = EnzymeRules.width(config)
    dys_t = width == 1 ? (dys,) : dys
    for i in 1:width
        dy = dys_t[i]
        if !(S isa Const)
            dS = width == 1 ? S.dval : S.dval[i]
            dS .+= dy * cache_bc'
        end
        if !(bc isa Const)
            dbc = width == 1 ? bc.dval : bc.dval[i]
            dbc .+= cache_S' * dy
        end
        dy .= eltype(dy)(0)
    end
    return (nothing, nothing)
end

# ---------------------------------------------------------------------------
# Linear solve: y = A \ b.  z = A' \ ȳ,  dA -= z y',  db += z
# LinearSolve's own solve! EnzymeRules use transpose(y) and are ~2× off
# on complex BEM; these rules intercept _linearsolve / gpu_linsolve instead.
# ---------------------------------------------------------------------------

function ChainRulesCore.rrule(::typeof(_linearsolve), A, b)
    y = _linearsolve(A, b)
    function _linearsolve_pullback(ȳ)
        ȳu = unthunk(ȳ)
        if ȳu isa AbstractZero
            return (NoTangent(), ZeroTangent(), ZeroTangent())
        end
        λ = _linearsolve(A', ȳu)
        return (NoTangent(), -λ * y', λ)
    end
    return y, _linearsolve_pullback
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(_linearsolve)},
    ::Type{RT},
    A::Annotation{<:Array},
    b::Annotation{<:AbstractArray},
) where {RT}
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : A.val
    res = _linearsolve(cache_A, b.val)
    dres = EnzymeRules.needs_shadow(config) ? _enzyme_zeros_like(res, config) : nothing
    retres = EnzymeRules.needs_primal(config) ? res : nothing
    cache_res = EnzymeRules.needs_primal(config) ? copy(res) : res
    cache = (cache_res, dres, cache_A)
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, cache)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(_linearsolve)},
    ::Type{RT},
    cache,
    A::Annotation{<:Array},
    b::Annotation{<:AbstractArray},
) where {RT}
    y, dys, cache_A = cache
    width = EnzymeRules.width(config)
    dys_t = width == 1 ? (dys,) : dys
    for i in 1:width
        dy = dys_t[i]
        z = _linearsolve(cache_A', dy)
        if !(A isa Const)
            dA = width == 1 ? A.dval : A.dval[i]
            dA .-= z * y'
        end
        if !(b isa Const)
            db = width == 1 ? b.dval : b.dval[i]
            db .+= z
        end
        dy .= eltype(dy)(0)
    end
    return (nothing, nothing)
end

function ChainRulesCore.rrule(::typeof(gpu_linsolve), A, b)
    x = _backend_ldiv(A, b)
    function gpu_linsolve_pullback(ȳ)
        ȳu = unthunk(ȳ)
        if ȳu isa AbstractZero
            return (NoTangent(), ZeroTangent(), ZeroTangent())
        end
        λ = _backend_ldiv(A', ȳu)
        return (NoTangent(), -λ * x', λ)
    end
    return x, gpu_linsolve_pullback
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(gpu_linsolve)},
    ::Type{RT},
    A::Annotation{<:AbstractArray},
    b::Annotation{<:AbstractArray},
) where {RT}
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : A.val
    res = gpu_linsolve(cache_A, b.val)
    dres = EnzymeRules.needs_shadow(config) ? _enzyme_zeros_like(res, config) : nothing
    retres = EnzymeRules.needs_primal(config) ? res : nothing
    cache_res = EnzymeRules.needs_primal(config) ? copy(res) : res
    cache = (cache_res, dres, cache_A)
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, cache)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(gpu_linsolve)},
    ::Type{RT},
    cache,
    A::Annotation{<:AbstractArray},
    b::Annotation{<:AbstractArray},
) where {RT}
    y, dys, cache_A = cache
    width = EnzymeRules.width(config)
    dys_t = width == 1 ? (dys,) : dys
    for i in 1:width
        dy = dys_t[i]
        z = gpu_linsolve(adjoint(cache_A), dy)
        if !(A isa Const)
            dA = width == 1 ? A.dval : A.dval[i]
            dA .-= z * y'
        end
        if !(b isa Const)
            db = width == 1 ? b.dval : b.dval[i]
            db .+= z
        end
        dy .= eltype(dy)(0)
    end
    return (nothing, nothing)
end

end # module ReverseAD
