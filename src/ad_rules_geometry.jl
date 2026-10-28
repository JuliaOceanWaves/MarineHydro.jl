"""
    MarineHydro.GeometryAD

Finite-difference reverse rules for mesh *generators* that Enzyme and Zygote
cannot trace (Capytaine / Python, or any other opaque mesher).

A generator is a function `f(x) -> Mesh` or `StaticArraysMesh` (or a Capytaine
mesh `PyObject`). `x` is either a `Real` (one design parameter) or an
`AbstractVector` of them. Topology (panel count / connectivity) must stay
fixed as `x` changes.

Capytaine is never traced. `fd_mesh_rules!(f)` installs:

- Zygote / Enzyme reverse: central-difference the mesh arrays, then
  `∂J/∂xᵢ = ⟨ȳ_vertices, ∂V/∂xᵢ⟩ + ⟨ȳ_centers, ∂C/∂xᵢ⟩ + ⋯`
- ForwardDiff: a `Dual` method on `f` that seeds Dual mesh fields from the
  same finite differences, so `StaticArraysMesh(f(p))` works with Dual `p`

Faces are treated as discrete (no cotangent).

# Example

```julia
using MarineHydro, PyCall, DifferentiationInterface, Enzyme
cpt = pyimport("capytaine")

function cylinder_smesh(p)
    StaticArraysMesh(cpt.mesh_horizontal_cylinder(
        radius=p[1], length=p[2], center=(0.0, 0.0, 0.0),
        faces_max_radius=0.5).immersed_part())
end
fd_mesh_rules!(cylinder_smesh)

const ω = 1.2
A33(p) = added_mass(
    solve(RadiationProblem(FloatingBody(cylinder_smesh(p), [:Heave], "cyl"), ω), DirectBEM())
).Heave

p = [1.0, 2.0]
gradient(A33, AutoForwardDiff(), p)
gradient(A33, AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)), p)
```
"""
module GeometryAD

using ..MarineHydro: Mesh, StaticArraysMesh, _mesh_to_smesh, _mesh_to_smesh_impl
using ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent, AbstractZero, unthunk
using LinearAlgebra: dot
using PyCall
using ForwardDiff
using ForwardDiff: Dual, value, partials, npartials, Partials
using StaticArrays: SVector
using Enzyme: EnzymeRules, Const, Active, Annotation

const _DEFAULT_H = 1e-5
const _MESH_FD_FIELDS = (:vertices, :centers, :normals, :areas, :radii)

_as_mesh(m::Mesh) = Mesh(
    float.(m.vertices), Int.(m.faces), float.(m.centers), float.(m.normals),
    float.(m.areas), float.(m.radii), m.nvertices, m.nfaces)
_as_mesh(m::PyObject) = _as_mesh(Mesh(m))

function _zero_mesh_like(m::Mesh)
    return Mesh(zero(m.vertices), m.faces, zero(m.centers), zero(m.normals),
        zero(m.areas), zero(m.radii), m.nvertices, m.nfaces)
end

function _mesh_scale_diff(mp::Mesh, mm::Mesh, den)
    return Mesh(
        (mp.vertices .- mm.vertices) ./ den, mp.faces,
        (mp.centers .- mm.centers) ./ den,
        (mp.normals .- mm.normals) ./ den,
        (mp.areas .- mm.areas) ./ den,
        (mp.radii .- mm.radii) ./ den,
        mp.nvertices, mp.nfaces)
end

function _fd_geom(m::Mesh)
    return _as_mesh(m)
end
_fd_geom(m::StaticArraysMesh) = m
_fd_geom(m::PyObject) = _as_mesh(m)

function _smesh_scale_diff(sp::StaticArraysMesh, sm::StaticArraysMesh, den)
    verts = [(sp.vertices[i] - sm.vertices[i]) / den for i in eachindex(sp.vertices)]
    cents = [(sp.centers[i] - sm.centers[i]) / den for i in eachindex(sp.centers)]
    norms = [(sp.normals[i] - sm.normals[i]) / den for i in eachindex(sp.normals)]
    areas = (sp.areas .- sm.areas) ./ den
    radii = (sp.radii .- sm.radii) ./ den
    return StaticArraysMesh(verts, sp.faces, cents, norms, areas, radii,
        sp.nvertices, sp.nfaces)
end

function _geom_scale_diff(a::Mesh, b::Mesh, den)
    return _mesh_scale_diff(a, b, den)
end
function _geom_scale_diff(a::StaticArraysMesh, b::StaticArraysMesh, den)
    return _smesh_scale_diff(a, b, den)
end

function _fd_perturb(f, r::Real, h)
    return _geom_scale_diff(_fd_geom(f(r + h)), _fd_geom(f(r - h)), 2h)
end

function _fd_perturb_index(f, p::AbstractVector, i, h)
    pp = collect(float.(p))
    pm = copy(pp)
    pp[i] += h
    pm[i] -= h
    return _geom_scale_diff(_fd_geom(f(pp)), _fd_geom(f(pm)), 2h)
end

function _field_cotangent(ȳ, name)
    ȳ isa AbstractZero && return nothing
    hasproperty(ȳ, name) || return nothing
    t = getproperty(ȳ, name)
    (t isa AbstractZero || t === nothing) && return nothing
    return t
end

function _mesh_vjp(ȳ, ∂m::Mesh)
    ȳu = unthunk(ȳ)
    s = 0.0
    for name in _MESH_FD_FIELDS
        t = _field_cotangent(ȳu, name)
        t === nothing && continue
        s += dot(vec(Float64.(t)), vec(Float64.(getfield(∂m, name))))
    end
    return s
end

function _flat_dot(a, b)
    s = 0.0
    @inbounds for i in eachindex(a, b)
        ai, bi = a[i], b[i]
        if ai isa AbstractArray
            s += dot(Float64.(ai), Float64.(bi))
        else
            s += Float64(ai) * Float64(bi)
        end
    end
    return s
end

function _smesh_vjp(ȳ, ∂s::StaticArraysMesh)
    ȳu = unthunk(ȳ)
    s = 0.0
    for name in _MESH_FD_FIELDS
        t = _field_cotangent(ȳu, name)
        t === nothing && continue
        s += _flat_dot(t, getfield(∂s, name))
    end
    return s
end

_geom_vjp(ȳ, ∂m::Mesh) = _mesh_vjp(ȳ, ∂m)
_geom_vjp(ȳ, ∂s::StaticArraysMesh) = _smesh_vjp(ȳ, ∂s)

_zero_geom_like(m::Mesh) = _zero_mesh_like(m)
_zero_geom_like(s::StaticArraysMesh) = _zero_smesh_like(s)
_zero_geom_shadow!(m::Mesh) = _zero_mesh_shadow!(m)
_zero_geom_shadow!(s::StaticArraysMesh) = _zero_smesh_shadow!(s)

function _zero_mesh_shadow!(m::Mesh)
    m.vertices .= 0
    m.centers .= 0
    m.normals .= 0
    m.areas .= 0
    m.radii .= 0
    return m
end

# Dual p cannot enter Capytaine. Seed Dual fields from the same central
# differences used by the reverse VJP, so ForwardDiff.gradient(f ∘ BEM, p) works.
function _dual_from_partials(val, d_dp, p::AbstractVector{<:Dual})
    Tag = typeof(first(p)).parameters[1]
    N = npartials(first(p))
    V = typeof(float(val))
    parts = ntuple(N) do k
        s = zero(V)
        @inbounds for i in eachindex(p)
            s += V(d_dp[i]) * partials(p[i], k)
        end
        s
    end
    return Dual{Tag}(V(val), Partials{N,V}(parts))
end

function _dual_from_partials(val, dval, r::Dual)
    Tag = typeof(r).parameters[1]
    V = typeof(float(val))
    return Dual{Tag}(V(val), V(dval) * partials(r))
end

function _dualize_array(A0, dAs, p::AbstractVector{<:Dual})
    out = similar(A0, eltype(p))
    n = length(p)
    for I in eachindex(A0)
        out[I] = _dual_from_partials(A0[I], ntuple(i -> dAs[i][I], n), p)
    end
    return out
end

function _dualize_array(A0, dA, r::Dual)
    out = similar(A0, typeof(r))
    for I in eachindex(A0)
        out[I] = _dual_from_partials(A0[I], dA[I], r)
    end
    return out
end

function _mesh_from_duals(f, p::AbstractVector{<:Dual}, h)
    p0 = collect(value.(p))
    m0 = _as_mesh(f(p0))
    n = length(p)
    ∂m = ntuple(i -> _fd_perturb_index(f, p0, i, h), n)
    return Mesh(
        _dualize_array(m0.vertices, ntuple(i -> ∂m[i].vertices, n), p),
        m0.faces,
        _dualize_array(m0.centers, ntuple(i -> ∂m[i].centers, n), p),
        _dualize_array(m0.normals, ntuple(i -> ∂m[i].normals, n), p),
        _dualize_array(m0.areas, ntuple(i -> ∂m[i].areas, n), p),
        _dualize_array(m0.radii, ntuple(i -> ∂m[i].radii, n), p),
        m0.nvertices, m0.nfaces)
end

function _mesh_from_duals(f, r::Dual, h)
    r0 = value(r)
    m0 = _as_mesh(f(r0))
    ∂m = _fd_perturb(f, r0, h)
    return Mesh(
        _dualize_array(m0.vertices, ∂m.vertices, r),
        m0.faces,
        _dualize_array(m0.centers, ∂m.centers, r),
        _dualize_array(m0.normals, ∂m.normals, r),
        _dualize_array(m0.areas, ∂m.areas, r),
        _dualize_array(m0.radii, ∂m.radii, r),
        m0.nvertices, m0.nfaces)
end

function _dualize_svecs(v0, dvs, p::AbstractVector{<:Dual})
    T = eltype(p)
    n = length(p)
    return map(eachindex(v0)) do j
        SVector{3,T}(ntuple(k -> _dual_from_partials(v0[j][k], ntuple(i -> dvs[i][j][k], n), p), 3))
    end
end

function _dualize_svecs(v0, dv, r::Dual)
    T = typeof(r)
    return map(eachindex(v0)) do j
        SVector{3,T}(ntuple(k -> _dual_from_partials(v0[j][k], dv[j][k], r), 3))
    end
end

function _smesh_from_duals(f, p::AbstractVector{<:Dual}, h)
    p0 = collect(value.(p))
    s0 = _fd_geom(f(p0))::StaticArraysMesh
    n = length(p)
    ∂s = ntuple(i -> _fd_perturb_index(f, p0, i, h), n)
    T = eltype(p)
    return StaticArraysMesh(
        _dualize_svecs(s0.vertices, ntuple(i -> ∂s[i].vertices, n), p),
        s0.faces,
        _dualize_svecs(s0.centers, ntuple(i -> ∂s[i].centers, n), p),
        _dualize_svecs(s0.normals, ntuple(i -> ∂s[i].normals, n), p),
        _dualize_array(s0.areas, ntuple(i -> ∂s[i].areas, n), p),
        _dualize_array(s0.radii, ntuple(i -> ∂s[i].radii, n), p),
        s0.nvertices, s0.nfaces)
end

function _smesh_from_duals(f, r::Dual, h)
    r0 = value(r)
    s0 = _fd_geom(f(r0))::StaticArraysMesh
    ∂s = _fd_perturb(f, r0, h)
    return StaticArraysMesh(
        _dualize_svecs(s0.vertices, ∂s.vertices, r),
        s0.faces,
        _dualize_svecs(s0.centers, ∂s.centers, r),
        _dualize_svecs(s0.normals, ∂s.normals, r),
        _dualize_array(s0.areas, ∂s.areas, r),
        _dualize_array(s0.radii, ∂s.radii, r),
        s0.nvertices, s0.nfaces)
end

function _geom_from_duals(f, x::Dual, h)
    g0 = _fd_geom(f(value(x)))
    return g0 isa StaticArraysMesh ? _smesh_from_duals(f, x, h) : _mesh_from_duals(f, x, h)
end
function _geom_from_duals(f, p::AbstractVector{<:Dual}, h)
    g0 = _fd_geom(f(collect(value.(p))))
    return g0 isa StaticArraysMesh ? _smesh_from_duals(f, p, h) : _mesh_from_duals(f, p, h)
end

# Callable wrapper: Dual methods cannot be `@eval`'d onto closures.
# Named functions like `cylinder` get Dual methods in `fd_mesh_rules!` instead.
struct FDMesher{F}
    generate::F
    h::Float64
end

(m::FDMesher)(x::Real) = _as_mesh(m.generate(x))
(m::FDMesher)(p::AbstractVector) = _as_mesh(m.generate(p))
(m::FDMesher)(x::Dual) = _mesh_from_duals(m.generate, x, m.h)
(m::FDMesher)(p::AbstractVector{<:Dual}) = _mesh_from_duals(m.generate, p, m.h)

function ChainRulesCore.rrule(m::FDMesher, r::Real)
    mesh = _as_mesh(m.generate(r))
    function mesh_pullback(ȳ)
        ȳu = unthunk(ȳ)
        ȳu isa AbstractZero && return (NoTangent(), ZeroTangent())
        return (NoTangent(), _mesh_vjp(ȳu, _fd_perturb(m.generate, r, m.h)))
    end
    return mesh, mesh_pullback
end

function ChainRulesCore.rrule(m::FDMesher, p::AbstractVector)
    mesh = _as_mesh(m.generate(p))
    function mesh_pullback(ȳ)
        ȳu = unthunk(ȳ)
        ȳu isa AbstractZero && return (NoTangent(), ZeroTangent())
        dp = similar(p, typeof(float(p[1])))
        for i in eachindex(p)
            dp[i] = _mesh_vjp(ȳu, _fd_perturb_index(m.generate, p, i, m.h))
        end
        return (NoTangent(), dp)
    end
    return mesh, mesh_pullback
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    m::Const{<:FDMesher},
    ::Type{RT},
    x::Annotation{<:Real},
) where {RT}
    mesh = _as_mesh(m.val.generate(x.val))
    retres = EnzymeRules.needs_primal(config) ? mesh : nothing
    dres = EnzymeRules.needs_shadow(config) ? _zero_mesh_like(mesh) : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, (x.val, dres, m.val.h, m.val.generate))
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    m::Const{<:FDMesher},
    ::Type{RT},
    tape,
    x::Active,
) where {RT}
    rval, dys, h, generate = tape
    dys === nothing && return (zero(rval),)
    dr::Float64 = _mesh_vjp(dys, _fd_perturb(generate, rval, h))
    _zero_mesh_shadow!(dys)
    return (dr,)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{<:FDMesher},
    ::Type{RT},
    ::Any,
    ::Const,
) where {RT}
    return (nothing,)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    m::Const{<:FDMesher},
    ::Type{RT},
    p::Annotation{<:AbstractVector},
) where {RT}
    mesh = _as_mesh(m.val.generate(p.val))
    retres = EnzymeRules.needs_primal(config) ? mesh : nothing
    dres = EnzymeRules.needs_shadow(config) ? _zero_mesh_like(mesh) : nothing
    cache_p = EnzymeRules.overwritten(config)[2] ? copy(p.val) : p.val
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, (cache_p, dres, m.val.h, m.val.generate))
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{<:FDMesher},
    ::Type{RT},
    tape,
    p::Annotation{<:AbstractVector},
) where {RT}
    pval, dys, h, generate = tape
    if !(p isa Const) && dys !== nothing
        dp = p.dval
        for i in eachindex(pval)
            dp[i] += _mesh_vjp(dys, _fd_perturb_index(generate, pval, i, h))
        end
        _zero_mesh_shadow!(dys)
    end
    return (nothing,)
end

function _is_generic_function(f)
    n = nameof(f)
    startswith(string(n), "#") && return false
    mod = parentmodule(f)
    return isdefined(mod, n) && getfield(mod, n) === f
end

function _add_dual_methods!(f, hh)
    _is_generic_function(f) || return f
    n = nameof(f)
    mod = parentmodule(f)
    dualT = Dual
    from_duals = _geom_from_duals
    Core.eval(mod, quote
        function $n(r::$dualT)
            return $from_duals($f, r, $hh)
        end
        function $n(p::AbstractVector{<:$dualT})
            return $from_duals($f, p, $hh)
        end
    end)
    return f
end

function _zero_smesh_like(s::StaticArraysMesh)
    z = zero(eltype(s.areas))
    return StaticArraysMesh(
        [zero(v) for v in s.vertices],
        s.faces,
        [zero(c) for c in s.centers],
        [zero(n) for n in s.normals],
        fill(z, length(s.areas)),
        fill(z, length(s.radii)),
        s.nvertices, s.nfaces)
end

function _zero_smesh_shadow!(s::StaticArraysMesh)
    @inbounds for i in eachindex(s.vertices)
        s.vertices[i] = zero(s.vertices[i])
    end
    @inbounds for i in eachindex(s.centers)
        s.centers[i] = zero(s.centers[i])
        s.normals[i] = zero(s.normals[i])
    end
    fill!(s.areas, zero(eltype(s.areas)))
    fill!(s.radii, zero(eltype(s.radii)))
    return s
end

function _accumulate_smesh_into_mesh!(dmesh::Mesh, ds::StaticArraysMesh)
    @inbounds for i in 1:ds.nvertices
        v = ds.vertices[i]
        dmesh.vertices[i, 1] += v[1]
        dmesh.vertices[i, 2] += v[2]
        dmesh.vertices[i, 3] += v[3]
    end
    @inbounds for i in 1:ds.nfaces
        c = ds.centers[i]
        n = ds.normals[i]
        dmesh.centers[i, 1] += c[1]
        dmesh.centers[i, 2] += c[2]
        dmesh.centers[i, 3] += c[3]
        dmesh.normals[i, 1] += n[1]
        dmesh.normals[i, 2] += n[2]
        dmesh.normals[i, 3] += n[3]
        dmesh.areas[i] += ds.areas[i]
        dmesh.radii[i] += ds.radii[i]
    end
    return dmesh
end

# Enzyme cannot reverse the Mesh → Vector{SVector} conversion loop
# ("illegal insertion" in type analysis). Copy smesh cotangents back onto Mesh.
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    ::Const{typeof(_mesh_to_smesh)},
    ::Type{RT},
    mesh::Annotation{<:Mesh},
) where {RT}
    sm = _mesh_to_smesh_impl(mesh.val)
    retres = EnzymeRules.needs_primal(config) ? sm : nothing
    dres = EnzymeRules.needs_shadow(config) ? _zero_smesh_like(sm) : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, dres)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfig,
    ::Const{typeof(_mesh_to_smesh)},
    ::Type{RT},
    tape,
    mesh::Annotation{<:Mesh},
) where {RT}
    dys = tape
    if !(mesh isa Const) && dys !== nothing
        _accumulate_smesh_into_mesh!(mesh.dval, dys)
        _zero_smesh_shadow!(dys)
    end
    return (nothing,)
end

"""
    fd_mesh_rules!(f; h=1e-5)

Install Zygote (`ChainRules`) reverse rules, Enzyme reverse rules, and a
ForwardDiff `Dual` method for `f`, which must be

- `f(x::Real) -> Mesh` / `StaticArraysMesh` / Capytaine mesh, or
- `f(p::AbstractVector) -> Mesh` / `StaticArraysMesh` / Capytaine mesh.

Call this once after defining `f`. Capytaine is never differentiated; the
rules finite-difference the returned mesh arrays. After this, `f(p)` with
Dual `p` returns a Dual `Mesh` or Dual `StaticArraysMesh`, so
`ForwardDiff.gradient` through `f(p)` works. Enzyme reverse uses the same
finite-difference VJP.
"""
function fd_mesh_rules!(f; h::Real=_DEFAULT_H)
    hh = float(h)

    @eval function ChainRulesCore.rrule(::typeof($f), r::Real)
        m = _fd_geom($f(r))
        function mesh_pullback(ȳ)
            ȳu = unthunk(ȳ)
            ȳu isa AbstractZero && return (NoTangent(), ZeroTangent())
            return (NoTangent(), _geom_vjp(ȳu, _fd_perturb($f, r, $hh)))
        end
        return m, mesh_pullback
    end

    @eval function ChainRulesCore.rrule(::typeof($f), p::AbstractVector)
        m = _fd_geom($f(p))
        function mesh_pullback(ȳ)
            ȳu = unthunk(ȳ)
            if ȳu isa AbstractZero
                return (NoTangent(), ZeroTangent())
            end
            dp = similar(p, typeof(float(p[1])))
            for i in eachindex(p)
                dp[i] = _geom_vjp(ȳu, _fd_perturb_index($f, p, i, $hh))
            end
            return (NoTangent(), dp)
        end
        return m, mesh_pullback
    end

    @eval function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        ::Const{typeof($f)},
        ::Type{RT},
        x::Annotation{<:Real},
    ) where {RT}
        m = _fd_geom($f(x.val))
        retres = EnzymeRules.needs_primal(config) ? m : nothing
        dres = EnzymeRules.needs_shadow(config) ? _zero_geom_like(m) : nothing
        return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, (x.val, dres))
    end

    @eval function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig,
        ::Const{typeof($f)},
        ::Type{RT},
        tape,
        x::Active,
    ) where {RT}
        rval, dys = tape
        dys === nothing && return (zero(rval),)
        dr::Float64 = _geom_vjp(dys, _fd_perturb($f, rval, $hh))
        _zero_geom_shadow!(dys)
        return (dr,)
    end

    @eval function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig,
        ::Const{typeof($f)},
        ::Type{RT},
        ::Any,
        ::Const,
    ) where {RT}
        return (nothing,)
    end

    @eval function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        ::Const{typeof($f)},
        ::Type{RT},
        p::Annotation{<:AbstractVector},
    ) where {RT}
        m = _fd_geom($f(p.val))
        retres = EnzymeRules.needs_primal(config) ? m : nothing
        dres = EnzymeRules.needs_shadow(config) ? _zero_geom_like(m) : nothing
        cache_p = EnzymeRules.overwritten(config)[2] ? copy(p.val) : p.val
        return EnzymeRules.augmented_rule_return_type(config, RT)(retres, dres, (cache_p, dres))
    end

    @eval function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig,
        ::Const{typeof($f)},
        ::Type{RT},
        tape,
        p::Annotation{<:AbstractVector},
    ) where {RT}
        pval, dys = tape
        if !(p isa Const) && dys !== nothing
            dp = p.dval
            for i in eachindex(pval)
                dp[i] += _geom_vjp(dys, _fd_perturb_index($f, pval, i, $hh))
            end
            _zero_geom_shadow!(dys)
        end
        return (nothing,)
    end

    _add_dual_methods!(f, hh)
    return f
end

"""
    fd_mesh_function(builder; h=1e-5)

Wrap `builder(x)` (returns `Mesh` or a Capytaine mesh) in a new function that
already has Zygote and Enzyme finite-difference mesh rules.
"""
function fd_mesh_function(builder; h::Real=_DEFAULT_H)
    return FDMesher(x -> _as_mesh(builder(x)), float(h))
end

"""
    make_sphere_mesher(; resolution=6, center=(0.0, 0.0, 0.0), h=1e-5)

Return `mesher(radius::Real) -> Mesh` for an immersed Capytaine sphere, with
finite-difference reverse rules already installed. Same role as the paper's
`differentiableMesh`, but the resolution is captured in the closure so each
geometry gets its own rules.

    hull = make_sphere_mesher(resolution=6)
    A(r) = calculate_radiation_forces(hull(r), dof, ω)[1]
"""
function make_sphere_mesher(; resolution::Integer=6, center=(0.0, 0.0, 0.0), h::Real=_DEFAULT_H)
    res = (Int(resolution), Int(resolution))
    c0 = (float(center[1]), float(center[2]), float(center[3]))
    function mesher(radius::Real)
        cpt = pyimport("capytaine")
        raw = cpt.mesh_sphere(
            name="sphere", radius=float(radius), center=c0, resolution=res,
        ).immersed_part()
        return _as_mesh(raw)
    end
    return FDMesher(mesher, float(h))
end

# Default sphere (resolution 6), rules installed at load — no Python until called.
const sphere_mesh = make_sphere_mesher()

end # module GeometryAD
