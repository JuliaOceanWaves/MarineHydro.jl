# Vectorized constant-source Rankine panel from Birk (2021):
# "A Comprehensive and Practical Guide to the Hess and Smith Constant Source and Dipole Panel"
# https://doi.org/10.1080/09377255.2021.1966575
#
# Birk defines the Rankine source as φ = -1/(4πr) and evaluates potential/velocity
# in a panel-local frame (sections 2–4), then transforms vectors back with T.
# MarineHydro / Capytaine use G = 1/r; matrix assembly later multiplies by -1/(4π).
# The conversion used by Rankine is therefore
#   ∫ G dS      = -4π φ_Birk
#   ∇_{x} ∫ G dS = -4π v_Birk
# where x is the field point (element_1).

"""
    Rankine()

Default Rankine source: Birk (2021) vectorized constant-source panel.

Point-source `greens` / `gradient_greens` are the same `1/r` kernel as
`DelhommeauRankine`. Panel integrals use the Birk local-frame formulas.
"""
struct Rankine <: GreensFunction end
wavenumber_independent(::Rankine) = true

const _BIRK_ZTOL = 1e-14
const _BIRK_DKTOL = 1e-14

# `abs(x)` and `sign(x)` zero Dual partials at x=0 (`sign(0)=0`). `ifelse` on the
# primal keeps the Dual of `x`, which is the equivalent one-sided derivative of |x|
# and matches signed-z AD through the coplanar (GZ=0) Rankine solid-angle term.
@inline _abs_ad(x) = ifelse(x < zero(x), -x, x)
@inline _sign_ad(x) = ifelse(x < zero(x), -one(x), one(x))

@inline _primal_val(x) = x
@inline _primal_val(x::ForwardDiff.Dual) = ForwardDiff.value(x)

# Force primal 0 (coplanar solid angle) without discarding Dual/tangent information.
@inline _zero_primal_keep_tangent(x) = x - oftype(x, _primal_val(x))

# atan(num/den) in (-π/2, π/2) without dividing. Division by dx=0 on vertical
# edges is IEEE-finite in primal (atan(±Inf)) but Dual/Zygote NaN.
# Use `_abs_ad` rather than `abs` so Duals at den=0 (z=0 or vertical edges) keep ∂den.
@inline function _atan_ratio(num, den)
    flipped = ifelse(den < zero(den), -num, num)
    return atan(flipped, _abs_ad(den))
end

# Birk uses signed z: φ_Ω = z Δatan, vz = Δatan. At z=0 the solid angle is 0;
# keep Δatan's tangent so AD sees ∂(z atan(c/z))/∂geometry, not the `ifelse` branch.
@inline function _inplane_atan(z, Δatan)
    if abs(_primal_val(z)) <= _BIRK_ZTOL
        return _zero_primal_keep_tangent(Δatan)
    end
    return Δatan
end

# Delhommeau near-field: same coplanar primal (AT=0 when |GZ| is tiny) with Duals
# kept from atan. |GZ| / sign(GZ) use `_abs_ad` / `_sign_ad` (equivalent signed GZ).
@inline function _delhommeau_atan(ANT, DNT, GZ, source_radius)
    AT = atan(ANT, DNT)
    if abs(_primal_val(GZ)) >= 1e-4 * _primal_val(source_radius)
        return AT
    end
    return _zero_primal_keep_tangent(AT)
end

@inline function _corner_svector(vertices, k)
    T = eltype(vertices)
    return SVector{3,T}(vertices[k, 1], vertices[k, 2], vertices[k, 3])
end

@inline function _corners(vertices)
    return ntuple(k -> _corner_svector(vertices, k), 4)
end

@inline function _as_center_type(center, v)
    return zero(center) .+ (v[1], v[2], v[3])
end

# Section 2.1: local panel frame from bimedians of the four corners.
function compute_local_system(corners)
    midpoints = ntuple(i -> (corners[i] + corners[mod1(i + 1, 4)]) / 2, 4)
    mgc = (midpoints[2] + midpoints[4]) / 2
    s = normalize(midpoints[2] - midpoints[4])
    t_temp = midpoints[3] - midpoints[1]
    n = normalize(cross(s, t_temp))
    t = normalize(cross(n, s))
    T = hcat(s, t, n)
    return T, mgc
end

# Section 2.2: temporary local corner coordinates with origin at the vertex centroid.
function compute_temporary_panel_corners(corners, mgc, T)
    s = T[:, 1]
    t = T[:, 2]
    Tcoord = eltype(eltype(corners))
    return ntuple(k -> SVector{3,Tcoord}(dot(corners[k] - mgc, s), dot(corners[k] - mgc, t), zero(Tcoord)), 4)
end

# Section 2.3: area and local centroid relative to the temporary origin.
function compute_area_and_centroid(temp_corners)
    T = eltype(eltype(temp_corners))
    terms = ntuple(4) do k
        xk, yk = temp_corners[k][1], temp_corners[k][2]
        xk1, yk1 = temp_corners[mod1(k + 1, 4)][1], temp_corners[mod1(k + 1, 4)][2]
        A_k = (yk1 - yk) * (xk + xk1) / 2
        Mx_k = (xk1 - xk) * (yk^2 + yk * yk1 + yk1^2) / 6
        My_k = (yk1 - yk) * (xk^2 + xk * xk1 + xk1^2) / 6
        (A_k, Mx_k, My_k)
    end
    A = sum(t -> t[1], terms)
    Mx = sum(t -> t[2], terms)
    My = sum(t -> t[3], terms)
    centroid_local = SVector{3,T}(My / A, -Mx / A, zero(T))
    return A, centroid_local
end

# Section 2.3, eq. (29): area centroid in global coordinates.
function compute_qgc(centroid_local, T, mgc)
    s = T[:, 1]
    t = T[:, 2]
    return mgc .+ centroid_local[1] .* s .+ centroid_local[2] .* t
end

# Section 2.4: local corners referred to the area centroid.
function compute_local_corners(temp_corners, centroid_local)
    return ntuple(k -> temp_corners[k] - centroid_local, 4)
end

function birk_panel_geometry(vertices)
    corners = _corners(vertices)
    T, mgc = compute_local_system(corners)
    temp_corners = compute_temporary_panel_corners(corners, mgc, T)
    area, centroid_local = compute_area_and_centroid(temp_corners)
    qgc = compute_qgc(centroid_local, T, mgc)
    local_corners = compute_local_corners(temp_corners, centroid_local)
    return T, qgc, local_corners, area
end

# Eq. (46). Uses atan (not atan2), as specified in Birk, including the paper's -1/(4π).
function velocity_potential(x, y, z, local_corners)
    contribs = ntuple(4) do k
        xk, yk = local_corners[k][1], local_corners[k][2]
        xk1, yk1 = local_corners[mod1(k + 1, 4)][1], local_corners[mod1(k + 1, 4)][2]
        dx = xk1 - xk
        dy = yk1 - yk
        dk = hypot(dx, dy)
        rk = sqrt((x - xk)^2 + (y - yk)^2 + z^2)
        rk1 = sqrt((x - xk1)^2 + (y - yk1)^2 + z^2)
        ek = (x - xk)^2 + z^2
        ek1 = (x - xk1)^2 + z^2
        hk = (x - xk) * (y - yk)
        hk1 = (x - xk1) * (y - yk1)
        log_term = log((rk + rk1 - dk) / (rk + rk1 + dk))
        log_part = ((x - xk) * dy - (y - yk) * dx) / dk * log_term
        # Rewrite (mk*ek - hk)/(z*rk) as (dy*ek - dx*hk)/(dx*z*rk) to allow vertical edges.
        # Signed z * Δatan; `_inplane_atan` zeros the primal at z=0 without killing Duals.
        Δatan = _inplane_atan(z,
            _atan_ratio(dy * ek - dx * hk, dx * z * rk) -
            _atan_ratio(dy * ek1 - dx * hk1, dx * z * rk1))
        atan_part = z * Δatan
        ifelse(dk <= _BIRK_DKTOL, zero(x), -(log_part + atan_part) / 4π)
    end
    return sum(contribs)
end

# Eqs. (47)–(49), including the paper's -1/(4π). Local components are transformed
# back to global coordinates by the caller with v_g = T * v_local.
function velocity_derivatives(x, y, z, local_corners)
    contribs = ntuple(4) do k
        xk, yk = local_corners[k][1], local_corners[k][2]
        xk1, yk1 = local_corners[mod1(k + 1, 4)][1], local_corners[mod1(k + 1, 4)][2]
        dx = xk1 - xk
        dy = yk1 - yk
        dk = hypot(dx, dy)
        rk = sqrt((x - xk)^2 + (y - yk)^2 + z^2)
        rk1 = sqrt((x - xk1)^2 + (y - yk1)^2 + z^2)
        ek = (x - xk)^2 + z^2
        ek1 = (x - xk1)^2 + z^2
        hk = (x - xk) * (y - yk)
        hk1 = (x - xk1) * (y - yk1)
        log_term = log((rk + rk1 - dk) / (rk + rk1 + dk))
        vx = (dy / dk) * log_term
        vy = (-dx / dk) * log_term
        vz = _inplane_atan(z,
            _atan_ratio(dy * ek - dx * hk, dx * z * rk) -
            _atan_ratio(dy * ek1 - dx * hk1, dx * z * rk1))
        ifelse(dk <= _BIRK_DKTOL, zero(SVector(x, x, x)), SVector(vx, vy, vz))
    end
    return (-1 / 4π) .* sum(contribs)
end

# Eqs. (50)–(58): Hessian of the Birk source potential in the panel-local frame.
# Only six unique entries are needed; the off-diagonals are computed independently
# so Laplace/symmetry checks can catch implementation errors. Global-frame
# Hessians are recovered with Hg = T * H * T' (eq. 37).
function velocity_hessian(x, y, z, local_corners)
    z0 = zero(x + local_corners[1][1])
    zeroH = SMatrix{3,3}(z0, z0, z0, z0, z0, z0, z0, z0, z0)
    contribs = ntuple(4) do k
        xk, yk = local_corners[k][1], local_corners[k][2]
        xk1, yk1 = local_corners[mod1(k + 1, 4)][1], local_corners[mod1(k + 1, 4)][2]
        dx = xk1 - xk
        dy = yk1 - yk
        dk = hypot(dx, dy)
        if dk <= _BIRK_DKTOL
            return zeroH
        end
        dxk = x - xk
        dyk = y - yk
        dxk1 = x - xk1
        dyk1 = y - yk1
        rk = sqrt(dxk^2 + dyk^2 + z^2)
        rk1 = sqrt(dxk1^2 + dyk1^2 + z^2)
        rsum = rk + rk1
        den = rsum^2 - dk^2
        ρk = rk * rk1 + dxk * dxk1 + dyk * dyk1 + z^2
        λk = dxk * dyk1 - dxk1 * dyk
        xterm = dxk / rk + dxk1 / rk1
        yterm = dyk / rk + dyk1 / rk1
        zterm = z / rk + z / rk1
        ρden = (rk * rk1) * ρk
        hxx = 2 * dy / den * xterm
        hyx = -2 * dx / den * xterm
        hxy = 2 * dy / den * yterm
        hyy = -2 * dx / den * yterm
        hxz = 2 * dy / den * zterm
        hyz = -2 * dx / den * zterm
        hzx = z * dy * rsum / ρden
        hzy = -z * dx * rsum / ρden
        hzz = λk * rsum / ρden
        # Column-major SMatrix: columns are (∂/∂x, ∂/∂y, ∂/∂z) of (φx, φy, φz).
        SMatrix{3,3}(hxx, hyx, hzx, hxy, hyy, hzy, hxz, hyz, hzz)
    end
    return (-1 / 4π) .* sum(contribs)
end

# Fused eq. (46) + eqs. (47)–(49): one edge loop returning (φ, v). Same values
# as `velocity_potential` / `velocity_derivatives` (φ's solid angle is z·vz per
# edge), but the per-corner radii, per-edge log and the Δatan pair are computed
# once instead of twice. This is the near-field hot path of `assemble_birk_rankine`.
function birk_phi_and_velocity(x, y, z, local_corners)
    sx = ntuple(k -> x - local_corners[k][1], 4)
    sy = ntuple(k -> y - local_corners[k][2], 4)
    r = ntuple(k -> sqrt(sx[k]^2 + sy[k]^2 + z^2), 4)
    e = ntuple(k -> sx[k]^2 + z^2, 4)
    h = ntuple(k -> sx[k] * sy[k], 4)
    contribs = ntuple(4) do k
        kn = mod1(k + 1, 4)
        dx = local_corners[kn][1] - local_corners[k][1]
        dy = local_corners[kn][2] - local_corners[k][2]
        dk = hypot(dx, dy)
        rsum = r[k] + r[kn]
        log_term = log((rsum - dk) / (rsum + dk))
        lt_dk = log_term / dk
        Δatan = _atan_ratio(dy * e[k] - dx * h[k], dx * z * r[k]) -
                _atan_ratio(dy * e[kn] - dx * h[kn], dx * z * r[kn])
        vzk = _inplane_atan(z, Δatan)
        φk = (sx[k] * dy - sy[k] * dx) * lt_dk + z * vzk
        zc = zero(φk)
        ifelse(dk <= _BIRK_DKTOL,
            SVector(zc, zc, zc, zc),
            SVector(φk, dy * lt_dk, -dx * lt_dk, vzk))
    end
    s = sum(contribs)
    scale = -1 / 4π
    return scale * s[1], SVector(scale * s[2], scale * s[3], scale * s[4])
end

# Reverse rules (Zygote ChainRules + EnzymeRules, including ForwardDiff
# corner VJPs) live in `ReverseAD` (`src/ad_rules.jl`).

# Source-panel frame is independent of the field point: assemble once per column.
@inline function _vrankine_from_geom(point, source_point, source_radius, source_area, Tmat, qgc, local_corners)
    r̄ = point - source_point
    r = norm(r̄)
    if r > 7 * source_radius
        return source_area / r, (r̄ / r^3) * source_area
    end
    PT = eltype(point)
    QT = eltype(qgc)
    field_local = Tmat' * (SVector{3,PT}(point[1], point[2], point[3]) - SVector{3,QT}(qgc[1], qgc[2], qgc[3]))
    φ_birk, v_local = birk_phi_and_velocity(field_local[1], field_local[2], field_local[3], local_corners)
    v_global = Tmat * v_local
    φ = -4π * φ_birk
    grad_wrt_source = 4π .* v_global
    return φ, grad_wrt_source
end

function _vrankine_integral_and_gradient(element_1, element_2)
    Tmat, qgc, local_corners, _area = birk_panel_geometry(vertices(element_2))
    return _vrankine_from_geom(center(element_1), center(element_2), radius(element_2),
        area(element_2), Tmat, qgc, local_corners)
end

function greens(::Rankine, element_1, element_2, wavenumber=nothing)
    return greens(DelhommeauRankine(), element_1, element_2, wavenumber)
end

function gradient_greens(::Rankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    return gradient_greens(DelhommeauRankine(), element_1, element_2, wavenumber; with_respect_to_first_variable)
end

function integral(::Rankine, element_1, element_2, wavenumber=nothing)
    φ, _ = _vrankine_integral_and_gradient(element_1, element_2)
    return φ
end

function integral_gradient(::Rankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    _, grad_wrt_source = _vrankine_integral_and_gradient(element_1, element_2)
    point = center(element_1)
    if with_respect_to_first_variable
        return _as_center_type(point, -grad_wrt_source)
    else
        return _as_center_type(point, grad_wrt_source)
    end
end

function both_integral_and_integral_gradient(::Rankine, element_1, element_2, wavenumber=nothing; with_respect_to_first_variable=false)
    φ, grad_wrt_source = _vrankine_integral_and_gradient(element_1, element_2)
    point = center(element_1)
    if with_respect_to_first_variable
        return φ, _as_center_type(point, -grad_wrt_source)
    else
        return φ, _as_center_type(point, grad_wrt_source)
    end
end
