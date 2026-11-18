
using LinearAlgebra
using Unitful: ustrip
using Statistics: mean

_coord(p) = Float64[
    ustrip(p.coords.x),
    ustrip(p.coords.y),
    ustrip(p.coords.z)
]

# Generalized hydrostatic DOF modes
abstract type AbstractHydroDofMode end

struct FunctionalDofMode <: AbstractHydroDofMode
    motion::Function
    gradient::Union{Function,Nothing}
    divergence::Union{Function,Nothing}
end

function FunctionalDofMode(motion; gradient=nothing, divergence=nothing)
    return FunctionalDofMode(motion, gradient, divergence)
end

struct PanelDofMode <: AbstractHydroDofMode
    motion::Matrix{Float64}      # nfaces × 3
    divergence::Any              # scalar or nfaces vector
end

function PanelDofMode(motion; divergence=0.0)
    return PanelDofMode(Matrix{Float64}(motion), divergence)
end

function evaluate_motion(mode::FunctionalDofMode, q, iface)
    return Float64.(mode.motion(q))
end

function evaluate_divergence(mode::FunctionalDofMode, q, iface)
    if mode.divergence !== nothing
        return mode.divergence(q)
    elseif mode.gradient !== nothing
        return tr(mode.gradient(q))
    else
        error("FunctionalDofMode needs either gradient or divergence.")
    end
end

function evaluate_motion(mode::PanelDofMode, q, iface)
    return mode.motion[iface, :]
end

function evaluate_divergence(mode::PanelDofMode, q, iface)
    if mode.divergence isa Number
        return mode.divergence
    else
        return mode.divergence[iface]
    end
end

# Clipping
function _clip_polygon_z(poly, zcut; keep_below=true, atol=1e-10)
    isempty(poly) && return poly

    inside(p) = keep_below ? p[3] <= zcut + atol : p[3] >= zcut - atol

    clipped = Vector{Vector{Float64}}()

    prev = poly[end]
    prev_inside = inside(prev)

    for curr in poly
        curr_inside = inside(curr)

        if curr_inside != prev_inside
            t = (zcut - prev[3]) / (curr[3] - prev[3])
            p_int = prev + t * (curr - prev)
            push!(clipped, p_int)
        end

        if curr_inside
            push!(clipped, curr)
        end

        prev = curr
        prev_inside = curr_inside
    end

    return clipped
end

function _triangulate_polygon(poly)
    tris = NTuple{3,Vector{Float64}}[]

    for i in 2:(length(poly)-1)
        push!(tris, (poly[1], poly[i], poly[i+1]))
    end

    return tris
end

function _tris_to_meshes_mesh(tris)
    points = [
        Meshes.Point(p[1], p[2], p[3])
        for tri in tris
        for p in tri
    ]

    connec = [
        Meshes.connect((3i-2, 3i-1, 3i), Meshes.Triangle)
        for i in 1:length(tris)
    ]

    return Meshes.SimpleMesh(points, connec)
end

const _TRI_QUAD3 = (
    (1/6, 1/6, 2/3, 1/3),
    (1/6, 2/3, 1/6, 1/3),
    (2/3, 1/6, 1/6, 1/3),
)

function _integrate_triangle(f, a, b, c)
    acc = zero(f((a + b + c) / 3))

    for (la, lb, lc, w) in _TRI_QUAD3
        q = la .* a .+ lb .* b .+ lc .* c
        acc += w * f(q)
    end

    return acc
end

function _clip_dof_to_immersed(dof, origin_ids)
    return dof[origin_ids, :]
end

function _clip_dof_to_immersed(dof::PanelDofMode, origin_ids)
    new_motion = dof.motion[origin_ids, :]

    new_divergence =
        dof.divergence isa Number ? dof.divergence : dof.divergence[origin_ids]

    return PanelDofMode(new_motion; divergence=new_divergence)
end

function _clip_dof_to_immersed(dof::FunctionalDofMode, origin_ids)
    return dof
end

function get_immersed_mesh(mesh; free_surface=0.0, water_depth=Inf, atol=1e-10, return_origin_ids=false)

    tris = NTuple{3,Vector{Float64}}[]
    origin_ids = Int[]

    ztop = free_surface
    zbot = free_surface - water_depth

    for (face_id, elem) in enumerate(Meshes.elements(mesh))
        poly = _coord.(Meshes.vertices(elem))

        # keep below free surface
        poly = _clip_polygon_z(poly, ztop; keep_below=true, atol=atol)

        # optionally keep above sea bottom
        if isfinite(water_depth)
            poly = _clip_polygon_z(poly, zbot; keep_below=false, atol=atol)
        end

        length(poly) < 3 && continue

        # drop artificial/free-surface cap if it lies exactly on waterplane
        if all(abs(p[3] - free_surface) < atol for p in poly)
            continue
        end

        new_tris = _triangulate_polygon(poly)

        append!(tris, new_tris)
        append!(origin_ids, fill(face_id, length(new_tris)))
    end

    immersed_mesh = _tris_to_meshes_mesh(tris)

    if return_origin_ids
        return immersed_mesh, origin_ids
    else
        return immersed_mesh
    end
end

function _is_immersed_mesh(mesh; free_surface=0.0, water_depth=Inf, atol=1e-10)
    for elem in Meshes.elements(mesh)
        for p in Meshes.vertices(elem)
            z = ustrip(p.coords.z)

            if z > free_surface + atol
                return false
            end

            if isfinite(water_depth)
                zbot = free_surface - water_depth
                if z < zbot - atol
                    return false
                end
            end
        end
    end

    return true
end

function get_immersed_part(body::FloatingBody; free_surface=0.0, water_depth=Inf, atol=1e-10)

    if _is_immersed_mesh(body.mesh; free_surface=free_surface, water_depth=water_depth, atol=atol)
        return body
    end

    immersed_mesh, origin_ids = get_immersed_mesh(
        body.mesh;
        free_surface=free_surface,
        water_depth=water_depth,
        atol=atol,
        return_origin_ids=true
    )

    # Update DOFs so they match the new clipped/triangulated mesh panels
    new_dofs = NamedTuple{keys(body.dofs)}(
        map(dof -> _clip_dof_to_immersed(dof, origin_ids), values(body.dofs))
    )

    new_body = FloatingBody(
        immersed_mesh,
        body.lid_mesh,
        new_dofs,
        body.body_name,
        body.center_of_mass,
        body.rotation_center,
        body.mass
    )

    return new_body
end

# Mass, volume, centers, etc.
function _tri_area_vector(a, b, c)
    0.5 * cross(b - a, c - a)
end

function _tri_avg_square(a, b, c, k)
    ak, bk, ck = a[k], b[k], c[k]
    return (ak^2 + bk^2 + ck^2 + ak*bk + bk*ck + ck*ak) / 6
end

function _volume_components(mesh)
    V = zeros(3)

    for elem in Meshes.elements(mesh)
        pts = _coord.(Meshes.vertices(elem))

        # split polygons/quads into triangles from vertex 1
        for i in 2:(length(pts)-1)
            a, b, c = pts[1], pts[i], pts[i+1]
            avec = _tri_area_vector(a, b, c)   # n dS vector
            centroid = (a + b + c) / 3

            V .+= centroid .* avec
        end
    end

    return V
end

function get_volume(mesh::Meshes.Mesh; rtol=1e-6)
    V_components = _volume_components(mesh)
    V = mean(V_components)

    if maximum(abs.(V_components .- V)) > rtol * max(abs(V), eps())
        @warn "Volume components disagree. Mesh may be open, non-watertight, or inconsistently oriented." V_components V
    end

    return V
end

function get_volume(body::FloatingBody)
    return get_volume(body.mesh)
end

function get_displaced_volume(mesh::Meshes.Mesh; free_surface=0.0, water_depth=Inf)
    immersed_mesh = get_immersed_mesh(mesh; free_surface=free_surface, water_depth=water_depth)
    return get_volume(immersed_mesh)
end

function get_displaced_volume(body::FloatingBody; free_surface=0.0, water_depth=Inf)
    return get_displaced_volume(body.mesh; free_surface=free_surface, water_depth=water_depth)
end

function get_mass(mesh::Meshes.Mesh; rho=1025.0)
    volume = get_volume(mesh)
    return rho * volume
end

function get_mass(body::FloatingBody; rho=1025.0)
    return get_mass(body.mesh; rho=rho)
end

function get_displaced_mass(mesh::Meshes.Mesh; rho=1025.0, free_surface=0.0, water_depth=Inf)
    immersed_mesh = get_immersed_mesh(mesh; free_surface=free_surface, water_depth=water_depth)
    return get_mass(immersed_mesh; rho=rho)
end

function get_displaced_mass(body::FloatingBody; rho=1025.0, free_surface=0.0, water_depth=Inf)
    return get_displaced_mass(body.mesh; rho=rho, free_surface=free_surface, water_depth=water_depth)
end

function get_surface_area(mesh::Meshes.Mesh)

    SA = 0.0

    for elem in Meshes.elements(mesh)
        pts = _coord.(Meshes.vertices(elem))

        # split polygons/quads into triangles from vertex 1
        for i in 2:(length(pts)-1)
            a, b, c = pts[1], pts[i], pts[i+1]
            avec = _tri_area_vector(a, b, c)   # n dS vector

            SA += norm(avec)
        end
    end

    return SA
end

function get_surface_area(body::FloatingBody)
    return get_surface_area(body.mesh)
end

function get_wet_surface_area(mesh::Meshes.Mesh; free_surface=0.0, water_depth=Inf)
    immersed_mesh = get_immersed_mesh(mesh; free_surface=free_surface, water_depth=water_depth)
    return get_surface_area(immersed_mesh)
end

function get_wet_surface_area(body::FloatingBody; free_surface=0.0, water_depth=Inf)
    return get_wet_surface_area(body.mesh; free_surface=free_surface, water_depth=water_depth)
end

function get_geometric_center(mesh::Meshes.Mesh)
    Vcomps = zeros(3)
    M = zeros(3)

    for elem in Meshes.elements(mesh)
        pts = _coord.(Meshes.vertices(elem))

        # fan triangulation
        for i in 2:(length(pts)-1)
            a, b, c = pts[1], pts[i], pts[i+1]
            avec = _tri_area_vector(a, b, c)  # n dS vector
            centroid = (a + b + c) / 3

            # volume components: ∫ xᵢ nᵢ dS
            Vcomps .+= centroid .* avec

            # first moments: 1/2 ∫ xᵢ² nᵢ dS
            for k in 1:3
                M[k] += 0.5 * _tri_avg_square(a, b, c, k) * avec[k]
            end
        end
    end

    V = mean(Vcomps)
    return M / V
end

function get_geometric_center(body::FloatingBody)
    return get_geometric_center(body.mesh)
end

function get_buoyancy_center(mesh::Meshes.Mesh; free_surface=0.0, water_depth=Inf)
    immersed_mesh = get_immersed_mesh(mesh; free_surface=free_surface, water_depth=water_depth)
    return get_geometric_center(immersed_mesh)
end

function get_buoyancy_center(body::FloatingBody; free_surface=0.0, water_depth=Inf)
    return get_buoyancy_center(body.mesh; free_surface=free_surface, water_depth=water_depth)
end

# hydrostatic stiffness functions
function _waterplane_integral_centroid(mesh::Meshes.Mesh, f)
    acc = 0.0

    for elem in Meshes.elements(mesh)
        pts = _coord.(Meshes.vertices(elem))

        for i in 2:(length(pts)-1)
            a, b, c = pts[1], pts[i], pts[i+1]
            area_vec = _tri_area_vector(a, b, c)
            q = (a + b + c) / 3

            acc += -area_vec[3] * f(q)
        end
    end

    return acc
end

function _waterplane_integral_tri3(mesh::Meshes.Mesh, f)
    acc = 0.0

    for elem in Meshes.elements(mesh)
        pts = _coord.(Meshes.vertices(elem))

        for i in 2:(length(pts)-1)
            a, b, c = pts[1], pts[i], pts[i+1]
            area_vec = _tri_area_vector(a, b, c)

            val = _integrate_triangle(a, b, c) do q
                -area_vec[3] * f(q)
            end

            acc += val
        end
    end

    return acc
end

function _waterplane_integral(mesh::Meshes.Mesh, f; quadrature=:tri3)
    if quadrature == :tri3
        return _waterplane_integral_tri3(mesh, f)
    elseif quadrature == :centroid
        return _waterplane_integral_centroid(mesh, f)
    else
        error("Unknown quadrature method: $quadrature")
    end
end

function get_waterplane_area(mesh::Meshes.Mesh; quadrature=:tri3)
    return _waterplane_integral(mesh, q -> 1.0; quadrature=quadrature)
end

function get_waterplane_area(body::FloatingBody; quadrature=:tri3)
    return get_waterplane_area(body.mesh; quadrature=quadrature)
end

function get_waterplane_center(mesh::Meshes.Mesh; quadrature=:tri3)
    waterplane_area = get_waterplane_area(mesh; quadrature=quadrature)
    if abs(waterplane_area) < 1e-10
        return nothing
    else
        x_center = _waterplane_integral(mesh, q -> q[1]; quadrature=quadrature) / waterplane_area
        y_center = _waterplane_integral(mesh, q -> q[2]; quadrature=quadrature) / waterplane_area
    end
    return (x_center, y_center)
end

function get_waterplane_center(body::FloatingBody; quadrature=:tri3)
    return get_waterplane_center(body.mesh; quadrature=quadrature)
end

function get_transversal_metacentric_radius(body::FloatingBody; free_surface=0.0, water_depth=Inf, quadrature=:tri3)
    immersed_body = get_immersed_part(body; free_surface=free_surface, water_depth=water_depth)

    Iwp = _waterplane_integral(
        immersed_body.mesh,
        q -> q[2]^2;
        quadrature=quadrature
    )

    return Iwp / get_volume(immersed_body)
end

function get_longitudinal_metacentric_radius(body::FloatingBody; free_surface=0.0, water_depth=Inf, quadrature=:tri3)
    immersed_body = get_immersed_part(body; free_surface=free_surface, water_depth=water_depth)

    Iwp = _waterplane_integral(
        immersed_body.mesh,
        q -> q[1]^2;
        quadrature=quadrature
    )

    return Iwp / get_volume(immersed_body)
end

function get_transversal_metacentric_height(body::FloatingBody; free_surface=0.0, water_depth=Inf, quadrature=:tri3)
    body.center_of_mass === nothing &&
        error("center_of_mass is required for metacentric height.")

    immersed_body = get_immersed_part(body; free_surface=free_surface, water_depth=water_depth)

    G = body.center_of_mass
    B = get_geometric_center(immersed_body.mesh)

    gb_z = G[3] - B[3]

    return get_transversal_metacentric_radius(body; free_surface=free_surface, water_depth=water_depth, quadrature=quadrature) - gb_z
end

function get_longitudinal_metacentric_height(body::FloatingBody; free_surface=0.0, water_depth=Inf, quadrature=:tri3)
    body.center_of_mass === nothing &&
        error("center_of_mass is required for metacentric height.")

    immersed_body = get_immersed_part(body; free_surface=free_surface, water_depth=water_depth)

    G = body.center_of_mass
    B = get_geometric_center(immersed_body.mesh)

    gb_z = G[3] - B[3]

    return get_longitudinal_metacentric_radius(body; free_surface=free_surface, water_depth=water_depth, quadrature=quadrature) - gb_z
end

const _RIGID_DOFS = Set([:Surge, :Sway, :Heave, :Roll, :Pitch, :Yaw])

_is_rigid_dof(name::Symbol, dof) =
    name in _RIGID_DOFS && !(dof isa AbstractHydroDofMode)

function _as_hydro_mode(dof)
    if dof isa AbstractHydroDofMode
        return dof
    else
        return PanelDofMode(dof; divergence=0.0)
    end
end

function _generalized_hydrostatic_stiffness_coef(
    immersed_mesh::Meshes.Mesh,
    influenced_dof,
    radiating_dof;
    rho=1025.0,
    g=9.81
)
    influenced_mode = _as_hydro_mode(influenced_dof)
    radiating_mode = _as_hydro_mode(radiating_dof)

    norm_hs_stiff = 0.0

    for (iface, elem) in enumerate(Meshes.elements(immersed_mesh))
        pts = _coord.(Meshes.vertices(elem))

        for i in 2:(length(pts)-1)
            a, b, c = pts[1], pts[i], pts[i+1]
            area_vec = _tri_area_vector(a, b, c)

            val = _integrate_triangle(a, b, c) do q
                z = q[3]

                Vi = evaluate_motion(influenced_mode, q, iface)
                Vj = evaluate_motion(radiating_mode, q, iface)
                Di = evaluate_divergence(influenced_mode, q, iface)

                -dot(area_vec, Vj) * (Vi[3] + z * Di)
            end

            norm_hs_stiff += val
        end
    end

    return rho * g * norm_hs_stiff
end

function _hydrostatic_stiffness_coef(
    immersed_mesh,
    influenced_dof::Symbol,
    radiating_dof::Symbol;
    rho=1025.0,
    g=9.81,
    mass=nothing,
    center_of_mass=(0.0, 0.0, 0.0),
    rotation_center=(0.0, 0.0, 0.0),
    quadrature=:tri3
)
    pair = (influenced_dof, radiating_dof)

    center_of_mass === nothing &&
        error("center_of_mass is required for rigid-body hydrostatic stiffness.")

    xc, yc, zc = rotation_center
    xg, yg, zg = center_of_mass

    V = get_volume(immersed_mesh)
    xb, yb, zb = get_geometric_center(immersed_mesh)

    m = mass === nothing ? rho * V : mass

    if pair == (:Heave, :Heave)
        norm_hs_stiff = get_waterplane_area(immersed_mesh; quadrature=quadrature)

    elseif pair in ((:Heave, :Roll), (:Roll, :Heave))
        norm_hs_stiff = _waterplane_integral(immersed_mesh, q -> q[2] - yc; quadrature=quadrature)

    elseif pair in ((:Heave, :Pitch), (:Pitch, :Heave))
        norm_hs_stiff = -_waterplane_integral(immersed_mesh, q -> q[1] - xc; quadrature=quadrature)

    elseif pair == (:Roll, :Roll)
        norm_hs_stiff =
            _waterplane_integral(immersed_mesh, q -> (q[2] - yc)^2; quadrature=quadrature) +
                V * (zb - zc) -
                (m / rho) * (zg - zc)

    elseif pair in ((:Roll, :Pitch), (:Pitch, :Roll))
        norm_hs_stiff =
            -_waterplane_integral(immersed_mesh, q -> (q[1] - xc) * (q[2] - yc); quadrature=quadrature)

    elseif pair == (:Roll, :Yaw)
        norm_hs_stiff =
            -V * (xb - xc) +
                (m / rho) * (xg - xc)

    elseif pair == (:Pitch, :Pitch)
        norm_hs_stiff =
            _waterplane_integral(immersed_mesh, q -> (q[1] - xc)^2; quadrature=quadrature) +
                V * (zb - zc) -
                (m / rho) * (zg - zc)

    elseif pair == (:Pitch, :Yaw)
        norm_hs_stiff =
            -V * (yb - yc) +
                (m / rho) * (yg - yc)

    else
        norm_hs_stiff = 0.0
    end

    return rho * g * norm_hs_stiff
end

function get_hydrostatic_stiffness(
    body::FloatingBody;
    free_surface=0.0,
    water_depth=Inf,
    rho=1025.0,
    g=9.81,
    quadrature=:tri3
)
    immersed_body = get_immersed_part(
        body;
        free_surface=free_surface,
        water_depth=water_depth
    )

    dof_names = collect(keys(immersed_body.dofs))
    ndofs = length(dof_names)

    K = zeros(Float64, ndofs, ndofs)

    for i in 1:ndofs
        for j in 1:ndofs
            influenced_name = dof_names[i]
            radiating_name = dof_names[j]

            influenced_dof = immersed_body.dofs[influenced_name]
            radiating_dof = immersed_body.dofs[radiating_name]

            if _is_rigid_dof(influenced_name, influenced_dof) && _is_rigid_dof(radiating_name, radiating_dof)
                K[i, j] = _hydrostatic_stiffness_coef(
                    immersed_body.mesh,
                    influenced_name,
                    radiating_name;
                    rho=rho,
                    g=g,
                    center_of_mass=body.center_of_mass,
                    mass=body.mass,
                    rotation_center=body.rotation_center,
                    quadrature=quadrature
                )
            else
                if body.mass !== nothing
                    disp_mass = rho * get_volume(immersed_body.mesh)

                    if !isapprox(body.mass, disp_mass; rtol=1e-4)
                        error(
                            "Generalized hydrostatic stiffness with flexible modes " *
                                "currently assumes neutral buoyancy. " *
                                "body.mass=$(body.mass), displaced_mass=$disp_mass"
                        )
                    end
                end

                K[i, j] = _generalized_hydrostatic_stiffness_coef(
                    immersed_body.mesh,
                    influenced_dof,
                    radiating_dof;
                    rho=rho,
                    g=g
                )
            end
        end
    end

    return (; matrix=K, dofs=dof_names)
end

# inertia functions
function _element_center_and_area_vector(elem)
    pts = _coord.(Meshes.vertices(elem))

    area_vec = zeros(3)

    for i in 2:(length(pts)-1)
        a, b, c = pts[1], pts[i], pts[i+1]
        area_vec .+= _tri_area_vector(a, b, c)
    end

    center = sum(pts) / length(pts)

    return center, area_vec
end

function _rigid_body_inertia_matrix_6dof(
    mesh::Meshes.Mesh;
    rho=1025.0,
    mass=nothing,
    center_of_mass,
    rotation_center=(0.0, 0.0, 0.0)
)
    center_of_mass === nothing && error("center_of_mass is required.")

    rc = Float64.(rotation_center)
    cog = Float64.(center_of_mass) .- rc

    V = get_volume(mesh)
    density = mass === nothing ? rho : mass / V

    integrals = zeros(3, 6)

    for elem in Meshes.elements(mesh)
        fc, area_vec = _element_center_and_area_vector(elem)

        f = fc .- rc
        x, y, z = f

        combinations = (
            x^2,
            y^2,
            z^2,
            x*y,
            y*z,
            z*x
        )

        # Matches Capytaine:
        # sum(normal_i * fcs[axis] * combination * area)
        integrals[1, :] .+= area_vec[1] * x .* combinations
        integrals[2, :] .+= area_vec[2] * y .* combinations
        integrals[3, :] .+= area_vec[3] * z .* combinations
    end

    inertias = zeros(6)

    inertias[1] = (
        integrals[1, 2] + integrals[1, 3] +
            integrals[2, 2]/3 + integrals[2, 3] +
            integrals[3, 2] + integrals[3, 3]/3
    ) / (3V)

    inertias[2] = (
        integrals[1, 1]/3 + integrals[1, 3] +
            integrals[2, 1] + integrals[2, 3] +
            integrals[3, 1] + integrals[3, 3]/3
    ) / (3V)

    inertias[3] = (
        integrals[1, 1]/3 + integrals[1, 2] +
            integrals[2, 1] + integrals[2, 2]/3 +
            integrals[3, 1] + integrals[3, 2]
    ) / (3V)

    inertias[4] = integrals[3, 4] / V  # xy product
    inertias[5] = integrals[1, 5] / V  # yz product
    inertias[6] = integrals[2, 6] / V  # zx product

    xg, yg, zg = cog

    volumic_inertia_matrix = V * [
        1.0 0.0 0.0 0.0 zg -yg;
        0.0 1.0 0.0 -zg 0.0 xg;
        0.0 0.0 1.0 yg -xg 0.0;
        0.0 -zg yg inertias[1] -inertias[4] -inertias[6];
        zg 0.0 -xg -inertias[4] inertias[2] -inertias[5];
        -yg xg 0.0 -inertias[6] -inertias[5] inertias[3]
    ]

    return density * volumic_inertia_matrix
end

function get_rigid_body_inertia(
    body::FloatingBody;
    rho=1025.0
)
    K6 = _rigid_body_inertia_matrix_6dof(
        body.mesh;
        rho=rho,
        mass=body.mass,
        center_of_mass=body.center_of_mass,
        rotation_center=body.rotation_center
    )

    standard = Dict(
        :Surge => 1,
        :Sway => 2,
        :Heave => 3,
        :Roll => 4,
        :Pitch => 5,
        :Yaw => 6
    )

    dof_names = collect(keys(body.dofs))
    n = length(dof_names)

    M = fill(NaN, n, n)

    for i in 1:n
        for j in 1:n
            if haskey(standard, dof_names[i]) && haskey(standard, dof_names[j])
                M[i, j] = K6[standard[dof_names[i]], standard[dof_names[j]]]
            end
        end
    end

    return (; matrix=M, dofs=dof_names)
end

function _mesh_vertices_matrix(mesh)
    pts = Vector{Vector{Float64}}()

    for elem in Meshes.elements(mesh)
        append!(pts, _coord.(Meshes.vertices(elem)))
    end

    P = zeros(Float64, length(pts), 3)
    for i in eachindex(pts)
        P[i, :] .= pts[i]
    end

    return P
end

function _mesh_bbox(mesh)
    P = _mesh_vertices_matrix(mesh)
    mins = vec(minimum(P; dims=1))
    maxs = vec(maximum(P; dims=1))
    return mins, maxs
end

function get_length_overall(mesh)
    mins, maxs = _mesh_bbox(mesh)
    return maxs[1] - mins[1]
end

get_length_overall(body::FloatingBody) = get_length_overall(body.mesh)

function get_breadth_overall(mesh)
    mins, maxs = _mesh_bbox(mesh)
    return maxs[2] - mins[2]
end

get_breadth_overall(body::FloatingBody) = get_breadth_overall(body.mesh)

function get_depth(mesh)
    mins, maxs = _mesh_bbox(mesh)
    return maxs[3] - mins[3]
end

get_depth(body::FloatingBody) = get_depth(body.mesh)

function get_draft(mesh; free_surface=0.0)
    mins, _ = _mesh_bbox(mesh)
    return free_surface - mins[3]
end

get_draft(body::FloatingBody; free_surface=0.0) =
    get_draft(body.mesh; free_surface=free_surface)

function get_length_overall_submerged(mesh; free_surface=0.0, water_depth=Inf)
    immersed_mesh = get_immersed_mesh(mesh; free_surface=free_surface, water_depth=water_depth)
    return get_length_overall(immersed_mesh)
end

get_length_overall_submerged(body::FloatingBody; free_surface=0.0, water_depth=Inf) =
    get_length_overall_submerged(body.mesh; free_surface=free_surface, water_depth=water_depth)

function get_breadth_overall_submerged(mesh; free_surface=0.0, water_depth=Inf)
    immersed_mesh = get_immersed_mesh(mesh; free_surface=free_surface, water_depth=water_depth)
    return get_breadth_overall(immersed_mesh)
end

get_breadth_overall_submerged(body::FloatingBody; free_surface=0.0, water_depth=Inf) =
    get_breadth_overall_submerged(body.mesh; free_surface=free_surface, water_depth=water_depth)

function _waterline_points(mesh; free_surface=0.0, atol=1e-8)
    P = _mesh_vertices_matrix(mesh)
    idx = findall(abs.(P[:, 3] .- free_surface) .< atol)
    return P[idx, :]
end

function get_length_at_waterline(mesh; free_surface=0.0, atol=1e-8)
    P = _waterline_points(mesh; free_surface=free_surface, atol=atol)
    isempty(P) && return 0.0
    return maximum(P[:, 1]) - minimum(P[:, 1])
end

get_length_at_waterline(body::FloatingBody; free_surface=0.0, atol=1e-8) =
    get_length_at_waterline(body.mesh; free_surface=free_surface, atol=atol)

function get_breadth_at_waterline(mesh; free_surface=0.0, atol=1e-8)
    P = _waterline_points(mesh; free_surface=free_surface, atol=atol)
    isempty(P) && return 0.0
    return maximum(P[:, 2]) - minimum(P[:, 2])
end

get_breadth_at_waterline(body::FloatingBody; free_surface=0.0, atol=1e-8) =
    get_breadth_at_waterline(body.mesh; free_surface=free_surface, atol=atol)

function get_hydrostatics(
    body::FloatingBody;
    rho=1025.0,
    g=9.81,
    free_surface=0.0,
    water_depth=Inf,
    quadrature=:tri3
)
    # Hydrostatic formulas assume free_surface = 0.0 for best accuracy.
    @info "Computing hydrostatics on immersed part of body."

    body.center_of_mass === nothing &&
        error("center_of_mass is required to compute hydrostatics.")

    immersed_body = get_immersed_part(
        body;
        free_surface=free_surface,
        water_depth=water_depth
    )

    wp_center = get_waterplane_center(immersed_body; quadrature=quadrature)
    wp_center_3d = wp_center === nothing ? nothing : (wp_center[1], wp_center[2], free_surface)

    hs = get_hydrostatic_stiffness(
        body;
        free_surface=free_surface,
        water_depth=water_depth,
        rho=rho,
        g=g,
        quadrature=quadrature
    )

    has_rigid = any(name -> _is_rigid_dof(name, body.dofs[name]), keys(body.dofs))
    inertia = has_rigid ? get_rigid_body_inertia(body; rho=rho) : nothing

    inertia_matrix = inertia === nothing ? nothing : inertia.matrix
    inertia_matrix_dofs = inertia === nothing ? Symbol[] : inertia.dofs

    displaced_volume = get_volume(immersed_body)
    immersed_center = get_geometric_center(immersed_body.mesh)

    return (
        g=g,
        rho=rho,
        center_of_mass=body.center_of_mass,
        wet_surface_area=get_surface_area(immersed_body),
        displaced_volume=displaced_volume,
        displaced_mass=rho * displaced_volume,
        center_of_buoyancy=immersed_center,
        waterplane_area=get_waterplane_area(immersed_body; quadrature=quadrature),
        waterplane_center=wp_center_3d,
        transversal_metacentric_radius=get_transversal_metacentric_radius(immersed_body; free_surface=free_surface, water_depth=water_depth, quadrature=quadrature),
        longitudinal_metacentric_radius=get_longitudinal_metacentric_radius(immersed_body; free_surface=free_surface, water_depth=water_depth, quadrature=quadrature),
        transversal_metacentric_height=get_transversal_metacentric_height(immersed_body; free_surface=free_surface, water_depth=water_depth, quadrature=quadrature),
        longitudinal_metacentric_height=get_longitudinal_metacentric_height(immersed_body; free_surface=free_surface, water_depth=water_depth, quadrature=quadrature),
        hydrostatic_stiffness=hs.matrix,
        hydrostatic_stiffness_dofs=hs.dofs,
        inertia_matrix=inertia_matrix,
        inertia_matrix_dofs=inertia_matrix_dofs,
        length_overall=get_length_overall(body),
        breadth_overall=get_breadth_overall(body),
        depth=get_depth(body),
        draft=get_draft(body; free_surface=free_surface),
        length_at_waterline=get_length_at_waterline(
            immersed_body;
            free_surface=free_surface
        ),
        breadth_at_waterline=get_breadth_at_waterline(
            immersed_body;
            free_surface=free_surface
        ),
        length_overall_submerged=get_length_overall_submerged(
            body;
            free_surface=free_surface,
            water_depth=water_depth
        ),
        breadth_overall_submerged=get_breadth_overall_submerged(
            body;
            free_surface=free_surface,
            water_depth=water_depth
        )
    )
end

function get_hydrostatics(mb::MultiFloatingBody; kwargs...)
    vals = [
        get_hydrostatics(body; kwargs...)
        for body in mb.bodies
    ]
    return NamedTuple{Tuple(mb.body_names)}(Tuple(vals))
end