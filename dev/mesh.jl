using Meshes
using CairoMakie: CairoMakie as Makie
using ForwardDiff
using Unitful: m, ustrip, Length

function axisymmetric_mesh(profile::Rope, n_theta::Integer=72)
    @assert n_theta >= 3 "n_theta must be at least 3"
    @assert length(profile.vertices) >= 2 "profile must contain at least two points"

    # Points: Revolve each 2D profile point (r, z) into a 3D ring,
    #  treat zero radius points as special cases
    p1 = coords(first(profile.vertices))
    points = typeof(Point(p1.x, zero(p1.x), p1.y))[]
    ring_starts = Int[]
    ring_sizes = Int[]

    for p in profile.vertices
        r = coords(p).x
        z = coords(p).y
        @assert r >= zero(r) "profile radii must be nonnegative"
        push!(ring_starts, length(points) + 1)
        if iszero(r)
            push!(ring_sizes, 1)
            push!(points, Point(zero(r), zero(r), z))
        else
            push!(ring_sizes, n_theta)
            for i in 0:(n_theta-1)
                θ = 2pi * i / n_theta
                push!(points, Point(r * cos(θ), r * sin(θ), z))
            end
        end
    end

    # Connections: use quads between regular rings and triangles for zero radius points.
    connec = Union{
        typeof(connect((1, 2, 3), Triangle)),
        typeof(connect((1, 2, 3, 4), Quadrangle)),
    }[]
    for layer in 1:(length(profile.vertices)-1)
        start_1 = ring_starts[layer]
        start_2 = ring_starts[layer+1]
        size_1 = ring_sizes[layer]
        size_2 = ring_sizes[layer+1]

        if size_1 == 1 && size_2 == 1
            error("adjacent points cannot both have radius 0")
        elseif size_1 == 1
            ir0 = start_1
            for i in 0:(n_theta-1)
                i3 = start_2 + i
                i4 = start_2 + mod(i + 1, n_theta)
                push!(connec, connect((ir0, i3, i4), Triangle))
            end
        elseif size_2 == 1
            ir0 = start_2
            for i in 0:(n_theta-1)
                i1 = start_1 + i
                i2 = start_1 + mod(i + 1, n_theta)
                push!(connec, connect((i1, ir0, i2), Triangle))
            end
        else
            for i in 0:(n_theta-1)
                i1 = start_1 + i
                i2 = start_1 + mod(i + 1, n_theta)
                i3 = start_2 + i
                i4 = start_2 + mod(i + 1, n_theta)
                push!(connec, connect((i1, i3, i4, i2), Quadrangle))
            end
        end
    end

    return SimpleMesh(points, connec)
end

function wavebot_profile(
    cylinder_height::Union{Length,Real}=0.16m,
    frustum_height::Union{Length,Real}=0.37m,
    top_radius::Union{Length,Real}=0.88m,
    bottom_radius::Union{Length,Real}=0.35m;
    n_points::NTuple{3,Integer}=(5, 10, 5),
)
    R = promote_type(typeof(top_radius), typeof(bottom_radius))
    Z = promote_type(typeof(cylinder_height), typeof(bottom_radius))
    profile_points = typeof(Point(zero(R), zero(Z)))[]

    n_bottom, n_frustum, n_cylinder = n_points

    for j in 0:(n_cylinder-1)
        t = j / (n_cylinder - 1)
        z = -cylinder_height * t
        push!(profile_points, Point(top_radius, z))
    end

    for j in 1:(n_frustum-1)
        t = j / (n_frustum - 1)
        radius = top_radius + (bottom_radius - top_radius) * t
        z = -cylinder_height - frustum_height * t
        push!(profile_points, Point(radius, z))
    end

    z_bottom = -cylinder_height - frustum_height
    for j in 1:(n_bottom-1)
        t = j / (n_bottom - 1)
        push!(profile_points, Point(bottom_radius * (1 - t), z_bottom))
    end

    return Rope(profile_points)
end

# Example: Create mesh & plot
profile = wavebot_profile()
mesh = axisymmetric_mesh(profile, 72)
display(mesh)













# profile_r = [ustrip.(v.coords.x) for v in profile.vertices]
# profile_z = [ustrip.(v.coords.y) for v in profile.vertices]

# fig = Makie.Figure(size=(1200, 600));
# ax = Makie.Axis(
#     fig[1, 1];
#     aspect=Makie.DataAspect(),
#     title="WaveBot Profile",
#     xlabel="r [m]",
#     ylabel="z [m]",
#     limits=((0, nothing), (nothing, 1e-6)),
# );
# Makie.lines!(ax, profile_r, profile_z, color=:gold, linewidth=2);
# Makie.scatter!(ax, profile_r, profile_z, color=:black, markersize=10);

# ax = Makie.Axis3(
#     fig[1, 2];
#     aspect=:data,
#     azimuth=π / 8,
#     elevation=π / 8,
#     title="WaveBot Mesh",
#     xlabel="x [m]",
#     ylabel="y [m]",
#     zlabel="z [m]",
# );
# viz!(ax, mesh, color=:gold, showsegments=true, segmentcolor=:grey50, segmentsize=0.25);

# display(fig)

# # Example: Automatic differentiation
# function wavebot_mesh(x)
#     hc, hf, rt, rb = x
#     n_points = (5, 10, 5)
#     n_theta = 72
#     profile = wavebot_profile(hc, hf, rt, rb; n_points)
#     return axisymmetric_mesh(profile, n_theta)
# end

# x0 = [0.16, 0.37, 0.88, 0.35]

# # The first vertex of the first panel has coordinate (radius_top, 0, 0),
# # so the gradient of its x component wrt the geometric parameters should be [0, 0, 1, 0]
# println(ForwardDiff.gradient(x -> ustrip(coords(wavebot_mesh(x)[1].vertices[1]).x), x0))

# # We will need the gradients of the `area` and `centroid` functions.
# # Note: `centroid` is the vertex centroid and not the are centroid.
# i = 1 # panel number
# ForwardDiff.gradient(x -> ustrip(area(wavebot_mesh(x)[i])), x0)
