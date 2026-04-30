import Meshes as Meshes
using CairoMakie: CairoMakie as Makie
using Unitful: m, ustrip, Length
using LinearAlgebra: cross, dot, norm


############
#  Meshes  #
############
import Base: +

struct Mesh
    vertices::AbstractMatrix
    faces::AbstractMatrix
    centers::AbstractMatrix
    normals::AbstractMatrix
    areas::AbstractVector
    radii::AbstractVector
    nvertices::Number
    nfaces::Number
    function Mesh(vertices::AbstractMatrix, faces::AbstractMatrix,
                  centers::AbstractMatrix, normals::AbstractMatrix, areas::AbstractVector,
                  radii::AbstractVector,nvertices::Number,
                  nfaces::Number
                  )
        # TODO: centers, areas, and radii(?) could be calculated from list of vertices and faces
        nvertices = size(vertices)[1]
        @assert (size(vertices) == (nvertices, 3)) "each vertex needs 3 coordinates"
        nfaces = size(faces)[1]
        @assert (size(faces) == (nfaces, 4)) "only quadrilateral panels are allowed"
        @assert (size(centers) == (nfaces, 3)) "centers must be [nfaces x 3]"
        @assert (size(centers) == (nfaces, 3)) "centers must be [nfaces x 3]"
        @assert (length(areas) == nfaces) "areas vector must have length nfaces"
        @assert (length(radii) == nfaces) "radii vector must have length nfaces"
        return new(vertices, faces, centers, normals, areas, radii, nvertices, nfaces)
    end
end

struct StaticArraysMesh
    vertices::Vector{SVector{3, Float64}}
    faces::Vector{SVector{4, Int64}}
    centers::Vector{SVector{3, Float64}}
    normals::Vector{SVector{3, Float64}}
    areas::Vector{Float64}
    radii::Vector{Float64}
    nvertices::Int64
    nfaces::Int64
end



#  Combining multiple Mesh structs into one Mesh struct  
function combine_meshes(meshlist::Vector{Mesh})

    # Make lists
    vetrices_list = [mesh.vertices for mesh in meshlist]
    faces_list = [mesh.faces for mesh in meshlist]
    centers_list = [mesh.centers for mesh in meshlist]
    normals_list = [mesh.normals for mesh in meshlist]
    areas_list = [mesh.areas for mesh in meshlist]
    radii_list = [mesh.radii for mesh in meshlist]
    nvetrices_list = [mesh.nvertices for mesh in meshlist]
    nfaces_list = [mesh.nfaces for mesh in meshlist]

    # Concatinate
    new_vertices = reduce(vcat, vetrices_list)
    new_centers = reduce(vcat, centers_list)
    new_normals = reduce(vcat, normals_list)
    new_areas = reduce(vcat, areas_list)
    new_radii = reduce(vcat, radii_list)

    # Sum 
    new_nvetrices = sum(nvetrices_list) 
    new_nfaces = sum(nfaces_list) 

    # Recount faces
    cum_nfaces_list = cumsum(nfaces_list)
    new_faces = zeros(Int, new_nfaces,4)
    for (nfaces_index,nfaces) in enumerate(nfaces_list)
        if nfaces_index == 1
            nbf = 0
            nbv = 0
        else
            nbf = cum_nfaces_list[nfaces_index-1]
            nbv = nvetrices_list[nfaces_index-1]
        end
        new_faces[nbf+1:nbf+nfaces,:] = faces_list[nfaces_index] .+ nbv
    end

    # Define combined Mesh struct
    return Mesh(new_vertices,
        new_faces,
        new_centers,
        new_normals,
        new_areas,
        new_radii,
        new_nvetrices,
        new_nfaces)    
end

function +(mesh1::Mesh, mesh2::Mesh)
    return combine_meshes([mesh1, mesh2])
end

function +(mesh1::Mesh, mesh_vec::Vector{Mesh})
    return combine_meshes(vcat(mesh1, mesh_vec))
end







###################################
#  Loading meshes from Capytaine  #
###################################


using PyCall

function example_mesh_from_capytaine(resolution=4)
    cpt = pyimport("capytaine")
    radius = 1.0
    resolution = (resolution, resolution)
    return cpt.mesh_sphere(radius=radius, center=(0, 0, 0), resolution=resolution, name="floating sphere").immersed_part()
end

function Mesh(mesh::PyObject)
    mesh = Mesh(
        mesh.vertices, mesh.faces, 
        mesh.faces_centers, mesh.faces_normals, mesh.faces_areas, mesh.faces_radiuses,mesh.nb_vertices,mesh.nb_faces
    )
    return mesh
end

function StaticArraysMesh(mesh::PyObject)
    mesh = StaticArraysMesh(
        [SVector(v[1], v[2], v[3]) for v in eachrow(mesh.vertices)],
        [SVector(f[1], f[2], f[3], f[4]) .+ 1 for f in eachrow(mesh.faces)],
        [SVector(c[1], c[2], c[3]) for c in eachrow(mesh.faces_centers)],
        [SVector(n[1], n[2], n[3]) for n in eachrow(mesh.faces_normals)],
        mesh.faces_areas,
        mesh.faces_radiuses,
        mesh.nb_vertices,
        mesh.nb_faces
    )
    return mesh
end


##############
#  Elements  #
##############

# Mesh elements can be provided individually as NamedTuples, for instance to test the Green's function
center(nt::NamedTuple) = nt[:center]
vertices(nt::NamedTuple) = nt[:vertices]
radius(nt::NamedTuple) = nt[:radius]
normal(nt::NamedTuple) = nt[:normal]
area(nt::NamedTuple) = nt[:area]
faces(nt::NamedTuple) = nt[:faces]


# When using a Mesh, the element is stored as a reference to the Mesh and a index
struct LazyElement
    mesh::Mesh
    index::Integer
end

function element(mesh::Mesh, J::Int)
    return LazyElement(mesh, J)
end

center(e::LazyElement) = e.mesh.centers[e.index, :] #, e.mesh.centers[e.index, 1], e.mesh.centers[e.index, 2]
normal(e::LazyElement) = e.mesh.normals[e.index, :] #, e.mesh.normals[e.index, 1], e.mesh.normals[e.index, 2]
area(e::LazyElement) = e.mesh.areas[e.index]
radius(e::LazyElement) = e.mesh.radii[e.index]
vertices(e::LazyElement) = e.mesh.vertices[e.mesh.faces[e.index, :] .+ 1, :]


# When using a StaticArraysMesh, the element is stored as a set of static arrays (although LazyElement might also work)
struct StaticElement
    center::SVector{3, Float64}
    vertices::SMatrix{4, 3, Float64, 12}
    normal::SVector{3, Float64}
    area::Float64
    radius::Float64
end

function element(mesh::StaticArraysMesh, J::Int)
    return StaticElement(
        mesh.centers[J],
        hcat(mesh.vertices[mesh.faces[J]]...)',  # makes a 4x3 SMatrix
        mesh.normals[J],
        mesh.areas[J],
        mesh.radii[J],
    )
end

center(e::StaticElement) = e.center
normal(e::StaticElement) = e.normal
area(e::StaticElement) = e.area
radius(e::StaticElement) = e.radius
vertices(e::StaticElement) = e.vertices


########################
#  Reflected elements  #
########################

struct ReflectedElement{T}
    element::T
end
free_surface_symmetry(e) = ReflectedElement(e)

center(e::ReflectedElement{T}) where T <: Union{NamedTuple, LazyElement} = (c = center(e.element); [c[1], c[2], -c[3]])
normal(e::ReflectedElement{T}) where T <: Union{NamedTuple, LazyElement} = (n = normal(e.element); [n[1], n[2], -n[3]])
area(e::ReflectedElement) = area(e.element)
radius(e::ReflectedElement) = radius(e.element)
function vertices(e::ReflectedElement{T} where T <: Union{NamedTuple, LazyElement})
    v = vertices(e.element)
    ET = eltype(v)
    return ET[
        v[4, 1]  v[4, 2]  -v[4, 3];
        v[3, 1]  v[3, 2]  -v[3, 3];
        v[2, 1]  v[2, 2]  -v[2, 3];
        v[1, 1]  v[1, 2]  -v[1, 3];
        ]   # Inverting order such that order is still consistent with normal vector
end

center(e::ReflectedElement{StaticElement}) = (c = center(e.element); SVector(c[1], c[2], -c[3]))
normal(e::ReflectedElement{StaticElement}) = (n = normal(e.element); SVector(n[1], n[2], -n[3]))
area(e::ReflectedElement{StaticElement}) = area(e.element)
radius(e::ReflectedElement{StaticElement}) = radius(e.element)
function vertices(e::ReflectedElement{StaticElement})
    v = vertices(e.element)
    return @SMatrix [
        v[4, 1] v[4, 2] -v[4, 3]
        v[3, 1] v[3, 2] -v[3, 3];
        v[2, 1] v[2, 2] -v[2, 3];
        v[1, 1] v[1, 2] -v[1, 3];
    ]   # Inverting order such that order is still consistent with normal vector
end

############################ Differentiable Mesh #####################################
function axisymmetric_mesh(profile::Meshes.Rope, n_theta::Integer=72)
    @assert n_theta >= 3 "n_theta must be at least 3"
    @assert length(profile.vertices) >= 2 "profile must contain at least two points"

    # Points: Revolve each 2D profile point (r, z) into a 3D ring,
    #  treat zero radius points as special cases
    p1 = Meshes.coords(first(profile.vertices))
    points = typeof(Meshes.Point(p1.x, zero(p1.x), p1.y))[]
    ring_starts = Int[]
    ring_sizes = Int[]

    for p in profile.vertices
        r = Meshes.coords(p).x
        z = Meshes.coords(p).y
        @assert r >= zero(r) "profile radii must be nonnegative"
        push!(ring_starts, length(points) + 1)
        if iszero(r)
            push!(ring_sizes, 1)
            push!(points, Meshes.Point(zero(r), zero(r), z))
        else
            push!(ring_sizes, n_theta)
            for i in 0:(n_theta-1)
                θ = 2pi * i / n_theta
                push!(points, Meshes.Point(r * cos(θ), r * sin(θ), z))
            end
        end
    end

    # Connections: use quads between regular rings and triangles for zero radius points.
    connec = Union{
        typeof(Meshes.connect((1, 2, 3), Meshes.Triangle)),
        typeof(Meshes.connect((1, 2, 3, 4), Meshes.Quadrangle)),
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
                push!(connec, Meshes.connect((ir0, i3, i4), Triangle))
            end
        elseif size_2 == 1
            ir0 = start_2
            for i in 0:(n_theta-1)
                i1 = start_1 + i
                i2 = start_1 + mod(i + 1, n_theta)
                push!(connec, Meshes.connect((i1, ir0, i2), Meshes.Triangle))
            end
        else
            for i in 0:(n_theta-1)
                i1 = start_1 + i
                i2 = start_1 + mod(i + 1, n_theta)
                i3 = start_2 + i
                i4 = start_2 + mod(i + 1, n_theta)
                push!(connec, Meshes.connect((i1, i3, i4, i2), Meshes.Quadrangle))
            end
        end
    end

    return Meshes.SimpleMesh(points, connec)
end

function wavebot_profile(
    cylinder_height::Union{Length,Number}=0.16m,
    frustum_height::Union{Length,Number}=0.37m,
    top_radius::Union{Length,Number}=0.88m,
    bottom_radius::Union{Length,Number}=0.35m;
    n_points::NTuple{3,Integer}=(5, 10, 5),
)
    R = promote_type(typeof(top_radius), typeof(bottom_radius))
    Z = promote_type(typeof(cylinder_height), typeof(bottom_radius))
    profile_points = typeof(Meshes.Point(zero(R), zero(Z)))[]

    n_bottom, n_frustum, n_cylinder = n_points

    for j in 0:(n_cylinder-1)
        t = j / (n_cylinder - 1)
        z = -cylinder_height * t
        push!(profile_points, Meshes.Point(top_radius, z))
    end

    for j in 1:(n_frustum-1)
        t = j / (n_frustum - 1)
        radius = top_radius + (bottom_radius - top_radius) * t
        z = -cylinder_height - frustum_height * t
        push!(profile_points, Meshes.Point(radius, z))
    end

    z_bottom = -cylinder_height - frustum_height
    for j in 1:(n_bottom-1)
        t = j / (n_bottom - 1)
        push!(profile_points, Meshes.Point(bottom_radius * (1 - t), z_bottom))
    end

    return Meshes.Rope(profile_points)
end



function Mesh(mesh::Meshes.SimpleMesh)

    # get vertices
    verts = mesh.vertices
    x_vals = [ustrip.(v.coords.x) for v in verts]
    y_vals = [ustrip.(v.coords.y) for v in verts]
    z_vals = [ustrip.(v.coords.z) for v in verts]
    vertices_mat = hcat(x_vals,y_vals,z_vals)

    # get faces
    connec = Meshes.topology(mesh).connec
    n_faces = length(connec)
    faces_mat = zeros(Number, n_faces, 4)
    for i in 1:n_faces
        inds = Meshes.indices(connec[i])
        inds_vec = collect(inds)
        if length(inds)==3 # triangle element (repeat last vertex like cpt)
            faces_mat[i, :] = vcat(inds_vec,inds_vec[3]) 
        else
            faces_mat[i, :] = inds_vec
        end
        # faces_mat[i, 1:length(inds)] .= inds
    end

    # get areas
    areas_vec = ustrip.(Meshes.area.(mesh))

    # get nvertices
    nvertices = length(vertices_mat[:,1])

    # get nfaces 
    nfaces = length(faces_mat[:,1])

    # get centers and normals (area centroid, not verex centroid)
    center_mat = zeros(Number, nfaces,3)
    normals_mat = zeros(Number, nfaces,3)
    radii_vec = zeros(Number, nfaces)
    for i in 1:nfaces

        # compute center
        faces_vec = faces_mat[i,:]
        if faces_vec[end]==-1 # this is a triangle element
            a = vertices_mat[faces_vec[1],:]
            b = vertices_mat[faces_vec[2],:]
            c = vertices_mat[faces_vec[3],:]
            center_vec = (a+b+c)/3
        else # this is a quadrilateral element
            a = vertices_mat[faces_vec[1],:]
            b = vertices_mat[faces_vec[2],:]
            c = vertices_mat[faces_vec[3],:]
            d = vertices_mat[faces_vec[4],:]
            area1 = 0.5 * norm(cross(b-a, c-a))
            area2 = 0.5 * norm(cross(c-a, d-a))
            c1 = (a+b+c)/3
            c2 = (a+c+d)/3
            center_vec = (c1*area1 + c2*area2) / (area1 + area2)
        end
        center_mat[i,:] = center_vec

        # compute normal
        normals_vec = cross(b-a, c-a)# not yet unit length
        normals_mat[i,:] = normals_vec/norm(normals_vec) # normalize

        # compute radii
        radii_vec[i] = norm(a-center_vec) # distance between center and first vertex
    end

    # need to subtract 1 from faces_mat to start indexing at 0 like cpt

    return Mesh(vertices_mat, faces_mat.-1, center_mat, normals_mat, areas_vec, radii_vec, nvertices, nfaces)
end   


function plot_mesh(mesh::Meshes.SimpleMesh, profile)

    profile_r = [ustrip.(v.coords.x) for v in profile.vertices]
    profile_z = [ustrip.(v.coords.y) for v in profile.vertices]

    fig = Makie.Figure(size=(1200, 600));
    ax = Makie.Axis(
        fig[1, 1];
        aspect=Makie.DataAspect(),
        title="WaveBot Profile",
        xlabel="r [m]",
        ylabel="z [m]",
        limits=((0, nothing), (nothing, 1e-6)),
    );
    Makie.lines!(ax, profile_r, profile_z, color=:gold, linewidth=2);
    Makie.scatter!(ax, profile_r, profile_z, color=:black, markersize=10);

    ax = Makie.Axis3(
        fig[1, 2];
        aspect=:data,
        azimuth=π / 8,
        elevation=π / 8,
        title="WaveBot Mesh",
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    );
    Meshes.viz!(ax, mesh, color=:gold, showsegments=true, segmentcolor=:grey50, segmentsize=0.25);

    display(fig)
end

function wavebot_mesh(r1,r2,d1,d2,n_points = (5, 10, 5),n_theta = 72)

    # geometric design variables
    cylinder_height=d1
    frustum_height=d2
    top_radius=r1
    bottom_radius=r2

    # mesh params
    # n_points = (5, 10, 5)
    # n_theta = 72

    profile = wavebot_profile(cylinder_height, frustum_height, top_radius, bottom_radius; n_points=n_points)
    mesh = axisymmetric_mesh(profile, n_theta)
    MHmesh = Mesh(mesh)

    # if show_plot
    #     plot_mesh(mesh, profile)
    # end
        
    return MHmesh
end