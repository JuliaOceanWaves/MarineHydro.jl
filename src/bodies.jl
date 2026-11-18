
using LinearAlgebra: cross, dot, norm
import Base: +
using Unitful: ustrip

# Mesh type is a parameter: `Mesh` (Zygote / dense) or `StaticArraysMesh`
# (Enzyme / ForwardDiff). Lid type is `Nothing` or the same mesh kind.
# `D` is the concrete NamedTuple type of `dofs` (Enzyme cannot reverse an
# abstract `NamedTuple` field nested in `RadiationProblem`).
struct FloatingBody{M,L,D,C,R,MS}
    mesh::M
    lid_mesh::L
    dofs::D
    body_name::String
    center_of_mass::C
    rotation_center::R
    mass::MS
end

# Main constructor for already-built dofs
function FloatingBody(
    mesh,
    lid_mesh,
    dofs::NamedTuple,
    body_name::String,
    center_of_mass,
    rotation_center::AbstractVector,
    mass
)
    cm = center_of_mass === nothing ? nothing : Float64.(center_of_mass)
    rc = Float64.(rotation_center)
    ms = mass === nothing ? nothing : Float64(mass)

    return FloatingBody{
        typeof(mesh),
        typeof(lid_mesh),
        typeof(dofs),
        typeof(cm),
        typeof(rc),
        typeof(ms)
    }(
        mesh,
        lid_mesh,
        dofs,
        body_name,
        cm,
        rc,
        ms
    )
end

function FloatingBody(
    mesh,
    lid_mesh,
    dofs::NamedTuple,
    body_name::String,
    center_of_mass,
    rotation_center::AbstractVector
)
    return FloatingBody(
        mesh,
        lid_mesh,
        dofs,
        body_name,
        center_of_mass,
        rotation_center,
        nothing
    )
end

# Existing old-style constructor: 5th positional arg is rotation_center
function FloatingBody(
    mesh,
    lid_mesh,
    dofs::NamedTuple,
    body_name::String,
    rotation_center::AbstractVector
)
    return FloatingBody(mesh, lid_mesh, dofs, body_name, nothing, rotation_center)
end

# Default center_of_mass = nothing, rotation_center = origin
function FloatingBody(
    mesh,
    lid_mesh,
    dofs::NamedTuple,
    body_name::String;
    center_of_mass=nothing,
    rotation_center=[0.0, 0.0, 0.0],
    mass=nothing
)
    return FloatingBody(mesh, lid_mesh, dofs, body_name, center_of_mass, rotation_center, mass)
end

function FloatingBody(
    mesh,
    dofs::NamedTuple,
    body_name::String;
    center_of_mass=nothing,
    rotation_center=[0.0, 0.0, 0.0],
    mass=nothing
)
    return FloatingBody(mesh, nothing, dofs, body_name;
        center_of_mass=center_of_mass,
        rotation_center=rotation_center,
        mass=mass
    )
end

function FloatingBody(
    mesh::StaticArraysMesh,
    lid_mesh,
    dofs::NamedTuple,
    body_name::String,
    center_of_mass,
    rotation_center::AbstractVector,
    mass
)
    sm = _repack_smesh(mesh)
    lid = lid_mesh isa StaticArraysMesh ? _repack_smesh(lid_mesh) : lid_mesh

    cm = center_of_mass === nothing ? nothing : Float64.(center_of_mass)
    rc = Float64.(rotation_center)
    ms = mass === nothing ? nothing : Float64(mass)

    return FloatingBody{
        typeof(sm),
        typeof(lid),
        typeof(dofs),
        typeof(cm),
        typeof(rc),
        typeof(ms)
    }(
        sm,
        lid,
        dofs,
        body_name,
        cm,
        rc,
        ms
    )
end

function FloatingBody(
    mesh::StaticArraysMesh,
    lid_mesh,
    dofs::NamedTuple,
    body_name::String,
    center_of_mass,
    rotation_center::AbstractVector
)
    return FloatingBody(
        mesh,
        lid_mesh,
        dofs,
        body_name,
        center_of_mass,
        rotation_center,
        nothing
    )
end

function FloatingBody(
    mesh::StaticArraysMesh,
    lid_mesh,
    dofs::NamedTuple,
    body_name::String,
    rotation_center::AbstractVector
)
    return FloatingBody(mesh, lid_mesh, dofs, body_name, nothing, rotation_center)
end

function FloatingBody(
    mesh::StaticArraysMesh,
    lid_mesh,
    dofs::NamedTuple,
    body_name::String;
    center_of_mass=nothing,
    rotation_center=[0.0, 0.0, 0.0],
    mass=nothing
)
    return FloatingBody(
        mesh,
        lid_mesh,
        dofs,
        body_name,
        center_of_mass,
        rotation_center,
        mass
    )
end

function FloatingBody(
    mesh::StaticArraysMesh,
    dofs::NamedTuple,
    body_name::String;
    center_of_mass=nothing,
    rotation_center=[0.0, 0.0, 0.0],
    mass=nothing
)
    return FloatingBody(mesh, nothing, dofs, body_name;
        center_of_mass=center_of_mass,
        rotation_center=rotation_center,
        mass=mass
    )
end

# Constructor from rigid DOF names
function FloatingBody(
    mesh,
    lid_mesh,
    rigid_dof_list::AbstractVector,
    rotation_center::AbstractVector,
    body_name::String;
    center_of_mass=nothing,
    mass=nothing
)
    rigid_dof_list = string.(rigid_dof_list)

    dof_pairs = (Symbol(name) => if name in ["Surge", "Sway", "Heave"]
        translational_dofs(mesh, name)
    else
        rotational_dofs(mesh, name, rotation_center)
    end for name in rigid_dof_list)

    dofs = (; dof_pairs...)

    return FloatingBody(
        mesh,
        lid_mesh,
        dofs,
        body_name,
        center_of_mass,
        rotation_center,
        mass
    )
end

coord(p) = Float64[
    ustrip(p.coords.x),
    ustrip(p.coords.y),
    ustrip(p.coords.z)
]

function get_num_panels(mesh)
    if hasproperty(mesh, :nfaces)
        return mesh.nfaces
    else
        return Meshes.nelements(mesh)
    end
end

function get_areas(mesh)
    if hasproperty(mesh, :areas)
        return mesh.areas
    else
        return [Meshes.measure(e) for e in Meshes.elements(mesh)]
    end
end

function get_centers(mesh)
    if hasproperty(mesh, :centers)
        return mesh.centers
    else
        centers = zeros(Float64, Meshes.nelements(mesh), 3)

        for (i, elem) in enumerate(Meshes.elements(mesh))
            pts = coord.(Meshes.vertices(elem))
            centers[i, :] .= sum(pts) / length(pts)
        end

        return centers
    end
end

# This is only used for forward speed problems, and rigid_dof_name must be a rigid body dof
function evaluate_gradient_of_motion(mesh, rigid_dof_name::String)
    rigid_dof_name in ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw") ||
        throw(ArgumentError("forward speed problems are only developed for rigid body dofs"))
    num_panels = get_num_panels(mesh)
    ddofdx = zeros(num_panels, 3)
    if rigid_dof_name=="Pitch"
        ddofdx[:, 3] .= 1
    elseif rigid_dof_name=="Yaw"
        ddofdx[:, 2] .= -1
    end
    return ddofdx
end


# rigid_dof_list can contain symbols or strings 
# function FloatingBody(mesh::Mesh, rigid_dof_list::Vector{Symbol}, rotation_center::AbstractVector, body_name::String)
#     return FloatingBody(mesh, string.(rigid_dof_list), rotation_center, body_name)
# end

function translational_dofs(mesh, dof_name::String)
    num_panels = get_num_panels(mesh)
    areas = get_areas(mesh)
    #T = eltype(areas)
    T = Float64
    dof = zeros(T, num_panels, 3)
    if dof_name=="Surge"
        dof[:, 1] .= one(T)
    elseif dof_name=="Sway"
        dof[:, 2] .= one(T)
    elseif dof_name=="Heave"
        dof[:, 3] .= one(T)
    end
    return dof
end

function rotational_dofs(mesh, dof_name::String, rotation_center::AbstractVector)
    face_centers = get_centers(mesh)
    if dof_name=="Roll"
        axis_of_rot = [1, 0, 0]
    elseif dof_name=="Pitch"
        axis_of_rot = [0, 1, 0]
    elseif dof_name=="Yaw"
        axis_of_rot = [0, 0, 1]
    end
    pos_vec = face_centers .- rotation_center'
    dof_vecs = cross.(Ref(axis_of_rot), eachrow(pos_vec))
    dof = copy(stack(dof_vecs)') # make vector of vectors into a matrix
    return dof
end

function rotational_dofs(mesh::StaticArraysMesh, dof_name::String, rotation_center::AbstractVector)
    areas = get_areas(mesh)
    #T = eltype(areas)
    T = Float64
    axis_of_rot = dof_name=="Roll" ? SVector{3,T}(1, 0, 0) :
                  dof_name=="Pitch" ? SVector{3,T}(0, 1, 0) :
                  dof_name=="Yaw" ? SVector{3,T}(0, 0, 1) :
                  throw(ArgumentError("unknown rotational dof $dof_name"))
    rc = SVector{3,T}(T(rotation_center[1]), T(rotation_center[2]), T(rotation_center[3]))
    dof = zeros(T, get_num_panels(mesh), 3)
    face_centers = get_centers(mesh)
    @inbounds for i in 1:get_num_panels(mesh)
        v = cross(axis_of_rot, face_centers[i] - rc)
        dof[i, 1] = v[1]
        dof[i, 2] = v[2]
        dof[i, 3] = v[3]
    end
    return dof
end

# If rotation center not specified, assume it is at origin.
function FloatingBody(
    mesh,
    rigid_dof_list::AbstractVector,
    rotation_center::AbstractVector,
    body_name::String;
    center_of_mass=nothing,
    mass=nothing
)
    return FloatingBody(mesh, nothing, rigid_dof_list, rotation_center, body_name;
        center_of_mass=center_of_mass, mass=mass
    )
end

function FloatingBody(
    mesh,
    lid_mesh,
    rigid_dof_list::AbstractVector,
    body_name::String;
    rotation_center=nothing,
    center_of_mass=nothing,
    mass=nothing
)
    rigid_dof_list_str = string.(rigid_dof_list)

    if rotation_center === nothing
        if any(dof -> dof in ("Roll", "Pitch", "Yaw"), rigid_dof_list_str)
            @warn "Rotation center not specified for rotational DOF(s). Using origin [0.0, 0.0, 0.0]."
        end
        rotation_center = [0.0, 0.0, 0.0]
    end

    return FloatingBody(
        mesh,
        lid_mesh,
        rigid_dof_list_str,
        rotation_center,
        body_name;
        center_of_mass=center_of_mass,
        mass=mass
    )
end

function FloatingBody(
    mesh,
    rigid_dof_list::AbstractVector,
    body_name::String;
    rotation_center=nothing,
    center_of_mass=nothing,
    mass=nothing
)
    return FloatingBody(
        mesh,
        nothing,
        rigid_dof_list,
        body_name;
        rotation_center=rotation_center,
        center_of_mass=center_of_mass,
        mass=mass
    )
end


function make_body_name_list_unique(strings::Vector{String})
    seen = Dict{String,Int}()
    result = Vector{String}(undef, length(strings))

    for (i, s) in enumerate(strings)
        if !haskey(seen, s)
            result[i] = s
            seen[s] = 2
        else
            count = seen[s]
            new_s = "$(s)_$(count)"

            while haskey(seen, new_s) || new_s in strings
                count += 1
                new_s = "$(s)_$(count)"
            end

            result[i] = new_s
            seen[s] = count + 1
            seen[new_s] = 1
        end
    end
    return result
end

struct MultiFloatingBody{FB<:FloatingBody}
    bodies::Vector{FB}
    name::String
    body_names::Vector{Symbol}
end

function MultiFloatingBody(fbs::AbstractVector{<:FloatingBody}, name::String)
    isempty(fbs) && error("Need at least one FloatingBody.")

    ns = Symbol.(make_body_name_list_unique(
        [replace(fb.body_name, " " => "_") for fb in fbs]
    ))

    return MultiFloatingBody(collect(fbs), name, ns)
end

Base.getindex(mb::MultiFloatingBody, i::Int) = mb.bodies[i]
Base.length(mb::MultiFloatingBody) = length(mb.bodies)
Base.iterate(mb::MultiFloatingBody, state...) = iterate(mb.bodies, state...)

#  Combining multiple FloatingBody structs into one FloatingBody struct  
function combine_floatingbodies(floatingbodylist::AbstractVector{<:FloatingBody}, new_body_name::String)

    mesh_list = [floatingbody.mesh for floatingbody in floatingbodylist]
    dof_list = [floatingbody.dofs for floatingbody in floatingbodylist]
    body_name_list_temp = [replace(floatingbody.body_name, " " => "_") for floatingbody in floatingbodylist]
    num_face_list = [get_num_panels(mesh) for mesh in mesh_list]
    cum_num_face_list = cumsum(num_face_list)
    tot_num_faces = sum(num_face_list)

    body_name_list = make_body_name_list_unique(body_name_list_temp)



    # New Mesh struct
    new_mesh = combine_meshes(mesh_list)

    # New FloatingBody dof_name and dof_value

    new_dof_keys = Symbol[]
    new_dof_mats = []
    for (body_index, body_name) in enumerate(body_name_list)
        # define nbf as cumalitive number of faces for previous body
        # This is used for shifting the location of the dof_mat 
        if body_index==1
            nbf = 0
        else
            nbf = cum_num_face_list[body_index-1]
        end
        dofs = dof_list[body_index]
        for dof_name in keys(dofs)
            new_dof_key = Symbol(join([body_name, dof_name], "__"))
            dof_mat = dofs[dof_name]
            n_after = tot_num_faces - nbf - size(dof_mat, 1)
            new_dof_mat = vcat(zeros(nbf, 3), dof_mat, zeros(n_after, 3))
            push!(new_dof_keys, new_dof_key)
            push!(new_dof_mats, new_dof_mat)
        end
    end
    new_dofs = NamedTuple{tuple(new_dof_keys...)}(tuple(new_dof_mats...))

    lids = [fb.lid_mesh for fb in floatingbodylist]
    present_lids = Mesh[m for m in lids if !isnothing(m)]
    new_lid = isempty(present_lids) ? nothing : combine_meshes(present_lids)

    return FloatingBody(
        new_mesh,
        new_lid,
        new_dofs,
        new_body_name;
        center_of_mass=nothing,
        rotation_center=[0.0, 0.0, 0.0],
        mass=nothing
    )
end

function combine_floatingbodies(floatingbodylist::AbstractVector{<:FloatingBody})
    # New FloatingBody name
    body_name_list_temp = [replace(floatingbody.body_name, " " => "_") for floatingbody in floatingbodylist]
    body_name_list = make_body_name_list_unique(body_name_list_temp)
    return combine_floatingbodies(floatingbodylist, join(body_name_list, "+"))
end

function combine_fb_meshes(floatingbodylist::AbstractVector{<:FloatingBody}, new_body_name::String)

    mesh_list = [floatingbody.mesh for floatingbody in floatingbodylist]
    dof_list = [floatingbody.dofs for floatingbody in floatingbodylist]
    body_name_list_temp = [replace(floatingbody.body_name, " " => "_") for floatingbody in floatingbodylist]
    num_face_list = [Meshes.nelements(mesh) for mesh in mesh_list]
    cum_num_face_list = cumsum(num_face_list)
    tot_num_faces = sum(num_face_list)

    body_name_list = make_body_name_list_unique(body_name_list_temp)

    # New Mesh struct
    new_mesh = reduce(Meshes.merge, mesh_list)

    # New FloatingBody dof_name and dof_value
    new_dof_keys = Symbol[]
    new_dof_mats = []
    for (body_index, body_name) in enumerate(body_name_list)
        # define nbf as cumalitive number of faces for previous body
        # This is used for shifting the location of the dof_mat 
        if body_index==1
            nbf = 0
        else
            nbf = cum_num_face_list[body_index-1]
        end
        dofs = dof_list[body_index]
        for dof_name in keys(dofs)
            new_dof_key = Symbol(join([body_name, dof_name], "__"))
            dof_mat = dofs[dof_name]
            n_after = tot_num_faces - nbf - size(dof_mat, 1)
            new_dof_mat = vcat(zeros(nbf, 3), dof_mat, zeros(n_after, 3))
            push!(new_dof_keys, new_dof_key)
            push!(new_dof_mats, new_dof_mat)
        end
    end
    new_dofs = NamedTuple{tuple(new_dof_keys...)}(tuple(new_dof_mats...))

    return FloatingBody(
        new_mesh,
        new_dofs,
        new_body_name;
        center_of_mass=nothing,
        rotation_center=[0.0, 0.0, 0.0],
        mass=nothing
    )
end

function combine_fb_meshes(floatingbodylist::AbstractVector{<:FloatingBody})
    # New FloatingBody name
    body_name_list_temp = [replace(floatingbody.body_name, " " => "_") for floatingbody in floatingbodylist]
    body_name_list = make_body_name_list_unique(body_name_list_temp)
    return combine_fb_meshes(floatingbodylist, join(body_name_list, "+"))
end

function +(fb1::FloatingBody, fb2::FloatingBody)
    return combine_floatingbodies([fb1, fb2])
end

function +(fb1::FloatingBody, fb_vec::AbstractVector{<:FloatingBody})
    return combine_floatingbodies(vcat(fb1, fb_vec))
end

