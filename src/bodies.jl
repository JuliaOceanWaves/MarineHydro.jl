
using LinearAlgebra: cross, dot, norm
import Base: +

struct FloatingBody
    mesh::Mesh
    dofs::NamedTuple 
    body_name::String

    function FloatingBody(mesh::Mesh, dofs::NamedTuple, body_name::String)
        # add assert statements

        return new(mesh, dofs, body_name)
    end
end



function FloatingBody(mesh::Mesh, rigid_dof_list::Vector{String}, rotation_center::AbstractVector, body_name::String)
    
    # generator
    dof_pairs = (Symbol(name) => if name in ["Surge", "Sway", "Heave"]
            translational_dofs(mesh, name)
        else
            rotational_dofs(mesh, name, rotation_center)
        end for name in rigid_dof_list)
            
    # convert Pair to NamedTuple using ; and splat
    dofs = (; dof_pairs...)

    return FloatingBody(mesh, dofs, body_name)
end




# rigid_dof_list can contain symbols or strings 
function FloatingBody(mesh::Mesh, rigid_dof_list::Vector{Symbol}, rotation_center::AbstractVector, body_name::String)
    return FloatingBody(mesh, string.(rigid_dof_list), rotation_center, body_name)
end

function translational_dofs(mesh::Mesh, dof_name::String)
    num_panels = mesh.nfaces
    dof = zeros(num_panels, 3)
    if dof_name=="Surge"
        dof[:,1] .= 1
    elseif dof_name=="Sway"
        dof[:,2] .= 1
    elseif dof_name=="Heave"
        dof[:,3] .= 1
    end
    return dof
end

function rotational_dofs(mesh::Mesh, dof_name::String, rotation_center::AbstractVector)
    face_centers = mesh.centers
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

# If rotation center not specified, assume it is at origin.
function FloatingBody(mesh::Mesh, rigid_dof_list::Vector{String}, body_name::String)
    rotation_center = [0.0,0.0,0.0]
    for dof in rigid_dof_list
        if dof in ["Roll","Pitch","Yaw"]
            display("Setting origin as rotation center.")
        end
    end
    return FloatingBody(mesh, rigid_dof_list, rotation_center, body_name)
end

function make_body_name_list_unique(strings::Vector{String})
    seen = Dict{String, Int}()
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


#  Combining multiple FloatingBody structs into one FloatingBody struct  
function combine_floatingbodies(floatingbodylist::Vector{FloatingBody},new_body_name::String)

    mesh_list = [floatingbody.mesh for floatingbody in floatingbodylist]
    dof_list = [floatingbody.dofs for floatingbody in floatingbodylist]  
    body_name_list_temp = [replace(floatingbody.body_name, " " => "_") for floatingbody in floatingbodylist]
    num_face_list = [mesh.nfaces for mesh in mesh_list]
    cum_num_face_list = cumsum(num_face_list)
    tot_num_faces = sum(num_face_list)

    body_name_list = make_body_name_list_unique(body_name_list_temp)



    # New Mesh struct
    new_mesh = combine_meshes(mesh_list)

    # New FloatingBody dof_name and dof_value

    new_dof_keys = Symbol[]
    new_dof_mats = []
    for (body_index,body_name) in enumerate(body_name_list)
        # define nbf as cumalitive number of faces for previous body
        # This is used for shifting the location of the dof_mat 
        if body_index==1
            nbf = 0
        else
            nbf = cum_num_face_list[body_index-1] 
        end
        dofs = dof_list[body_index]
        for dof_name in keys(dofs)
            new_dof_key = Symbol(join([body_name,dof_name],"__"))
            dof_mat = dofs[dof_name]
            new_dof_mat = zeros(tot_num_faces,3)
            new_dof_mat[nbf+1:nbf+length(dof_mat[:,1]),:] = dof_mat  
            push!(new_dof_keys,new_dof_key)
            push!(new_dof_mats,new_dof_mat)        
        end
    end
    new_dofs = NamedTuple{tuple(new_dof_keys...)}(tuple(new_dof_mats...))

    return FloatingBody(new_mesh,new_dofs,new_body_name)
end

function combine_floatingbodies(floatingbodylist::Vector{FloatingBody})
    # New FloatingBody name
    body_name_list_temp = [replace(floatingbody.body_name, " " => "_") for floatingbody in floatingbodylist]
    body_name_list = make_body_name_list_unique(body_name_list_temp)
    return combine_floatingbodies(floatingbodylist::Vector{FloatingBody},join(body_name_list,"+"))
end

function +(fb1::FloatingBody, fb2::FloatingBody)
    return combine_floatingbodies([fb1, fb2])
end

function +(fb1::FloatingBody, fb_vec::Vector{FloatingBody})
    return combine_floatingbodies(vcat(fb1, fb_vec))
end