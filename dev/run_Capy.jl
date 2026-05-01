using PyCall
cpt = pyimport("capytaine")

# 1. Use the new recommended function for a box mesh
# Note: we use 'cpt.meshes.predefined' to match the library structure
box = cpt.meshes.predefined.mesh_parallelepiped(size=(1.0, 1.0, 1.0), center=(0, 0, 0))

# 2. Check the number of faces (using the mesh property)
# In Julia/PyCall, we access Python attributes just like Julia fields
num_faces = box.nb_faces

println("Success! The box mesh has $num_faces faces.")