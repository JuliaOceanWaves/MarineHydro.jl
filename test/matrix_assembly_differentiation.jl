using Test
using Zygote
using MarineHydro
using PyCall
using LinearAlgebra

@testset "Matrix Differentiability Tests" begin
    mesh = MarineHydro.Mesh(MarineHydro.example_mesh_from_capytaine())
    green_functions = (Rankine(), RankineReflected(), GFWu())

    # Define parameters
    wavenumber = 1.0
    ω = √(wavenumber * 9.8)
    dof = [0, 0, 1]

    @testset "Radiation Boundary Condition Tests" begin
        # Test differentiability using Zygote
        Ji_bc_mesh, Ji_bc_dof , Ji_bc_ω = Zygote.jacobian((mesh, dof, ω) -> imag.(radiation_bc(mesh, dof, ω)), mesh, dof, ω)

        @test Ji_bc_ω !== nothing
        @test Ji_bc_dof !== nothing
        @test typeof(Ji_bc_ω) == Vector{Float64}
        #Test the size of the gradient

        # Jr_bc_mesh, Jr_bc_dof , Jr_bc_ω = Zygote.jacobian((mesh, dof, ω) -> real.(radiation_bc(mesh, dof, ω)), mesh,dof,ω)

        # @test Jr_bc_ω !== nothing
        # @test typeof(Jr_bc_ω) == Vector{Float64}
        # @test Ji_bc_mesh !== nothing  # this fails not sure why ;  probably due to vector with respect to matrix?
        # Test the size of the gradient
        # @test size(real_grad_bc.normals) == size(mesh.normals)
    end

    @testset "Differentiability of assemble_matrices with respect to mesh.vertices" begin
        # Function to test differentiability of `assemble_matrices` with respect to vertices
        function test_S_assemble_matrices(vertices) #To do - split S and D/K matrix test
            mesh_new = Mesh(
                vertices, mesh.faces,
                mesh.centers, mesh.normals,
                mesh.areas, mesh.radii, mesh.nvertices, mesh.nfaces
            )
            S, _ = assemble_matrices(green_functions, mesh_new, ω)
            return vec(real(S))  # Return the real part of S as a vector for Jacobian computation
        end

        # Compute Jacobian of `assemble_matrices` with respect to `mesh.vertices`
        jacobian_vertices, = Zygote.jacobian(test_S_assemble_matrices, mesh.vertices)

        # Test the Jacobian
        @test typeof(jacobian_vertices) == Matrix{Float64}
        @test !any(isnan, jacobian_vertices[1])

        function test_D_assemble_matrices(vertices)
            mesh_new = Mesh(
                vertices, mesh.faces,
                mesh.centers, mesh.normals,
                mesh.areas, mesh.radii, mesh.nvertices, mesh.nfaces
            )
            _, D = assemble_matrices(green_functions, mesh_new, ω)
            return vec(real(D))  # Return the real part of S as a vector for Jacobian computation
        end

        # # Compute Jacobian of `assemble_matrices` with respect to `mesh.vertices`
        jacobian_vertices, = Zygote.jacobian(test_D_assemble_matrices, mesh.vertices)

        # # Test the Jacobian
        @test typeof(jacobian_vertices) == Matrix{Float64}
        @test !any(isnan, jacobian_vertices)
    end

    @testset "Differentiability of solve (direct and indirect)" begin
        #real BEM matrix not required for just testing solve
        D = rand(3, 3)
        S = rand(3, 3)
        bc = rand(3)
        #  `D` is invertible if its diagonally dominant
        for i in eachindex(D[:,1])
            D[i, i] += sum(abs, D[i, :])
        end
        # Compute gradients using Zygote
        JD, JS, Jbc = Zygote.jacobian((D, S, bc) -> solve(D, S, bc; direct=true)[1], D, S, bc)

        # Check that Jacobian are not `nothing` and have the correct dimensions
        @test size(JD) == (3, 9)  # Jacobian w.r.t. D should have size (output_dim, input_dim_D1 * input_dim_D2) : Zygote flattens input matrix ?
        @test size(JS) == (3, 9)  # Jacobian w.r.t. S should have size (output_dim, input_dim_S1, input_dim_S2)
        @test size(Jbc) == (3, 3)    # Jacobian w.r.t. bc should have size (output_dim, input_dim_bc)
        @test JD !== nothing
        @test JS !== nothing
        @test Jbc !== nothing
        @test typeof(JD) == Matrix{Float64}
        @test typeof(JS) == Matrix{Float64}
        @test typeof(Jbc) == Matrix{Float64}

        # Test indirect mode
        # Compute Jacobians using Zygote
        JD, JS, Jbc = Zygote.jacobian((D, S, bc) -> solve(D, S, bc; direct=false)[1], D, S, bc)

        # Test Jacobian sizes
        @test size(JD) == (3, 9)  # Jacobian w.r.t. D should have size (output_dim, input_dim_D1 * input_dim_D2) : Zygote flattens input matrix ?
        @test size(JS) == (3, 9)  # Jacobian w.r.t. S should have size (output_dim, input_dim_S1, input_dim_S2)
        @test size(Jbc) == (3, 3)    # Jacobian w.r.t. bc should have size (output_dim, input_dim_bc)
        @test JD !== nothing
        @test JS !== nothing
        @test Jbc !== nothing
        @test typeof(JD) == Matrix{Float64}
        @test typeof(JS) == Matrix{Float64}
        @test typeof(Jbc) == Matrix{Float64}
    end
end
