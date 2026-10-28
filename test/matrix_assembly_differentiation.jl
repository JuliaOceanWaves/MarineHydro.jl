using Test
using Zygote
using ForwardDiff
using StaticArrays
using MarineHydro
using PyCall
using LinearAlgebra

const HAS_ENZYME = try
    using Enzyme
    true
catch
    false
end

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

        if HAS_ENZYME
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            function enzyme_S_sum(vertices, faces, centers, normals, areas, radii, nv, nf, gfs, w)
                mesh_new = Mesh(vertices, faces, centers, normals, areas, radii, nv, nf)
                S, _ = assemble_matrices(gfs, mesh_new, w)
                return real(sum(S))
            end
            function enzyme_D_sum(vertices, faces, centers, normals, areas, radii, nv, nf, gfs, w)
                mesh_new = Mesh(vertices, faces, centers, normals, areas, radii, nv, nf)
                _, D = assemble_matrices(gfs, mesh_new, w)
                return real(sum(D))
            end
            gS_zy = Zygote.gradient(v -> real(sum(assemble_matrices(green_functions,
                Mesh(v, mesh.faces, mesh.centers, mesh.normals, mesh.areas, mesh.radii,
                    mesh.nvertices, mesh.nfaces), ω)[1])), mesh.vertices)[1]
            gS_ez = first(Enzyme.gradient(mode, enzyme_S_sum, copy(mesh.vertices),
                Enzyme.Const(mesh.faces), Enzyme.Const(mesh.centers), Enzyme.Const(mesh.normals),
                Enzyme.Const(mesh.areas), Enzyme.Const(mesh.radii), Enzyme.Const(mesh.nvertices),
                Enzyme.Const(mesh.nfaces), Enzyme.Const(green_functions), Enzyme.Const(ω)))
            @test all(isfinite, gS_ez)
            @test gS_ez ≈ gS_zy rtol=1e-8 atol=1e-10

            gD_zy = Zygote.gradient(v -> real(sum(assemble_matrices(green_functions,
                Mesh(v, mesh.faces, mesh.centers, mesh.normals, mesh.areas, mesh.radii,
                    mesh.nvertices, mesh.nfaces), ω)[2])), mesh.vertices)[1]
            gD_ez = first(Enzyme.gradient(mode, enzyme_D_sum, copy(mesh.vertices),
                Enzyme.Const(mesh.faces), Enzyme.Const(mesh.centers), Enzyme.Const(mesh.normals),
                Enzyme.Const(mesh.areas), Enzyme.Const(mesh.radii), Enzyme.Const(mesh.nvertices),
                Enzyme.Const(mesh.nfaces), Enzyme.Const(green_functions), Enzyme.Const(ω)))
            @test all(isfinite, gD_ez)
            @test gD_ez ≈ gD_zy rtol=1e-8 atol=1e-10
        end
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

        if HAS_ENZYME
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            function enzyme_solve_sum(D, S, bc, direct)
                return real(sum(solve(D, S, bc; direct=direct)[1]))
            end
            for direct in (true, false)
                zy = Zygote.gradient((D, S, bc) -> real(sum(solve(D, S, bc; direct=direct)[1])), D, S, bc)
                ez = Enzyme.gradient(mode, enzyme_solve_sum, copy(D), copy(S), copy(bc), Enzyme.Const(direct))
                @test all(isfinite, ez[1]) && all(isfinite, ez[2]) && all(isfinite, ez[3])
                @test ez[1] ≈ zy[1] rtol=1e-8 atol=1e-10
                @test ez[2] ≈ zy[2] rtol=1e-8 atol=1e-10
                @test ez[3] ≈ zy[3] rtol=1e-8 atol=1e-10
            end
        end
    end

    @testset "solve multi-RHS (N×k) primal and AD" begin
        D = rand(4, 4)
        S = rand(4, 4)
        bc = rand(4, 2)
        for i in axes(D, 1)
            D[i, i] += sum(abs, D[i, :])
        end
        for direct in (true, false)
            ϕ, src = solve(D, S, bc; direct=direct)
            @test size(ϕ) == (4, 2)
            if !direct
                @test size(src) == (4, 2)
            end
            ϕ1, _ = solve(D, S, bc[:, 1]; direct=direct)
            ϕ2, _ = solve(D, S, bc[:, 2]; direct=direct)
            @test ϕ[:, 1] ≈ ϕ1
            @test ϕ[:, 2] ≈ ϕ2

            fsum(D, S, bc) = real(sum(solve(D, S, bc; direct=direct)[1]))
            zy = Zygote.gradient(fsum, D, S, bc)
            @test all(isfinite, zy[1]) && all(isfinite, zy[2]) && all(isfinite, zy[3])
            # Dual numbers take native `\`, which must accept the N×k RHS
            e11 = zeros(size(D)); e11[1, 1] = 1
            fdA = ForwardDiff.derivative(x -> fsum(D .+ x .* e11, S, bc), 0.0)
            @test fdA ≈ zy[1][1, 1] rtol=1e-6 atol=1e-8

            if HAS_ENZYME
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
                function enzyme_solve_sum_mat(D, S, bc, direct)
                    return real(sum(solve(D, S, bc; direct=direct)[1]))
                end
                ez = Enzyme.gradient(mode, enzyme_solve_sum_mat, copy(D), copy(S), copy(bc), Enzyme.Const(direct))
                @test all(isfinite, ez[1]) && all(isfinite, ez[2]) && all(isfinite, ez[3])
                @test ez[1] ≈ zy[1] rtol=1e-8 atol=1e-10
                @test ez[2] ≈ zy[2] rtol=1e-8 atol=1e-10
                @test ez[3] ≈ zy[3] rtol=1e-8 atol=1e-10
            end
        end

        # N×1 diffraction-style RHS must not collapse to a Vector inside LinearSolve
        bc1 = rand(4, 1)
        ϕ, src = solve(D, S, bc1; direct=false)
        @test size(ϕ) == (4, 1) && size(src) == (4, 1)
    end

    @testset "Fused Wu assemble d/dk via ForwardDiff (no assemble rrule)" begin
        smesh = MarineHydro.StaticArraysMesh(mesh)
        k0 = 1.2
        function f(k)
            S, D = MarineHydro.assemble_wu_centers(smesh, k, Val(true); include_identity=false)
            return real(sum(S) + sum(D))
        end
        fd = ForwardDiff.derivative(f, k0)
        @test isfinite(fd)
        for direct in (true, false)
            g(k) = begin
                S, D = MarineHydro.assemble_wu_centers(smesh, k, Val(direct); include_identity=false)
                real(sum(S)) + imag(sum(D))
            end
            @test isfinite(ForwardDiff.derivative(g, k0))
        end
    end

    @testset "Fused Wu assemble d/d center via ForwardDiff" begin
        smesh = MarineHydro.StaticArraysMesh(mesh)
        k0 = 1.2
        c0 = smesh.centers[1]
        function f(cx)
            T = typeof(cx)
            cT(v) = SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
            centers = [i == 1 ? SVector{3,T}(cx, T(c0[2]), T(c0[3])) : cT(smesh.centers[i])
                       for i in 1:smesh.nfaces]
            m = MarineHydro.StaticArraysMesh(
                cT.(smesh.vertices), smesh.faces, centers, cT.(smesh.normals),
                T.(smesh.areas), T.(smesh.radii), smesh.nvertices, smesh.nfaces)
            S, D = MarineHydro.assemble_wu_centers(m, k0, Val(true); include_identity=false)
            return real(sum(S) + sum(D))
        end
        fd = ForwardDiff.derivative(f, c0[1])
        @test isfinite(fd)
        @test abs(fd) > 0
    end

    @testset "Enzyme matches Zygote dA/d vertices on dense Mesh" begin
        if !HAS_ENZYME
            @test_skip "Enzyme not installed"
        else
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            dof_v = [0.0, 0.0, 1.0]
            function enzyme_A_verts(vertices, faces, centers, normals, areas, radii, nv, nf, omega, dof)
                m = Mesh(vertices, faces, centers, normals, areas, radii, nv, nf)
                return calculate_radiation_forces(m, dof, omega)[1]
            end
            g_zy = Zygote.gradient(v -> calculate_radiation_forces(
                Mesh(v, mesh.faces, mesh.centers, mesh.normals, mesh.areas, mesh.radii,
                    mesh.nvertices, mesh.nfaces), dof_v, ω)[1], mesh.vertices)[1]
            g_ez = first(Enzyme.gradient(mode, enzyme_A_verts, copy(mesh.vertices),
                Enzyme.Const(mesh.faces), Enzyme.Const(mesh.centers), Enzyme.Const(mesh.normals),
                Enzyme.Const(mesh.areas), Enzyme.Const(mesh.radii), Enzyme.Const(mesh.nvertices),
                Enzyme.Const(mesh.nfaces), Enzyme.Const(ω), Enzyme.Const(dof_v)))
            @test all(isfinite, g_zy) && all(isfinite, g_ez)
            @test g_ez ≈ g_zy rtol=1e-6 atol=1e-8
        end
    end
end
