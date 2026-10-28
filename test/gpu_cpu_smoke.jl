# CPU-only smoke of the GPU-generic broadcasting / linsolve path (no CUDA, no Capytaine).
using Test
using LinearAlgebra
using StaticArrays
using ForwardDiff
using DifferentiationInterface
using MarineHydro

const MH = MarineHydro

const HAS_ENZYME = try
    using Enzyme
    true
catch
    false
end

vertices = [
    SVector(-1.0, -1.0, -1.0),
    SVector( 0.0, -1.0, -1.0),
    SVector( 0.0,  0.0, -1.0),
    SVector(-1.0,  0.0, -1.0),
    SVector( 1.0, -1.0, -1.0),
    SVector( 1.0,  0.0, -1.0),
]
faces = [SVector(1, 2, 3, 4), SVector(2, 5, 6, 3)]
centers = [SVector(-0.5, -0.5, -1.0), SVector(0.5, -0.5, -1.0)]
normals = [SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 1.0)]
smesh = MH.StaticArraysMesh(vertices, faces, centers, normals, [1.0, 1.0], [sqrt(0.5), sqrt(0.5)], 6, 2)

gfs = (Rankine(), RankineReflected(), GFWu())
gfs_del = (DelhommeauRankine(), DelhommeauRankineReflected(), GFWu())
k = 1.0
smoke_ω = 1.0
bc = [-1im * smoke_ω * n[3] for n in smesh.normals]

function enzyme_assembly_k_sum(kabs, gfs, smesh)
    S, _ = assemble_matrices(gfs, smesh, kabs)
    return real(sum(S))
end

# Geometry AD: the active array is an explicit argument so Enzyme can allocate a
# shadow. Do not capture smesh in a closure (that aliases primal storage).
function enzyme_wu_from_centers(centers, vertices, faces, normals, areas, radii, nv, nf, k)
    m = MH.StaticArraysMesh(vertices, faces, centers, normals, areas, radii, nv, nf)
    S, D = MH.assemble_wu_centers(m, k, Val(true); include_identity=false)
    return real(sum(S) + sum(D))
end

function enzyme_from_vertices(vertices, faces, centers, normals, areas, radii, nv, nf, k, gfs)
    m = MH.StaticArraysMesh(vertices, faces, centers, normals, areas, radii, nv, nf)
    S, D = assemble_matrices(gfs, m, k; include_identity=false)
    return real(sum(S) + sum(D))
end

function enzyme_A_of_omega(omega, smesh, dof)
    return calculate_radiation_forces(smesh, dof, omega)[1]
end

function enzyme_A_from_centers(centers, vertices, faces, normals, areas, radii, nv, nf, omega, dof)
    m = MH.StaticArraysMesh(vertices, faces, centers, normals, areas, radii, nv, nf)
    return calculate_radiation_forces(m, dof, omega)[1]
end

function enzyme_A_from_vertices(vertices, faces, centers, normals, areas, radii, nv, nf, omega, dof)
    m = MH.StaticArraysMesh(vertices, faces, centers, normals, areas, radii, nv, nf)
    return calculate_radiation_forces(m, dof, omega)[1]
end

function enzyme_A_of_radius(r, smesh0, omega, dof)
    s = r
    z = zero(s)
    m = MH.StaticArraysMesh(
        [s * v for v in smesh0.vertices],
        smesh0.faces,
        [s * c for c in smesh0.centers],
        [n .+ z for n in smesh0.normals],
        [a * (s * s) for a in smesh0.areas],
        [ρ * s for ρ in smesh0.radii],
        smesh0.nvertices,
        smesh0.nfaces,
    )
    return calculate_radiation_forces(m, dof, omega)[1]
end

@testset "CPU broadcasting autodispatch" begin
    S, D = MH.assemble_matrices_broadcasting(gfs, smesh, k)
    elements = [element(smesh, i) for i in 1:smesh.nfaces]
    S2, D2 = MH.assemble_matrices_broadcasting(gfs, elements, k)
    S3, D3 = assemble_matrices(gfs, smesh, k)
    @test S ≈ S2 ≈ S3
    @test D ≈ D2 ≈ D3

    Sd, Dd = assemble_matrices(gfs_del, smesh, k)
    @test size(Sd) == (2, 2)
    @test Sd ≈ S rtol=1e-2 atol=1e-3

    ϕ, _ = solve(D, S, bc; direct=true)
    @test length(ϕ) == 2
    @test all(isfinite, ϕ)

    ϕi, src = solve(D, S, bc; direct=false)
    @test length(ϕi) == 2
    @test src !== nothing

    S_ka, D_ka = MH.assemble_matrices_ka(gfs, smesh, k)
    @test Array(S_ka) ≈ S rtol=1e-8 atol=1e-10
    @test Array(D_ka) ≈ D rtol=1e-8 atol=1e-10

    fd = ForwardDiff.derivative(κ -> real(sum(assemble_matrices(gfs, smesh, κ)[1])), k)
    @test isfinite(fd)

    function wu_center_obj(cx)
        T = typeof(cx)
        cT(v) = SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
        centers = [i == 1 ? SVector{3,T}(cx, T(smesh.centers[1][2]), T(smesh.centers[1][3])) : cT(smesh.centers[i])
                   for i in 1:smesh.nfaces]
        m = MH.StaticArraysMesh(
            cT.(smesh.vertices), smesh.faces, centers, cT.(smesh.normals),
            T.(smesh.areas), T.(smesh.radii), smesh.nvertices, smesh.nfaces)
        S, D = MH.assemble_wu_centers(m, k, Val(true); include_identity=false)
        return real(sum(S) + sum(D))
    end
    fd_cx = ForwardDiff.derivative(wu_center_obj, smesh.centers[1][1])
    @test isfinite(fd_cx)
    @test abs(fd_cx) > 0

    if HAS_ENZYME
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        ez = first(Enzyme.gradient(mode, enzyme_assembly_k_sum, k,
            Enzyme.Const(gfs), Enzyme.Const(smesh)))
        @test ez ≈ fd rtol=1e-5 atol=1e-6

        centers_ad = copy(smesh.centers)
        gcx = first(Enzyme.gradient(mode, enzyme_wu_from_centers, centers_ad,
            Enzyme.Const(smesh.vertices), Enzyme.Const(smesh.faces),
            Enzyme.Const(smesh.normals), Enzyme.Const(smesh.areas),
            Enzyme.Const(smesh.radii), Enzyme.Const(smesh.nvertices),
            Enzyme.Const(smesh.nfaces), Enzyme.Const(k)))
        @test gcx[1][1] ≈ fd_cx rtol=1e-8 atol=1e-10
        @test centers_ad == smesh.centers

        function fd_vx(vx)
            T = typeof(vx)
            cT(v) = SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
            vs = [i == 1 ? SVector{3,T}(vx, T(smesh.vertices[1][2]), T(smesh.vertices[1][3])) :
                  cT(smesh.vertices[i]) for i in 1:smesh.nvertices]
            m = MH.StaticArraysMesh(vs, smesh.faces, cT.(smesh.centers), cT.(smesh.normals),
                T.(smesh.areas), T.(smesh.radii), smesh.nvertices, smesh.nfaces)
            S, D = assemble_matrices(gfs, m, k; include_identity=false)
            return real(sum(S) + sum(D))
        end
        fd_vx0 = ForwardDiff.derivative(fd_vx, smesh.vertices[1][1])
        verts_ad = copy(smesh.vertices)
        gvx = first(Enzyme.gradient(mode, enzyme_from_vertices, verts_ad,
            Enzyme.Const(smesh.faces), Enzyme.Const(smesh.centers),
            Enzyme.Const(smesh.normals), Enzyme.Const(smesh.areas),
            Enzyme.Const(smesh.radii), Enzyme.Const(smesh.nvertices),
            Enzyme.Const(smesh.nfaces), Enzyme.Const(k), Enzyme.Const(gfs)))
        @test gvx[1][1] ≈ fd_vx0 rtol=1e-8 atol=1e-10
        @test verts_ad == smesh.vertices

        dof = SVector(0.0, 0.0, 1.0)
        A_fd_ω = ForwardDiff.derivative(w -> calculate_radiation_forces(smesh, dof, w)[1], smoke_ω)
        A_ez_ω = first(Enzyme.gradient(mode, enzyme_A_of_omega, smoke_ω,
            Enzyme.Const(smesh), Enzyme.Const(dof)))
        @test A_ez_ω ≈ A_fd_ω rtol=1e-5 atol=1e-6

        function fd_A_cx(cx)
            T = typeof(cx)
            cT(v) = SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
            centers = [i == 1 ? SVector{3,T}(cx, T(smesh.centers[1][2]), T(smesh.centers[1][3])) :
                       cT(smesh.centers[i]) for i in 1:smesh.nfaces]
            m = MH.StaticArraysMesh(
                cT.(smesh.vertices), smesh.faces, centers, cT.(smesh.normals),
                T.(smesh.areas), T.(smesh.radii), smesh.nvertices, smesh.nfaces)
            return calculate_radiation_forces(m, dof, smoke_ω)[1]
        end
        fd_A_cx0 = ForwardDiff.derivative(fd_A_cx, smesh.centers[1][1])
        centers_A = copy(smesh.centers)
        gA_cx = first(Enzyme.gradient(mode, enzyme_A_from_centers, centers_A,
            Enzyme.Const(smesh.vertices), Enzyme.Const(smesh.faces),
            Enzyme.Const(smesh.normals), Enzyme.Const(smesh.areas),
            Enzyme.Const(smesh.radii), Enzyme.Const(smesh.nvertices),
            Enzyme.Const(smesh.nfaces), Enzyme.Const(smoke_ω), Enzyme.Const(dof)))
        @test gA_cx[1][1] ≈ fd_A_cx0 rtol=1e-5 atol=1e-6
        @test centers_A == smesh.centers

        function fd_A_vx(vx)
            T = typeof(vx)
            cT(v) = SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
            vs = [i == 1 ? SVector{3,T}(vx, T(smesh.vertices[1][2]), T(smesh.vertices[1][3])) :
                  cT(smesh.vertices[i]) for i in 1:smesh.nvertices]
            m = MH.StaticArraysMesh(vs, smesh.faces, cT.(smesh.centers), cT.(smesh.normals),
                T.(smesh.areas), T.(smesh.radii), smesh.nvertices, smesh.nfaces)
            return calculate_radiation_forces(m, dof, smoke_ω)[1]
        end
        fd_A_vx0 = ForwardDiff.derivative(fd_A_vx, smesh.vertices[1][1])
        verts_A = copy(smesh.vertices)
        gA_vx = first(Enzyme.gradient(mode, enzyme_A_from_vertices, verts_A,
            Enzyme.Const(smesh.faces), Enzyme.Const(smesh.centers),
            Enzyme.Const(smesh.normals), Enzyme.Const(smesh.areas),
            Enzyme.Const(smesh.radii), Enzyme.Const(smesh.nvertices),
            Enzyme.Const(smesh.nfaces), Enzyme.Const(smoke_ω), Enzyme.Const(dof)))
        @test gA_vx[1][1] ≈ fd_A_vx0 rtol=1e-5 atol=1e-6
        @test verts_A == smesh.vertices

        A_fd_r = ForwardDiff.derivative(r -> enzyme_A_of_radius(r, smesh, smoke_ω, dof), 1.0)
        A_ez_r = first(Enzyme.gradient(mode, enzyme_A_of_radius, 1.0,
            Enzyme.Const(smesh), Enzyme.Const(smoke_ω), Enzyme.Const(dof)))
        @test isfinite(A_ez_r)
        @test A_ez_r ≈ A_fd_r rtol=1e-5 atol=1e-6
    else
        @test_skip "Enzyme not installed"
    end
end
