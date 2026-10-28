# Comprehension is the Zygote / Dual-geometry / `all_normals` path. Do not fold
# it into the StaticArraysMesh CPU kernels: those mutate and are not Zygote-safe.
# Public `assemble_matrices` dispatches on mesh / array / Green's-function types.

function assemble_matrices_comprehension(green_functions, mesh, wavenumber; direct=true, all_normals=nothing, include_identity=true)
    
    free_surface = 0.0

    if eltype(mesh.vertices)<: ForwardDiff.Dual
        T = eltype(mesh.vertices)
    else
        T = eltype(wavenumber)
    end

    S = @inbounds [-1/2τ̅ * Complex{T}(integral(green_functions, element(mesh, i), element(mesh, j), wavenumber)) for i in 1:mesh.nfaces, j in 1:mesh.nfaces]

    D = @inbounds [begin
            element_i = element(mesh, i)
            element_j = element(mesh, j)

            norm_vec = direct ? normal(element_j) : normal(element_i)
        
            n = isnothing(all_normals) ? norm_vec : all_normals                      

            c = i == j ? Complex{T}(1.0, 0.0) : Complex{T}(0.0, 0.0) # if diagonal

            constant = abs(mesh.centers[i,3]-free_surface) < 1e-8 ? c : c/2 # if panel on surface

            # (n' * norm_vec)=1 when n==panel normal vector. When all_normals is provided (for forward speed problems), this is not always the case.
            jump = include_identity ? (constant * (n' * norm_vec)) : zero(Complex{T})
            jump - 1/2τ̅ * Complex{T}(n' * integral_gradient(green_functions, element_i, element_j, wavenumber; with_respect_to_first_variable=!direct))
        end for i in 1:mesh.nfaces, j in 1:mesh.nfaces]

    return S, D
end

function assemble_matrices_explicit_both(green_functions, mesh, wavenumber; direct=true)
    # Benchmark-only variant using `both_integral_and_integral_gradient`.
    # Not on the `assemble_matrices` dispatch path; not Zygote-safe.
    S = Array{ComplexF64, 2}(undef, (mesh.nfaces, mesh.nfaces))
    D = Array{ComplexF64, 2}(undef, (mesh.nfaces, mesh.nfaces))
    for i in 1:mesh.nfaces
        for j in 1:mesh.nfaces
            element_i = element(mesh, i)
            element_j = element(mesh, j)
            Sij, Dij = both_integral_and_integral_gradient(green_functions, element_i, element_j, wavenumber; with_respect_to_first_variable=!direct)
            S[i, j] = -1/4π * Complex(Sij)
            n = direct ? normal(element_j) : normal(element_i)
            D[i, j] = -1/4π * Complex(n' * Dij)
        end
        D[i, i] = D[i, i] + 0.5
    end
    return S, D
end

function assemble_matrices_broadcasting(greens_functions, mesh, wavenumber; direct=true, arrtype=Array, include_identity=true, all_normals=nothing)
    # Broadcasting variant: `arrtype=CuArray` (or another GPU array type) uploads
    # isbits StaticElements and compiles the Green's kernels as a GPU broadcast.
    # Mesh geometry is constant for frequency AD; dropping it from the tape avoids
    # a Zygote crash in `element(::StaticArraysMesh)` vertex indexing.
    elements = ChainRulesCore.ignore_derivatives() do
        arrtype([element(mesh, i) for i in 1:mesh.nfaces])
    end
    return assemble_matrices_broadcasting(greens_functions, elements, wavenumber; direct, include_identity, all_normals)
end

# Autodispatch when the caller already holds a GPU (or other) vector of elements.
# Delhommeau `both_integral` S differs from `integral()` in the near field, so
# S and D stay on those kernels. Rankine CPU assemble uses `assemble_birk_rankine`.
function assemble_matrices_broadcasting(greens_functions, elements::AbstractVector, wavenumber; direct=true, include_identity=true, all_normals=nothing)
    co_elements = reshape(elements, (1, length(elements)))
    S_kernel(e1, e2) = integral(greens_functions, e1, e2, wavenumber)
    S_matrix = (-1 / 4π) .* S_kernel.(elements, co_elements)

    fixed = _fixed_normal(all_normals, real(eltype(S_matrix)))
    _D_normal(e1, e2) = fixed === nothing ? (direct ? normal(e2) : normal(e1)) : fixed
    D_kernel(e1, e2) = _D_normal(e1, e2)' * integral_gradient(greens_functions, e1, e2,
        wavenumber; with_respect_to_first_variable=!direct)
    D_matrix = (-1 / 4π) .* D_kernel.(elements, co_elements)

    T = eltype(D_matrix)
    include_identity || return S_matrix, D_matrix
    if fixed === nothing
        return S_matrix, D_matrix + T(0.5) * I
    end
    jump = [T(0.5) * (fixed' * normal(e)) for e in vec(elements)]
    return S_matrix, D_matrix + Diagonal(jump)
end

# Vendor-agnostic panel-panel assembly. `get_backend(elements)` selects CPU,
# CUDA, Metal, ROC, or the JLArrays dummy GPU from the array type.
@kernel function _assemble_S_D!(S, D, @Const(elements), wavenumber, gfs, direct, include_identity)
    i, j = @index(Global, NTuple)
    @inbounds begin
        e1 = elements[i]
        e2 = elements[j]
        S[i, j] = -(1 / 4π) * integral(gfs, e1, e2, wavenumber)
        if direct
            n = normal(e2)
            grad = integral_gradient(gfs, e1, e2, wavenumber; with_respect_to_first_variable=false)
        else
            n = normal(e1)
            grad = integral_gradient(gfs, e1, e2, wavenumber; with_respect_to_first_variable=true)
        end
        D[i, j] = -(1 / 4π) * (n' * grad)
        if include_identity && i == j
            D[i, j] += convert(eltype(D), 0.5)
        end
    end
end

function assemble_matrices_ka(greens_functions, mesh, wavenumber; direct=true, arrtype=Array, include_identity=true)
    elements = arrtype([element(mesh, i) for i in 1:mesh.nfaces])
    return assemble_matrices_ka(greens_functions, elements, wavenumber; direct, include_identity)
end

function assemble_matrices_ka(greens_functions, elements::AbstractVector, wavenumber; direct=true, include_identity=true)
    backend = KernelAbstractions.get_backend(elements)
    n = length(elements)
    RT = typeof(float(real(wavenumber)))
    T = Complex{RT}
    S = KernelAbstractions.zeros(backend, T, n, n)
    D = KernelAbstractions.zeros(backend, T, n, n)
    kernel = _assemble_S_D!(backend)
    kernel(S, D, elements, wavenumber, greens_functions, direct, include_identity; ndrange=(n, n))
    _ka_synchronize(backend)
    return S, D
end

function _ka_synchronize(backend)
    try
        KernelAbstractions.synchronize(backend)
    catch e
        (e isa MethodError) || rethrow()
    end
end


"""
    assemble_matrices(green_functions, mesh, wavenumber; direct=true)

Assembles the influence matrices based on the tuple of provided Green's functions, mesh, and wavenumber.

# Arguments
- `green_functions`: Iterable of `GreensFunction` objects.
- `mesh`: Floating body mesh with panel information such as vertices, faces, normals, areas etc.
- `wavenumber`: Incoming ocean wavenumber
- `direct=true`: A flag to specify whether to use direct BEM vs Indirect BEM.

# Returns
- A tuple of assembled matrices. S and (D or K) depending on the flag.
"""
# Dispatch on the incoming types. Each backend is a different AD / hardware
# constraint, not a duplicate of the same loop:
#   Mesh (incl. Dual vertices, `all_normals`) → comprehension
#   StaticArraysMesh + `arrtype ≠ Array`      → broadcasting (GPU)
#   AbstractVector of elements                → broadcasting (already uploaded)
#   StaticArraysMesh CPU                      → `_assemble_cpu` on the GF tuple
#     Tuple{Rankine, RankineReflected, GFWu}  → Birk Rankine + Wu centers
#     Tuple{Rankine, RankineReflected}        → Birk Rankine
#     Tuple{GFWu}                             → Wu centers
#     other tuples / vectors                  → partition static vs wave, then
#                                               Birk / Wu / broadcasting
assemble_matrices(green_functions, mesh, wavenumber; kwargs...) =
    assemble_matrices_comprehension(green_functions, mesh, wavenumber; kwargs...)

function assemble_matrices(green_functions, mesh::StaticArraysMesh, wavenumber;
        direct=true, arrtype=Array, all_normals=nothing, include_identity=true)
    if arrtype !== Array
        isnothing(all_normals) || throw(ArgumentError(
            "all_normals is not supported on the GPU assembly path; use arrtype=Array or a dense Mesh"))
        return assemble_matrices_broadcasting(green_functions, mesh, wavenumber; direct, arrtype, include_identity)
    end
    return _assemble_cpu(green_functions, mesh, wavenumber; direct, include_identity, all_normals)
end

assemble_matrices(green_functions, elements::AbstractVector, wavenumber; kwargs...) =
    assemble_matrices_broadcasting(green_functions, elements, wavenumber; kwargs...)

# CPU StaticArraysMesh. Concrete GF tuples skip `partition_greens_functions`.
# Mutating loops here are Enzyme/ForwardDiff-safe; Zygote should use dense Mesh.
# `all_normals` replaces the panel normal in the D/K dot product (forward speed).
function _assemble_cpu(gfs::Tuple{Rankine, RankineReflected, GFWu}, mesh::StaticArraysMesh, wavenumber; direct=true, include_identity=true, all_normals=nothing)
    SD = assemble_birk_rankine((Rankine(), RankineReflected()), mesh; direct, include_identity, all_normals)
    Sw, Dw = assemble_wu_centers(mesh, wavenumber, Val(direct); include_identity=false, all_normals)
    return SD[1] .+ Sw, SD[2] .+ Dw
end

function _assemble_cpu(gfs::Tuple{Rankine, RankineReflected}, mesh::StaticArraysMesh, wavenumber; direct=true, include_identity=true, all_normals=nothing)
    return assemble_birk_rankine(gfs, mesh; direct, include_identity, all_normals)
end

function _assemble_cpu(gfs::Tuple{GFWu}, mesh::StaticArraysMesh, wavenumber; direct=true, include_identity=true, all_normals=nothing)
    return assemble_wu_centers(mesh, wavenumber, Val(direct); include_identity, all_normals)
end

function _assemble_cpu(green_functions, mesh::StaticArraysMesh, wavenumber; direct=true, include_identity=true, all_normals=nothing)
    gfs = green_functions isa Tuple ? green_functions : (green_functions...,)
    static_gfs, wave_gfs = partition_greens_functions(gfs)
    SD_static = isempty(static_gfs) ? nothing : _assemble_static_cpu(static_gfs, mesh; direct, include_identity, all_normals)
    if isempty(wave_gfs)
        return SD_static
    end
    wave_identity = isnothing(SD_static) && include_identity
    Sw, Dw = _assemble_wave_cpu(wave_gfs, mesh, wavenumber; direct, include_identity=wave_identity, all_normals)
    isnothing(SD_static) && return Sw, Dw
    return SD_static[1] .+ Sw, SD_static[2] .+ Dw
end

function _assemble_static_cpu(gfs::Tuple, mesh::StaticArraysMesh; direct=true, include_identity=true, all_normals=nothing)
    _birk_rankine_tuple(gfs) || return assemble_matrices_broadcasting(gfs, mesh, 0.0; direct, include_identity, all_normals)
    return assemble_birk_rankine(gfs, mesh; direct, include_identity, all_normals)
end

function _assemble_wave_cpu(gfs::Tuple{GFWu}, mesh::StaticArraysMesh, wavenumber; direct=true, include_identity=true, all_normals=nothing)
    return assemble_wu_centers(mesh, wavenumber, Val(direct); include_identity, all_normals)
end

function _assemble_wave_cpu(gfs::Tuple, mesh::StaticArraysMesh, wavenumber; direct=true, include_identity=true, all_normals=nothing)
    length(gfs) == 1 && gfs[1] isa GFWu && return assemble_wu_centers(mesh, wavenumber, Val(direct); include_identity, all_normals)
    return assemble_matrices_broadcasting(gfs, mesh, wavenumber; direct, include_identity, all_normals)
end

# Fixed replacement normal (forward speed uses x̂). `nothing` keeps panel normals.
_fixed_normal(::Nothing, ::Type) = nothing
_fixed_normal(v, ::Type{T}) where {T} = SVector{3, T}(v[1], v[2], v[3])

@inline _panel_or_fixed(::Nothing, direct::Bool, n2, normals, i) =
    direct ? n2 : normals[i]
@inline _panel_or_fixed(fixed::SVector, direct::Bool, n2, normals, i) = fixed

# Diagonal jump `constant * (n' * norm_vec)` from the comprehension path: 1/2
# with panel normals, 1/2 (a' nᵢ) with a fixed normal (submerged panels).
function _add_identity!(D, ::Nothing, normals, half)
    @inbounds for i in axes(D, 1)
        D[i, i] += half
    end
    return D
end
function _add_identity!(D, fixed::SVector, normals, half)
    @inbounds for i in axes(D, 1)
        D[i, i] += half * (fixed' * normals[i])
    end
    return D
end

_birk_rankine_tuple(gfs::Tuple) = all(_is_birk_rankine, gfs)
_is_birk_rankine(::Rankine) = true
_is_birk_rankine(::RankineReflected) = true
_is_birk_rankine(::RankineReflectedNegative) = true
_is_birk_rankine(::GreensFunction) = false

@inline function _face_smatrix(mesh::StaticArraysMesh, j)
    f = mesh.faces[j]
    v1 = mesh.vertices[f[1]]
    v2 = mesh.vertices[f[2]]
    v3 = mesh.vertices[f[3]]
    v4 = mesh.vertices[f[4]]
    T = eltype(v1)
    return SMatrix{4,3,T,12}(
        v1[1], v2[1], v3[1], v4[1],
        v1[2], v2[2], v3[2], v4[2],
        v1[3], v2[3], v3[3], v4[3],
    )
end

@inline _reflect_z(c::StaticVector{3}) = typeof(c)(c[1], c[2], -c[3])

@inline function _source_birk_geom(mesh::StaticArraysMesh, j)
    Tmat, qgc, local_corners, _ = birk_panel_geometry(_face_smatrix(mesh, j))
    return (T=Tmat, qgc=qgc, local_corners=local_corners)
end

function assemble_birk_rankine(gfs::Tuple, mesh::StaticArraysMesh; direct=true, include_identity=true, all_normals=nothing)
    n = mesh.nfaces
    centers = mesh.centers
    normals = mesh.normals
    areas = mesh.areas
    radii = mesh.radii
    need_R = any(gf -> gf isa Rankine, gfs)
    need_I = any(gf -> gf isa RankineReflected, gfs)
    need_N = any(gf -> gf isa RankineReflectedNegative, gfs)
    image = need_I | need_N

    GT = eltype(eltype(centers))
    fixed = _fixed_normal(all_normals, GT)
    scale = convert(GT, -1 / 4π)
    image_sign = convert(GT, (need_I ? 1 : 0) + (need_N ? -1 : 0))
    S = Matrix{Complex{GT}}(undef, n, n)
    D = Matrix{Complex{GT}}(undef, n, n)
    geoms = [_source_birk_geom(mesh, j) for j in 1:n]
    z = zero(GT)
    half = convert(GT, 0.5)
    @inbounds for j in 1:n
        c2 = centers[j]
        r2 = radii[j]
        a2 = areas[j]
        n2 = normals[j]
        geom = geoms[j]
        Tmat, qgc, lc = geom.T, geom.qgc, geom.local_corners
        for i in 1:n
            c1 = centers[i]
            nvec = _panel_or_fixed(fixed, direct, n2, normals, i)
            s = z
            d = z
            if need_R
                φ, gsrc = _vrankine_from_geom(c1, c2, r2, a2, Tmat, qgc, lc)
                g = direct ? gsrc : -gsrc
                s += φ
                d += nvec' * _as_center_type(c1, g)
            end
            if image
                c1r = _reflect_z(c1)
                φr, gsrcr = _vrankine_from_geom(c1r, c2, r2, a2, Tmat, qgc, lc)
                if direct
                    gr = _as_center_type(c1r, gsrcr)
                else
                    gr = vertical_reflection(_as_center_type(c1r, -gsrcr))
                end
                s += image_sign * φr
                d += image_sign * (nvec' * gr)
            end
            S[i, j] = scale * s
            D[i, j] = scale * d
        end
    end
    if include_identity
        _add_identity!(D, fixed, normals, half)
    end
    return S, D
end

function assemble_wu_centers(mesh::StaticArraysMesh, wavenumber; direct=true, include_identity=true, all_normals=nothing)
    assemble_wu_centers(mesh, wavenumber, Val(direct); include_identity, all_normals)
end

function assemble_wu_centers(mesh::StaticArraysMesh, wavenumber, ::Val{direct}; include_identity=true, all_normals=nothing) where {direct}
    _assemble_wu_centers_loop(mesh, wavenumber, Val(direct), include_identity, all_normals)
end

# Fused S/D primal. No assemble rrule: ForwardDiff Duals and Enzyme/Mooncake
# reverse trace this mutating loop (k and geometry). Zygote cannot.
function _assemble_wu_centers_loop(mesh, k, ::Val{direct}, include_identity::Bool, all_normals=nothing) where {direct}
    n = mesh.nfaces
    centers = mesh.centers
    areas = mesh.areas
    normals = mesh.normals
    RT = promote_type(typeof(float(real(k))), eltype(areas))
    T = Complex{RT}
    fixed = _fixed_normal(all_normals, RT)
    scale = convert(T, -1 / 4π)
    S = Matrix{T}(undef, n, n)
    D = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n
        c2 = centers[j]
        a2 = areas[j]
        n2 = normals[j]
        for i in 1:n
            nvec = _panel_or_fixed(fixed, direct, n2, normals, i)
            sij, dij = _wu_sd_centers(centers[i], c2, nvec, a2, k, Val(direct))
            S[i, j] = scale * sij
            D[i, j] = scale * dij
        end
    end
    if include_identity
        _add_identity!(D, fixed, normals, convert(T, 0.5))
    end
    return S, D
end


function assemble_matrix_wu(mesh, wavenumber; direct=true, all_normals=nothing)
    return assemble_matrices((Rankine(), RankineReflected(), GFWu()), mesh, wavenumber; direct, all_normals)
end


# Array matvec is a comprehension so Enzyme reverse does not enter BLAS `*`.
# Reverse VJP is in ReverseAD (same as Zygote: ȳ bc', S' ȳ). Vector and
# multi-RHS (N×k) share the same formula.
function _mulvec(S::Array, bc::AbstractVector)
    n = length(bc)
    T = promote_type(eltype(S), eltype(bc))
    return T[sum(S[i, j] * bc[j] for j in 1:n) for i in 1:n]
end
function _mulvec(S::Array, bc::AbstractMatrix)
    n, k = size(S, 1), size(bc, 2)
    m = size(S, 2)
    T = promote_type(eltype(S), eltype(bc))
    return T[sum(S[i, j] * bc[j, c] for j in 1:m) for i in 1:n, c in 1:k]
end
_mulvec(S, bc) = S * bc

# Dual numbers use native `\` (ForwardDiff). CPU Array uses LinearSolve
# LUFactorization. Reverse of that solve is the IFT in ReverseAD (same VJP
# as Zygote; adjoint `y'`). GPU arrays use `gpu_linsolve`.
function linsolve(A, b)
    if eltype(real(A)) <: ForwardDiff.Dual
        return A \ b
    elseif A isa Array
        return _linearsolve(A, b)
    else
        return gpu_linsolve(A, b)
    end
end

# Primal LinearSolve. Use the returned `sol.u` (SciML/LinearSolve.jl#479).
# Enzyme/Zygote never differentiate this body: ReverseAD intercepts
# `_linearsolve` and runs IFT with primal solves only.
# Pass `u0` with `b`'s shape: LinearSolve's default is `similar(b, size(A,2))`,
# which collapses an N×k RHS to a Vector and then `ldiv!` fails.
function _linearsolve(A, b)
    T = promote_type(eltype(A), eltype(b))
    u0 = similar(b, T)
    sol = LinearSolve.solve(LinearSolve.LinearProblem(A, b; u0), LinearSolve.LUFactorization())
    return sol.u
end

gpu_linsolve(A, b) = _backend_ldiv(A, b)

# Prefer native `\` (cuSOLVER). Dummy GPU arrays (JLArray) and some Metal
# types hit generic LAPACK via scalar indexing, so factorize on the host.
function _backend_ldiv(A, b)
    if A isa Array || _uses_device_ldiv(A)
        return A \ b
    end
    x = Array(A) \ Array(b)
    out = similar(b, eltype(x), size(x))
    copyto!(out, x)
    return out
end

function _uses_device_ldiv(A)
    return string(nameof(typeof(A))) == "CuArray"
end

function solve(D, S, bc; direct::Bool=true)
    # Keep an N×k (matrix) RHS intact so `_mulvec(::Array, ::AbstractMatrix)`
    # and LinearSolve's `u0` see the correct shape; only flatten other shapes.
    b = bc isa AbstractMatrix ? bc : vec(bc)
    if direct
        ϕ = linsolve(D, _mulvec(S, b))
        sources = nothing
    else
        K = D
        sources = linsolve(K, b)
        ϕ = _mulvec(S, sources)
    end
    return ϕ, sources
end

