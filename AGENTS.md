# MarineHydro.jl

Julia BEM solver for wave–structure hydrodynamics. Setup and usage for the package itself are in `README.md` and `.github/workflows/run_tests.yml`.

## Cdev info

This is a library, not a server. There is no `dev` process to start. Development is a Julia REPL / script session with the root project and a Capytaine-enabled Python interpreter wired through PyCall.

### Runtime layout

- Julia is installed via `juliaup`. Default channel is `release` (currently 1.12.x) so it matches the committed root `Manifest.toml` on this branch (`julia_version = "1.12.2"`).
- Capytaine `2.2.1` lives in `$HOME/.capy-venv` (not in the repo). Point PyCall at it with `PYTHON=$HOME/.capy-venv/bin/python` **before** any `Pkg.build("PyCall")` / `Pkg.instantiate()` / test run.
- After changing `PYTHON`, rebuild PyCall: `PYTHON=$HOME/.capy-venv/bin/python julia --project=. -e 'using Pkg; Pkg.build("PyCall")'`. A stale PyCall build silently uses the wrong interpreter.

### Commands

Standard commands are in `README.md` and `Project.toml`. In this environment:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
export PYTHON="$HOME/.capy-venv/bin/python"
julia --project=.                         # REPL / scripts
julia --project=. -e 'using Pkg; Pkg.test()'
```

There is no dedicated lint or formatter config.

### Testing caveats

- `Pkg.test()` on Julia 1.12 segfaults inside Enzyme (signal 11) in `test/greens_function_differentiation.jl` and `test/matrix_assembly_differentiation.jl`. This is the same failure as current GitHub Actions `Julia 1 / GPU` on `main`.
- Core physics + Zygote/ForwardDiff tests run from the root project and pass on Julia 1.12:

```bash
PYTHON=$HOME/.capy-venv/bin/python julia --project=. -e '
using Test, MarineHydro
@testset "core physics" begin
    include("test/consistency_with_Capytaine.jl")
    include("test/greens_function.jl")
    include("test/rankine_vectorized.jl")
    include("test/matrix_assembly.jl")
    include("test/consistency_with_analytical_solutions.jl")
end
@testset "zygote ad" begin
    include("test/outputs_differentiation.jl")
end
'
```

- `origin/main` gitignores `/Manifest.toml` and `/test/Manifest.toml` and tests Julia 1.10 for Enzyme-heavy jobs. This branch still commits those manifests (1.12 root, 1.10 test). Instantiating the committed root manifest with Julia 1.10 fails (`JuliaSyntaxHighlighting` / “can not merge projects”). Do not delete the committed manifests unless you are intentionally matching `main`’s resolve-per-version workflow.
- GPU tests (`test/gpu.jl`, `test/gpu_cpu_smoke.jl`) are optional and skip when CUDA/Metal are missing.
- `paper/` and `benchmark/` have their own environments; they are not required for everyday package work.

### Hello-world check

```julia
using MarineHydro, PyCall, Zygote
cpt = pyimport("capytaine")
cptmesh = cpt.mesh_sphere(name="sphere", radius=1.0, center=(0,0,0), resolution=(10,10))
cptmesh.keep_immersed_part(inplace=true)
mesh = Mesh(cptmesh)
ω = 1.03; ζ = [0,0,1]
F = DiffractionForce(mesh, ω, ζ)
A, B = calculate_radiation_forces(mesh, ζ, ω)
A_w_grad, = Zygote.gradient(w -> calculate_radiation_forces(mesh,ζ,w)[1], ω)
```

Expect a complex diffraction force, real added mass / damping, and a real `dA/dω`.
