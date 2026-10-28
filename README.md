![Test Status](https://github.com/JuliaOceanWaves/MarineHydro.jl/actions/workflows/run_tests.yml/badge.svg)
![GitHub](https://img.shields.io/github/license/JuliaOceanWaves/MarineHydro.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19859297.svg)](https://doi.org/10.5281/zenodo.19859297)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

# MarineHydro.jl

Differentiable boundary-element solver for marine hydrodynamics: radiation and diffraction, direct and indirect formulations, Wu and Delhommeau Green functions.

The user-facing API is mostly:

`problem` + `formulation` → `solve` → real coefficients → [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)



> **The public API has changed since the paper.** The article used `calculate_radiation_forces`, dense `Mesh`, and Zygote `rrule`s on Capytaine arrays (`paper/MeshGradients_singlebody.jl`). That is **not** the current interface. Use `FloatingBody` + `solve(prob, formulation)` + `DifferentiationInterface` as in the [tutorial](#marinehydrojl-tutorial) below. Scripts under `[paper/](paper/)` still reproduce the published figures. An LLM assisted the rewrite of the user-facing API.

## Citation

Khanal, K., Michelén Ströfer, C. A., Ancellin, M., & Haji, M. N. (2025). Fully differentiable boundary element solver for hydrodynamic sensitivity analysis of wave-structure interactions. *Applied Ocean Research*, 163, 104707. [doi:10.1016/j.apor.2025.104707](https://doi.org/10.1016/j.apor.2025.104707) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0141118725002937) · [arXiv:2501.06988](https://arxiv.org/abs/2501.06988) · [PDF](paper/Khanal_et_al_2025.pdf)

```bibtex
@article{khanal2025marinehydro,
  title   = {Fully differentiable boundary element solver for hydrodynamic sensitivity analysis of wave-structure interactions},
  author  = {Khanal, Kapil and Michel{\'e}n Str{\"o}fer, Carlos A. and Ancellin, Matthieu and Haji, Maha N.},
  journal = {Applied Ocean Research},
  volume  = {163},
  pages   = {104707},
  year    = {2025},
  doi     = {10.1016/j.apor.2025.104707}
}
```

## Install

Julia from [julialang.org](https://julialang.org/downloads/). Capytaine must be importable from the Python that [PyCall](https://github.com/JuliaPy/PyCall.jl) uses.

```bash
git clone https://github.com/JuliaOceanWaves/MarineHydro.jl.git
cd MarineHydro.jl
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
# ENV["PYTHON"] = "/path/to/python"   # `which python` if PyCall needs a Capytaine env
```

---

# MarineHydro.jl tutorial

The same walkthrough is in [`mh_tutorial.ipynb`](mh_tutorial.ipynb).

- **Radiation:** `RadiationProblem(body, ω)` → `added_mass` / `radiation_damping`
- **Diffraction:** `DiffractionProblem(body, ω; beta)` → `diffraction_force` / `froude_krylov_force` / `excitation_force`
- **Many ω:** `hydrodynamic_coefficients(body, params, formulation)`
- **Formulation:** `DirectBEM()` / `IndirectBEM()`, wave Green `GFWu` or `ExactGuevelDelhommeau`

AD engines (same named `f`):

- `AutoForwardDiff()` — forward (Duals)
- `AutoEnzyme` reverse — adjoint. Enzyme is reverse-only here (forward hits Python).

Capytaine never sees Duals. `fd_mesh_rules!(sphere_smesh)` finite-differences the mesh. Topology (panel count) must stay fixed as radius changes.

```julia
using MarineHydro
using PyCall
using StaticArrays
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme

cpt = pyimport("capytaine")

function sphere_smesh(r)
    StaticArraysMesh(cpt.mesh_sphere(
        name="sphere", radius=r, center=(0.0, 0.0, 0.0),
        resolution=(6, 6)).immersed_part())
end

fd_mesh_rules!(sphere_smesh) #to write the rrules for geometry

sphere_body(r) = FloatingBody(sphere_smesh(r), [:Heave], "sphere")

const formulation = DirectBEM()   # default: direct + Wu
const r0 = 1.0
const ω0 = 1.03
const β0 = 0.0
const ωs = [0.8, 1.03, 1.4]
const fwd = AutoForwardDiff()
const rev = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
```

## 1. One radiation solve

`DirectBEM()` holds the Green functions. `RadiationProblem(body, ω)` fills in the wavenumber and the single radiating dof.

```julia
radiation_solve(r, ω, form=formulation) = solve(RadiationProblem(sphere_body(r), ω), form)

sol = radiation_solve(r0, ω0)
added_mass(sol).Heave
radiation_damping(sol).Heave
```

## 2. DifferentiationInterface — the simple pattern

One active argument. Freeze the rest as `const`. Same `f` for ForwardDiff and Enzyme.

```julia
added_mass_r(r) = added_mass(radiation_solve(r, ω0)).Heave
added_mass_ω(ω) = added_mass(radiation_solve(r0, ω)).Heave

DI.derivative(added_mass_r, fwd, r0)
DI.derivative(added_mass_r, rev, r0)   # first Enzyme call compiles
DI.derivative(added_mass_ω, fwd, ω0)
DI.derivative(added_mass_ω, rev, ω0)
```

## 3. Diffraction

`DiffractionProblem(body, ω; beta)`. Outputs are complex forces; AD needs Re or Im.

```julia
diffraction_solve(r, ω, form=formulation) =
    solve(DiffractionProblem(sphere_body(r), ω; beta=β0), form)

dsol = diffraction_solve(r0, ω0)
diffraction_force(dsol).Heave
froude_krylov_force(dsol).Heave
excitation_force(dsol).Heave

Fex_r(r) = real(excitation_force(diffraction_solve(r, ω0)).Heave)
Fex_ω(ω) = real(excitation_force(diffraction_solve(r0, ω)).Heave)

DI.derivative(Fex_r, fwd, r0)
DI.derivative(Fex_r, rev, r0)
DI.derivative(Fex_ω, fwd, ω0)
DI.derivative(Fex_ω, rev, ω0)
```

## 4. Many frequencies

`hydrodynamic_coefficients` builds the problems and returns a NamedTuple of real arrays (the differentiable payload). Label with `create_DimStack` *after* AD, not through it.

```julia
params_rad = (wave_frequencies=ωs, radiating_dofs=[:Heave])
params_all = (wave_frequencies=ωs, radiating_dofs=[:Heave], wave_directions=[β0])

data = hydrodynamic_coefficients(sphere_body(r0), params_all, formulation)
vec(data.added_mass)
vec(data.radiation_damping)
vec(data.excitation_force)
```

Jacobian of the whole \(A(\omega)\) curve:

- wrt radius (one geometry parameter → \(A\) at each \(\omega\)): `jacobian` on \(p = [r]\)
- wrt the frequency vector: `jacobian` of \(A(\omega_s)\) w.r.t. \(\omega_s\)
- Enzyme reverse on a scalar reduction (`sum(A)`), then `gradient` on \(\omega_s\)

```julia
A_of_r(r) = vec(hydrodynamic_coefficients(sphere_body(r), params_rad, formulation).added_mass)
A_of_ωs(ωvec) = vec(hydrodynamic_coefficients(sphere_body(r0),
    (wave_frequencies=ωvec, radiating_dofs=[:Heave]), formulation).added_mass)
sumA_r(r) = sum(A_of_r(r))
sumA_ωs(ωvec) = sum(A_of_ωs(ωvec))

DI.jacobian(p -> A_of_r(p[1]), fwd, [r0])     # size (nω, 1)
DI.jacobian(A_of_ωs, fwd, copy(ωs))           # size (nω, nω)
DI.derivative(sumA_r, rev, r0)                # Enzyme: d(∑A)/dr
DI.gradient(sumA_ωs, rev, copy(ωs))           # Enzyme: d(∑A)/dωᵢ
```

## 5. Switch Green function and direct / indirect

The formulation is the method. Same `solve(prob, formulation)` / `hydrodynamic_coefficients(..., formulation)`.

```julia
formulations = (
    "direct Wu"           => DirectBEM(),
    "indirect Wu"         => IndirectBEM(),
    "direct Delhommeau"   => DirectBEM("ExactGuevelDelhommeau"),
    "indirect Delhommeau" => IndirectBEM("ExactGuevelDelhommeau"),
)

for (name, form) in formulations
    A = added_mass(radiation_solve(r0, ω0, form)).Heave
    B = radiation_damping(radiation_solve(r0, ω0, form)).Heave
    Fe = real(excitation_force(diffraction_solve(r0, ω0, form)).Heave)
    # ForwardDiff through each formulation:
    dAdr = DI.derivative(r -> added_mass(radiation_solve(r, ω0, form)).Heave, fwd, r0)
end
```

Enzyme reverse is shown on the default `DirectBEM()` only — each new `formulation` is a fresh Enzyme compile.

```julia
fwd = AutoForwardDiff()
rev = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
f(r) = added_mass(solve(RadiationProblem(sphere_body(r), ω0), DirectBEM())).Heave
DI.derivative(f, fwd, r0)
DI.derivative(f, rev, r0)
```


|                              |                                                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Green function               | `DirectBEM()` Wu, `DirectBEM("ExactGuevelDelhommeau")`                                                      |
| Formulation                  | `DirectBEM` vs `IndirectBEM`                                                                                |
| Many \omega                  | `hydrodynamic_coefficients(body, (wave_frequencies=ωs, radiating_dofs=[:Heave], wave_directions=[β]), formulation)` |
| Scalar r,\omega              | `DI.derivative`                                                                                             |
| Vector \omega_s or A(\omega) | `DI.gradient` / `DI.jacobian`                                                                               |
| Label axes                   | `create_DimStack(data, params, body)` after AD                                                              |


---

## Paper reproduction

The [paper PDF](paper/Khanal_et_al_2025.pdf) and scripts in `[paper/](paper/)` use the **pre-refactor** surface (`calculate_radiation_forces`, Zygote mesh `rrule`s). Use them to reproduce figures, not as the current public API.

## Layout

- `src/` — package
- `test/` — primal and AD tests
- `mh_tutorial.ipynb` — current public API walkthrough
- `paper/` — article PDF, scripts, and data for the paper

## License

[MIT](LICENSE) (code). The article is available under [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/).