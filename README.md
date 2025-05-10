# KramersKronig.jl


[![Build
Status](https://github.com/stakahama/KramersKronig.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stakahama/KramersKronig.jl/actions/workflows/CI.yml?query=branch%3Amain)

This module provides functions for calculating the real part from the
imaginary part of a spectrum (vector) and vice versa. The functions
implement evaluation of the Cauchy Principal Value of integrals arising
from Kramers-Kronig relations along presampled points of a digitized
spectrum.

The functions and algorithms are adapted from

- Lucarini, V., J.J. Saarinen, K.-E. Peipon, and E.M. Vartiainen. 2005.
  Kramers–Kronig Relations in Optical Materials Research. Berlin:
  Springer. doi:[10.1007/b138913](https://doi.org/10.1007/b138913).
- Mathworks File Exchange
  [id=8135](https://www.mathworks.com/matlabcentral/fileexchange/8135-tools-for-data-analysis-in-optics-acoustics-signal-processing).
- Ohta and Ishida, Appl. Spec.,
  doi:[10.1366/0003702884430380](https://doi.org/10.1366/0003702884430380),
  1988.

## Installation

``` julia
using Pkg
Pkg.add(url="https://github.com/stakahama/KramersKronig.jl")
```

## Example usage

``` julia
using KramersKronig
using Plots
```

Lorentzian line shape function (peak-height-normalized form).

``` julia
x₀ = 1000
Γ = 20
L(x) = 1 / (1 + ((x - x₀) / (Γ / 2))^2)

x = range((@. x₀ + 20 * Γ * [-1, 1])..., length = 1000)
curve = L.(x)
```

Find complementary (real) part.

``` julia
y1 = kkre(x, curve) # default: alg=McLaurin()
y2 = kkre(x, curve; alg=Rectangle())
y3 = kkre_trapz(x, curve)
```

``` julia
plot()
plot!(x, curve, label="Im")
plot!(x, y1, label="Re - McLaurin")
plot!(x, y2, label="Re - Rectangle")
plot!(x, y3, label="Re - Trapezium", linestyle=:dash)
plot!(size=(400, 250))
```

![](README_files/figure-commonmark/cell-5-output-1.svg)

## Notes

KramersKronig.jl provides a vector and return the complementary vector
using algorithms evaluated for this purpose. Related libraries that
could be further adapted for the problem:

- [NumericalIntegration.jl](https://github.com/JuliaMath/NumericalIntegration.jl):
  integration of presampled data
- [Integrals.jl](https://github.com/SciML/Integrals.jl): integration of
  analytical functions
- [other
  alternatives](https://discourse.julialang.org/t/numerical-integration-of-cauchy-principal-value/36059)
