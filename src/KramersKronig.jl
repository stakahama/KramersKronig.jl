#=
================================================================================

Kramers Kronig relations

adapted from
  Lucarini, V., J.J. Saarinen, K.-E. Peipon, and E.M. Vartiainen. 2005.
  Kramers–Kronig Relations in Optical Materials Research.
  Berlin: Springer. doi:10.1007/b138913.
and Mathworks File Exchange
  https://www.mathworks.com/matlabcentral/fileexchange/8135-tools-for-data-analysis-in-optics-acoustics-signal-processing
and also
   Ohta and Ishida, Appl. Spec., doi:10.1366/0003702884430380, 1988.

Satoshi Takahama (satoshi.takahama@gmail.com)

Distributed under GNU GPL v3.

Example usage (as local file in same directory as script):
  include("./KramersKronig.jl")
  using .KramersKronig

================================================================================
=#

module KramersKronig

export kkim, kkre, sskkim, sskkre, kkre_trapz, integrate, Maclaurin, Rectangle

## -----------------------------------------------------------------------------

## For method dispatch

abstract type KKIntegrator end

struct Maclaurin <: KKIntegrator end
struct Rectangle <: KKIntegrator end

## -----------------------------------------------------------------------------

## Integrate functions

"""
    function integrate(f::Function, N::Integer, ...)

Calculate Cauchy Principal Value of integrand `f` from points 1 to `N` along discrete spectrum using one of the provided integration methods. `f` is a closure to which data vectors are attached, and the function arguments are two indices - e.g., `i` and `j`, where `i` is the index at which the integrand is to be evaluated, and `j` is the index over which the summation is to be performed.

### Parameters
- `f::Function`: integrand to integrate
- `N::Integer`: maximum number of points in spectrum vector
- `...`: type of `struct` determines integration algorithm

### Returns
- `Vector{<:Float64}`: spectrum vector
"""
function integrate end

"""
    function integrate(f::Function, N::Integer, ::Maclaurin)::Vector{<:Float64}

Calculate Cauchy Principal Value of integrand `f` over spectrum points 1 to `N` using Maclaurin's formula (Ohta and Ishida, doi:10.1366/0003702884430380, 1988).
"""
function integrate(f::Function, N::Integer, ::Maclaurin)::Vector{<:Float64}
    pv(i) = 2 * sum(f.(i, range(iseven(i) ? 1 : 2, N, step=2)))
    return pv.(1:N)
end


"""
    function integrate(f::Function, N::Integer, ::Rectangle)::Vector{<:Float64}

Calculate Cauchy Principal Value of integrand `f` over spectrum points 1 to `N` using rectangle formula (Lucarini et al., doi:10.1007/b138913, 2005).
"""
function integrate(f::Function, N::Integer, ::Rectangle)::Vector{<:Float64}
    pv(i) = sum(f.(i, 1:i-1); init=0.) + sum(f.(i, i+1:N); init=0.)
    return pv.(1:N)
end

## -----------------------------------------------------------------------------

## KK functions

"""
    function kkre(ω::AbstractVector{<:Real}, Imχ::AbstractVector{<:Real}; alg::KKIntegrator=Maclaurin())::Vector{<:Float64}

Calculate real part from imaginary part of a vector variable using Kramers-Kronig relations.

### Parameters
- `ω::AbstractVector{<:Real}`: frequency or wavenumber
- `Imχ::AbstractVector{<:Real}`: spectrum of imaginary part (real values)
- `alg::KKIntegrator`: integration algorithm [either `Maclaurin()` (default) or `Rectangle()`]

### Returns
- `Vector{<:Float64}`: spectrum vector
"""
function kkre(ω::AbstractVector{<:Real}, Imχ::AbstractVector{<:Real}; alg::KKIntegrator=Maclaurin())::Vector{<:Float64}
    N = length(ω)
    dω = abs(only(diff(ω[1:2])))
    integrand(j, k) = ω[k] * Imχ[k] / (ω[k]^2 - ω[j]^2)
    spectrum = dω .* integrate(integrand, N, alg)
    Reχ = similar(spectrum)
    @. Reχ = 2 / π * spectrum
    return Reχ
end

"""
    function kkim(ω::AbstractVector{<:Real}, Reχ::AbstractVector{<:Real}; alg::KKIntegrator=Maclaurin())::Vector{<:Float64}

Calculate imaginary part from real part of a vector variable using Kramers-Kronig relations.

### Parameters
- `ω::AbstractVector{<:Real}`: frequency or wavenumber
- `Reχ::AbstractVector{<:Real}`: spectrum of real part
- `alg::KKIntegrator`: integration algorithm [either `Maclaurin()` (default) or `Rectangle()`]

### Returns
- `Vector{<:Float64}`: spectrum vector
"""
function kkim(ω::AbstractVector{<:Real}, Reχ::AbstractVector{<:Real}; alg::KKIntegrator=Maclaurin())::Vector{<:Float64}
    N = length(ω)
    dω = abs(only(diff(ω[1:2])))
    integrand(j, k) = Reχ[k] / (ω[k]^2 - ω[j]^2)
    spectrum = dω .* integrate(integrand, N, alg)
    Imχ = similar(spectrum)
    @. Imχ = -2 / π * ω * spectrum
    return Imχ
end

"""
    function kkre_trapz(ω::AbstractVector{<:Real}, Imχ::AbstractVector{<:Real})::Vector{<:Float64}

Calculate real part from imaginary part of a vector variable using Kramers-Kronig relations with trapezium integration.

### Parameters
- `ω::AbstractVector{<:Real}`: frequency or wavenumber
- `Imχ::AbstractVector{<:Real}`: spectrum of imaginary part (real values)
- `alg::KKIntegrator`: integration algorithm [either `Maclaurin()` (default) or `Rectangle()`]

### Returns
- `Vector{<:Float64}`: spectrum vector
"""
function kkre_trapz(ω::AbstractVector{<:Real}, Imχ::AbstractVector{<:Real})::Vector{<:Float64}
    N = length(ω)
    dω = abs(only(diff(ω[1:2])))
    integrand(j, k) = ω[k] * Imχ[k] / (ω[k]^2 - ω[j]^2)    
    f(i) = integrand.(i, filter(j -> j != i, 1:N))
    dImχ(i) = if i in (2, N-1)
        (Imχ[i+1] - Imχ[i-1]) / (2 * dω)
    elseif i in (3, N-2)
        (8 * (Imχ[i+1] - Imχ[i-1]) - (Imχ[i+2] - Imχ[i-2])) / (12 * dω)
    elseif i in range(1+3, N-3)
        (45 * (Imχ[i+1] - Imχ[i-1]) -
            9 * (Imχ[i+2] - Imχ[i-2]) +
            (Imχ[i+3] - Imχ[i-3])) / (60 * dω)
    elseif i==1
        (4 * (Imχ[2] - Imχ[1]) - (Imχ[3] - Imχ[1])) / (2 * dω)
    elseif i==N
        (4 * (Imχ[N] - Imχ[N-1]) - (Imχ[N] - Imχ[N-2])) / (2 * dω)        
    else
        0
    end
    Ic(i) = 2 / π * dω * sum(f(i))
    Ip(i) = 1 / π * dω * Imχ[i] / (2 * ω[i])
    Im(i) = 1 / π * dω * dImχ(i)
    pv(i) = Ic(i) + Ip(i) + Im(i)
    return pv.(1:N)
end

## -----------------------------------------------------------------------------

## Singly-subtractive KK functions

"""
    function sskkre(ω::AbstractVector{<:Real}, Imχ::AbstractVector{<:Real}, ω1::Real, Imχ1::Real; alg::KKIntegrator=Maclaurin())::Vector{<:Float64}

Calculate real part from imaginary part of a vector variable using Kramers-Kronig relations using one anchor point.

### Parameters
- `ω::AbstractVector{<:Real}`: frequency or wavenumber
- `Imχ::AbstractVector{<:Real}`: spectrum of imaginary part (real values)
- `ω1::Real`: frequency or wavenumber at anchor point
- `Imχ1::Real`: spectrum value at anchor point
- `alg::KKIntegrator`: integration algorithm [either `Maclaurin()` (default) or `Rectangle()`]

### Returns
- `Vector{<:Float64}`: spectrum vector
"""
function sskkre(ω::AbstractVector{<:Real}, Imχ::AbstractVector{<:Real}, ω1::Real, Imχ1::Real; alg::KKIntegrator=Maclaurin())::Vector{<:Float64}
    k = argmin((ω .- ω1).^2)
    Reχ = kkre(ω, Imχ; alg=alg)
    return Reχ .+ (Reχ1 .- Reχ[k])
end

"""
    function sskkim(ω::AbstractVector{<:Real}, Reχ::AbstractVector{<:Real}, ω1::Real, Reχ1::Real; alg::KKIntegrator=Maclaurin())::Vector{<:Float64}

Calculate imaginary part from real part of a vector variable using Kramers-Kronig relations using one anchor point.

### Parameters
- `ω::AbstractVector{<:Real}`: frequency or wavenumber
- `Reχ::AbstractVector{<:Real}`: spectrum of real part
- `ω1::Real`: frequency or wavenumber at anchor point
- `Reχ1::Real`: spectrum value at anchor point
- `alg::KKIntegrator`: integration algorithm [either `Maclaurin()` (default) or `Rectangle()`]

### Returns
- `Vector{<:Float64}`: spectrum vector
"""
function sskkim(ω::AbstractVector{<:Real}, Reχ::AbstractVector{<:Real}, ω1::Real, Reχ1::Real; alg::KKIntegrator=Maclaurin())::Vector{<:Float64}
    k = argmin((ω .- ω1).^2)
    Imχ = kkim(ω, Reχ; alg=alg)
    return Imχ .+ ω1.^(-1) .* ω.* (Imχ1 .- Imχ[k])
end

## -----------------------------------------------------------------------------

end
