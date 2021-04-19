using Distributed
using BenchmarkTools
using LinearAlgebra
using Trapz
using DifferentialEquations
using Base.Threads
include("ode_amherst_cyling.jl")
include("utils/general_ode_functions.jl")

Γ = 2pi*1.6e6
Ω1 = Γ
Ω2 = Γ
Δ1 = 0.
Δ2 = 0.
νp1 = 1e6
νp2 = 1e6
params = ones(100)*2.8e6

p = [Ω1, νp1, Δ1, Ω2, νp2, Δ2]

ρ = zeros(ComplexF64, 16, 16)
ρ[1,1] = 1.
tspan = (0,300e-6)

du = zeros(ComplexF64, 16, 16)


function prob_func(prob,i,repeat)
    remake(prob,p=[Ω1; params[i]; Δ1; Ω2; params[i]; Δ2])
end

function output_func(sol)
    return trapz(sol.t, [real(sum(diag(sol.u[j])[13:end])) for j in 1:size(sol.u)[1]])
end

function ode_routine(ν, Ω1, Δ1, Ω2, Δ2)
    p = [Ω1; ν; Δ1; Ω2; ν; Δ2]
    prob = ODEProblem(Lindblad_rhs!,ρ,tspan,p)
    sol = solve(prob, Tsit5(), abstol = 1e-9, reltol = 1e-6, dt = 1e-9)
    return output_func(sol)
end

function threaded(params, Ω1, Δ1, Ω2, Δ2)
    results = zeros(size(params)[1])
    ν = 0.
    @inbounds begin
        @threads for i in 1:size(params)[1]
            ν = params[i]
            results[i] = ode_routine(ν, Ω1, Δ1, Ω2, Δ2)
        end
    end
end

threaded(params, Ω1, Δ1, Ω2, Δ2)

@benchmark threaded(params, Ω1, Δ1, Ω2, Δ2)

"""
"""