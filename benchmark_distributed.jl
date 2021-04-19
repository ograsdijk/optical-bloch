
using Distributed
using BenchmarkTools
addprocs(5)

@everywhere begin
    using LinearAlgebra
    using Trapz
    using DifferentialEquations
    include("ode_amherst_cyling.jl")
    include("utils/general_ode_functions.jl")
end

@everywhere begin
    Γ = 2pi*1.6e6
    Ω1 = Γ
    Ω2 = Γ
    Δ1 = 0.
    Δ2 = 0.
    νp1 = 1e6
    νp2 = 1e6
    params = ones(100)*2.8e6

    p = [Ω1, νp1, Δ1, Ω2, νp2, Δ2]
end
ρ = zeros(ComplexF64, 16, 16)
ρ[1,1] = 1.
tspan = (0,300e-6)

du = zeros(ComplexF64, 16, 16)

prob = ODEProblem(Lindblad_rhs!,ρ,tspan,p)

@everywhere function prob_func(prob,i,repeat)
    remake(prob,p=[Ω1; params[i]; Δ1; Ω2; params[i]; Δ2])
end

@everywhere function output_func(sol,i)
    return trapz(sol.t, [real(sum(diag(sol.u[j])[13:end])) for j in 1:size(sol.u)[1]]), false
end

ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)

sim = solve(ens_prob, Tsit5(), EnsembleDistributed(), save_start = true, save_end = true,
save_everystep = true; trajectories = size(params)[1], dt = 1e-9, adaptive=true,
abstol = 5e-7, reltol = 5e-4)

@benchmark solve(ens_prob, Tsit5(), EnsembleDistributed(), save_start = true, save_end = true,
save_everystep = true; trajectories = size(params)[1], dt = 1e-9, adaptive=true,
abstol = 5e-7, reltol = 5e-4)

"""
BenchmarkTools.Trial:
  memory estimate:  326.52 KiB
  allocs estimate:  10036
  --------------
  minimum time:     5.865 s (0.00% GC)
  median time:      5.865 s (0.00% GC)
  mean time:        5.865 s (0.00% GC)
  maximum time:     5.865 s (0.00% GC)
  --------------
  samples:          1
  evals/sample:     1
"""