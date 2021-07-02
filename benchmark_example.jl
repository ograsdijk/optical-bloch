using BenchmarkTools
using LinearAlgebra
using Trapz
using DifferentialEquations

# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30

function sine_wave(t, frequency, phase)
    0.5.*(1+sin(2*pi.*frequency.*t .+ phase))
end

function prob_func(prob,i,repeat)
    remake(prob,p=[Ω1; params[i]; Δ1; Ω2; params[i]; Δ2])
end

function output_func(sol,i)
    return trapz(sol.t, [real(sum(diag(sol.u[j])[13:end])) for j in 1:size(sol.u)[1]]), false
end

function Lindblad_rhs!(du, ρ, p, t)
	@inbounds begin
		Ω1 = p[1]
		νp1 = p[2]
		Δ1 = p[3]
		Ω2 = p[4]
		νp2 = p[5]
		Δ2 = p[6]
		Ω1ᶜ = conj(Ω1)
		Ω2ᶜ = conj(Ω2)
		Px1 = sine_wave(t, νp1, 4.71238898038469)
		Pz1 = sine_wave(t, νp1, 1.5707963267948966)
		Px2 = sine_wave(t, νp2, 1.5707963267948966)
		Pz2 = sine_wave(t, νp2, 4.71238898038469)
		norm1 = sqrt(Px1^2+Pz1^2)
		norm2 = sqrt(Px2^2+Pz2^2)
		Px1 /= norm1
		Pz1 /= norm1
		Px2 /= norm2
		Pz2 /= norm2
		du[1,1] = 2198254.70329198*ρ[14,14] + 2198721.58229457*ρ[15,15] + 2199188.50568063*ρ[16,16] - 1.0*1im*(0.487104585992817*Ω2*ρ[14,1]*Px2 - 0.688943961004229*Ω2*ρ[15,1]*Pz2 - 0.487209306651389*Ω2*ρ[16,1]*Px2 - 0.487104585992817*Ω2ᶜ*ρ[1,14]*Px2 + 0.688943961004229*Ω2ᶜ*ρ[1,15]*Pz2 + 0.487209306651389*Ω2ᶜ*ρ[1,16]*Px2)
		# du[1,2] = -1.0*1im*(-0.607920432419512*Ω1ᶜ*ρ[1,13]*Px1 + 0.487104536814559*Ω2*ρ[14,2]*Px2 - 0.688943961004228*Ω2*ρ[15,2]*Pz2 - 0.487209355797023*Ω2*ρ[16,2]*Px2 - 0.674428203801002*Ω2ᶜ*ρ[1,14]*Pz2 - 0.476889169493474*Ω2ᶜ*ρ[1,15]*Px2 - 139741.094467163*ρ[1,2])
		# du[1,3] = -1.0*1im*(0.859730538004616*Ω1ᶜ*ρ[1,13]*Pz1 + 0.487104536814559*Ω2*ρ[14,3]*Px2 - 0.688943961004228*Ω2*ρ[15,3]*Pz2 - 0.487209355797023*Ω2*ρ[16,3]*Px2 - 0.476946169171229*Ω2ᶜ*ρ[1,14]*Px2 - 0.476834769689028*Ω2ᶜ*ρ[1,16]*Px2 - 139730.395599365*ρ[1,3])
		# du[1,4] = -1.0*1im*(0.607922162407229*Ω1ᶜ*ρ[1,13]*Px1 + 0.487104536814559*Ω2*ρ[14,4]*Px2 - 0.688943961004228*Ω2*ρ[15,4]*Pz2 - 0.487209355797023*Ω2*ρ[16,4]*Px2 - 0.476891775125028*Ω2ᶜ*ρ[1,15]*Px2 + 0.674421743590696*Ω2ᶜ*ρ[1,16]*Pz2 - 139719.693450928*ρ[1,4])
		# du[1,5] = -1.0*1im*(-0.353554866850789*Ω1ᶜ*ρ[1,13]*Px1 + 0.487104536814559*Ω2*ρ[14,5]*Px2 - 0.688943961004228*Ω2*ρ[15,5]*Pz2 - 0.487209355797023*Ω2*ρ[16,5]*Px2 + 0.283550246614439*Ω2ᶜ*ρ[1,14]*Pz2 + 0.200553927269744*Ω2ᶜ*ρ[1,15]*Px2 - 1245271.96743774*ρ[1,5])
		# du[1,6] = -1.0*1im*(0.5*Ω1ᶜ*ρ[1,13]*Pz1 + 0.487104536814559*Ω2*ρ[14,6]*Px2 - 0.688943961004228*Ω2*ρ[15,6]*Pz2 - 0.487209355797023*Ω2*ρ[16,6]*Px2 + 0.200508115928818*Ω2ᶜ*ρ[1,14]*Px2 + 0.200548440065767*Ω2ᶜ*ρ[1,16]*Px2 - 1245272.14598083*ρ[1,6])
		# du[1,7] = -1.0*1im*(0.353551915425323*Ω1ᶜ*ρ[1,13]*Px1 + 0.487104536814559*Ω2*ρ[14,7]*Px2 - 0.688943961004228*Ω2*ρ[15,7]*Pz2 - 0.487209355797023*Ω2*ρ[16,7]*Px2 + 0.200502623030383*Ω2ᶜ*ρ[1,15]*Px2 - 0.283629383189517*Ω2ᶜ*ρ[1,16]*Pz2 - 1245272.32510376*ρ[1,7])
		# du[1,8] = -1.0*1im*(0.487104536814559*Ω2*ρ[14,8]*Px2 - 0.688943961004228*Ω2*ρ[15,8]*Pz2 - 0.487209355797023*Ω2*ρ[16,8]*Px2 - 0.43301270380128*Ω2ᶜ*ρ[1,14]*Px2 - 1336642.61555481*ρ[1,8])
		# du[1,9] = -1.0*1im*(0.487104536814559*Ω2*ρ[14,9]*Px2 - 0.688943961004228*Ω2*ρ[15,9]*Pz2 - 0.487209355797023*Ω2*ρ[16,9]*Px2 + 0.433033582428828*Ω2ᶜ*ρ[1,14]*Pz2 - 0.306171284690352*Ω2ᶜ*ρ[1,15]*Px2 - 1336632.31462097*ρ[1,9])
		# du[1,10] = -1.0*1im*(0.487104536814559*Ω2*ρ[14,10]*Px2 - 0.688943961004228*Ω2*ρ[15,10]*Pz2 - 0.487209355797023*Ω2*ρ[16,10]*Px2 + 0.176793733186045*Ω2ᶜ*ρ[1,14]*Px2 + 0.5*Ω2ᶜ*ρ[1,15]*Pz2 - 0.176759656638464*Ω2ᶜ*ρ[1,16]*Px2 - 1336622.01316833*ρ[1,10])
		# du[1,11] = -1.0*1im*(0.487104536814559*Ω2*ρ[14,11]*Px2 - 0.688943961004228*Ω2*ρ[15,11]*Pz2 - 0.487209355797023*Ω2*ρ[16,11]*Px2 + 0.306201155498907*Ω2ᶜ*ρ[1,15]*Px2 + 0.432991815984885*Ω2ᶜ*ρ[1,16]*Pz2 - 1336611.71125793*ρ[1,11])
		# du[1,12] = -1.0*1im*(0.487104536814559*Ω2*ρ[14,12]*Px2 - 0.688943961004228*Ω2*ρ[15,12]*Pz2 - 0.487209355797023*Ω2*ρ[16,12]*Px2 + 0.433012703800787*Ω2ᶜ*ρ[1,16]*Px2 - 1336601.40879822*ρ[1,12])
		# du[1,13] = -5026548.21497942*ρ[1,13] - 1.0*1im*(-Δ1*ρ[1,13] - 0.607920432419512*Ω1*ρ[1,2]*Px1 + 0.859730538004616*Ω1*ρ[1,3]*Pz1 + 0.607922162407229*Ω1*ρ[1,4]*Px1 - 0.353554866850789*Ω1*ρ[1,5]*Px1 + 0.5*Ω1*ρ[1,6]*Pz1 + 0.353551915425323*Ω1*ρ[1,7]*Px1 + 0.487104536814559*Ω2*ρ[14,13]*Px2 - 0.688943961004228*Ω2*ρ[15,13]*Pz2 - 0.487209355797023*Ω2*ρ[16,13]*Px2 - 1245272.14598083*ρ[1,13])
		# du[1,14] = -5026548.24574367*ρ[1,14] - 1.0*1im*(-0.487104536814559*Ω2*ρ[1,1]*Px2 - 0.674428203801002*Ω2*ρ[1,2]*Pz2 - 0.476946169171229*Ω2*ρ[1,3]*Px2 + 0.283550246614439*Ω2*ρ[1,5]*Pz2 + 0.200508115928818*Ω2*ρ[1,6]*Px2 - 0.43301270380128*Ω2*ρ[1,8]*Px2 + 0.433033582428828*Ω2*ρ[1,9]*Pz2 + 0.176793733186045*Ω2*ρ[1,10]*Px2 + 0.487104536814559*Ω2*ρ[14,14]*Px2 - 0.688943961004228*Ω2*ρ[15,14]*Pz2 - 0.487209355797023*Ω2*ρ[16,14]*Px2 - ρ[1,14]*(Δ2 + 91343.5840148926) - 1245272.14598083*ρ[1,14])
    end
    nothing
end

Γ = 2pi*1.6e6
Ω1 = Γ
Ω2 = Γ
Δ1 = 0.
Δ2 = 0.
νp1 = 1e6
νp2 = 1e6
n_states = 16

params = ones(100)*2.8e6

p = [Ω1, νp1, Δ1, Ω2, νp2, Δ2]

ρ = zeros(ComplexF64, n_states, n_states)
ρ[1,1] = 1.
tspan = (0,300e-6)

du = zeros(ComplexF64, n_states, n_states)

prob = ODEProblem(Lindblad_rhs!,ρ,tspan,p)

display(@benchmark solve(prob, Tsit5(), dt = 1e-9, adaptive=true, abstol = 5e-7, reltol = 5e-4))

# ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)

# Lindblad_rhs!(du, ρ, p, 0.1)
# display(@btime Lindblad_rhs!(du, ρ, p, 0.1))

# display(@benchmark solve(ens_prob, Tsit5(), EnsembleSerial(), save_start = true, save_end = true,
#                  save_everystep = true; trajectories = size(params)[1],
#                  dt = 1e-9, adaptive=true, abstol = 5e-7, reltol = 5e-4))

# display(@benchmark solve(ens_prob, Tsit5(), EnsembleThreads(), save_start = true, save_end = true,
#               save_everystep = true; trajectories = size(params)[1],
#               dt = 1e-9, adaptive=true, abstol = 5e-7, reltol = 5e-4))


# first line only
# BenchmarkTools.Trial:
#   memory estimate:  30.18 MiB
#   allocs estimate:  14322
#   --------------
#   minimum time:     6.780 ms (0.00% GC)
#   median time:      8.922 ms (22.01% GC)
#   mean time:        8.486 ms (13.97% GC)
#   maximum time:     22.557 ms (31.40% GC)
#   --------------
#   samples:          3530
#   evals/sample:     1

# BenchmarkTools.Trial:
#   memory estimate:  30.19 MiB
#   allocs estimate:  14355
#   --------------
#   minimum time:     1.827 ms (0.00% GC)
#   median time:      4.166 ms (0.00% GC)
#   mean time:        5.854 ms (33.06% GC)
#   maximum time:     61.606 ms (92.00% GC)
#   --------------
#   samples:          5112
#   evals/sample:     1

# slow line
# BenchmarkTools.Trial:
#   memory estimate:  6.01 GiB
#   allocs estimate:  2063222
#   --------------
#   minimum time:     1.851 s (14.80% GC)
#   median time:      1.887 s (15.18% GC)
#   mean time:        1.914 s (15.31% GC)
#   maximum time:     2.070 s (16.10% GC)
#   --------------
#   samples:          16
#   evals/sample:     1

# BenchmarkTools.Trial:
#   memory estimate:  6.01 GiB
#   allocs estimate:  2063255
#   --------------
#   minimum time:     3.825 s (50.15% GC)
#   median time:      4.199 s (47.21% GC)
#   mean time:        4.212 s (46.91% GC)
#   maximum time:     4.551 s (49.54% GC)
#   --------------
#   samples:          8
#   evals/sample:     1

# single solve slow
# BenchmarkTools.Trial:
#   memory estimate:  39.23 MiB
#   allocs estimate:  10929
#   --------------
#   minimum time:     8.964 ms (0.00% GC)
#   median time:      10.482 ms (0.00% GC)
#   mean time:        11.642 ms (14.00% GC)
#   maximum time:     32.532 ms (49.23% GC)
#   --------------
#   samples:          2574
#   evals/sample:     1

# single solve fast
# BenchmarkTools.Trial:
#   memory estimate:  299.78 KiB
#   allocs estimate:  105
#   --------------
#   minimum time:     51.900 μs (0.00% GC)
#   median time:      134.700 μs (0.00% GC)
#   mean time:        142.521 μs (17.37% GC)
#   maximum time:     8.636 ms (98.65% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1