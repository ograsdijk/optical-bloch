import TlFOpticalBloch

du = zeros(ComplexF64, 39, 39)
ρ = zeros(ComplexF64, 39, 39)

TlFOpticalBloch.square_wave(0., 1., 0.)

p = ones(5)
TlFOpticalBloch.Lindblad_rhs_P2F1!(du, ρ, p, 1.)

p = ones(9)
TlFOpticalBloch.Lindblad_rhs_P2F1_J12!(du, ρ, p, 1.)