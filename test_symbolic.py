import numpy as np
from sympy import Symbol
from optical_bloch.utils.general import flatten
from optical_bloch import Hamiltonian, Dissipator, BlochEquations

ωa = Symbol('ω_a', real = True)
ωb = Symbol('ω_b', real = True)
ωc = Symbol('ω_c', real = True)
ωd = Symbol('ω_d', real = True)
ωE1 = Symbol('ω_E1', real = True)
# ωE2 = Symbol('ω_E2', real = True)

ω_F1mF0 = Symbol('ω_F1mF0', real = True)
Ω_F1mF0 = Symbol('Ω_F1mF0', real = True)
ω_F0mF0  = Symbol('ω_F0mF0 ', real = True)
Ω_F0mF0 = Symbol('Ω_F0mF0', real = True)

# indices = [(0,14), (1,13), (3,15), (4,13), (6,15), (8,13), (9,14), (10,15)]
# other_couplings = [(ind1, ind2, Symbol(f'Ω{ind1}{ind2}', real = True), Symbol(f'ω{ind1}{ind2}', real = True)) for ind1, ind2 in indices]

Δ1 = Symbol('Δ1', real = True)
Δ2 = Symbol('Δ2', real = True)

ham = Hamiltonian(16)
ham.addEnergies([ωa, ωb, ωb, ωb, ωc, ωc, ωc, ωd, ωd, ωd, ωd, ωd, ωE1, ωE2, ωE2, ωE2])
ham.addCoupling(2,12,Ω_F0mF0,ω_F0mF0)
ham.addCoupling(5,12,Ω_F1mF0,ω_F1mF0)
# for ind1, ind2, Ω, ω in other_couplings:
#     ham.addCoupling(ind1,ind2,Ω,ω)
ham.eqnTransform()
# ham.defineZero(ωE1)
ham.defineStateDetuning(2,12,Δ1)
ham.defineStateDetuning(5,12,Δ1)
# for ind1, ind2 in indices:
#     ham.defineStateDetuning(ind1, ind2,Δ2)
