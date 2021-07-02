from copy import copy
from sympy import Symbol

def recursive_subscript(i):
    # chr(0x2080+i) is unicode for
    # subscript num(i), resulting in x₀₀ for example
    if i < 10:
        return chr(0x2080+i)
    else:
        return recursive_subscript(i//10)+chr(0x2080+i%10)

def subs_rabi_rate(hamiltonian, originals, replacement):
    ham = copy(hamiltonian)
    Ωr = Symbol(f'Ω{replacement}', complex = True)
    Ωrᶜ = Symbol(f'Ω{replacement}ᶜ', complex = True)
    for original in originals:
        Ω = Symbol(f'Ω{original}', complex = True)
        Ωᶜ = Symbol(f'Ω{original}ᶜ', complex = True)
        ham = ham.subs((Ω, Ωr), (Ωᶜ, Ωrᶜ))
    return ham

def multi_C_ρ_Cconj(C, Cᶜ, ρ):
    return C@ρ@Cᶜ