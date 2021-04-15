from sympy import Symbol, zeros

def generate_symbolic_hamiltonian(n_states, laser_fields):
    hamiltonian = zeros(n_states, n_states)

    for idf, (laser_field, ME_main) in enumerate(laser_fields):
        Ω = Symbol(f'Ω{idf+1}', complex = True)
        Ωᶜ = Symbol(f'Ω{idf+1}ᶜ', complex = True)
        hamiltonian += (Ω/ME_main)/2 * laser_field

    # ensure Hermitian Hamiltonian for complex Ω
    for idx in range(n_states):
        for idy in range(n_states):
            if idx > idy:
                for idf in range(len(laser_field)):
                    Ω = Symbol(f'Ω{idf+1}', complex = True)
                    Ωᶜ = Symbol(f'Ω{idf+1}ᶜ', complex = True)
                    hamiltonian[idx,idy] = hamiltonian[idx,idy].subs(Ω, Ωᶜ)
    symbols = [(Symbol(f'Ω{idf+1}', complex = True), 
                Symbol(f'Ω{idf+1}ᶜ', complex = True))
                for idf in range(len(laser_fields))]
    return hamiltonian, symbols

# def generate_detunings()
#     for excited_state in excited_state_indices:
#         hamiltonian[excited_state, excited_state] += Δ
    
#     return hamiltonian

def generate_symbolic_detunings(n_states, detunings):
    detuning = zeros(n_states, n_states)
    
    if len(detunings) == 1:
        for idd, indices in enumerate(detunings):
            Δ = Symbol(f'Δ', real = True)
            for idx in indices:
                detuning[idx, idx] += Δ

        symbols = [Symbol(f'Δ', complex = True)]
    else:
        for idd, indices in enumerate(detunings):
            Δ = Symbol(f'Δ{idd+1}', real = True)
            for idx in indices:
                detuning[idx, idx] += Δ

        symbols = [Symbol(f'Δ{idd+1}', complex = True)
                    for idd in range(len(detunings))]
    return detuning, symbols