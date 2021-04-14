from sympy import Symbol, zeros

def generate_symbolic_hamiltonian(n_states, ME_main, laser_fields, 
                                    excited_state_indices):
    Ω = Symbol('Ω', complex = True)
    Ωᶜ = Symbol('Ω', complex = True)
    Δ = Symbol('Δ', real = True)

    hamiltonian = zeros(n_states, n_states)

    for laser_field in laser_fields:
        hamiltonian += (Ω/ME_main)/2 * laser_field

    # ensure Hermitian Hamiltonian for complex Ω
    for idx in range(n_states):
        for idy in range(n_states):
            if idx > idy:
                hamiltonian[idx,idy] = hamiltonian[idx,idy].subs(Ω, Ωᶜ)
    
    for excited_state in excited_state_indices:
        hamiltonian[excited_state, excited_state] += Δ
    
    return hamiltonian