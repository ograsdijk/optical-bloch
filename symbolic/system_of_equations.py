import numpy as np
from tqdm import tqdm
from sympy import zeros
from symbolic.density_matrix import generate_density_matrix_symbolic


def generate_system_of_equations(hamiltonian, C_array, progress = False):
    n_states = hamiltonian.shape[0]
    ρ = generate_density_matrix_symbolic(n_states)
    C_conj_array = np.einsum('ijk->ikj', C_array.conj())

    matrix_mult_sum = zeros(n_states, n_states)
    if progress:
        for idx in tqdm(range(C_array.shape[0])):
            matrix_mult_sum[:,:] += C_array[idx]@ρ@C_conj_array[idx]
    else:
        for idx in range(C_array.shape[0]):
            matrix_mult_sum[:,:] += C_array[idx]@ρ@C_conj_array[idx]

    Cprecalc = np.einsum('ijk,ikl', C_conj_array, C_array)

    a = -0.5 * (Cprecalc@ρ + ρ@Cprecalc)
    b = -1j*(hamiltonian@ρ - ρ@hamiltonian)

    system = zeros(n_states, n_states)
    system += matrix_mult_sum
    system += a
    system += b
    return system