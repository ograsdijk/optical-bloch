import multiprocessing
import numpy as np
from tqdm import tqdm
from sympy import zeros
from multiprocessing import Pool
from symbolic.utils import multi_C_ρ_Cconj
from symbolic.density_matrix import generate_density_matrix_symbolic


def generate_system_of_equations(hamiltonian, C_array, progress = False, 
                                    nprocs = 1, fast = False,
                                    split_output = False):
    n_states = hamiltonian.shape[0]
    ρ = generate_density_matrix_symbolic(n_states)
    C_conj_array = np.einsum('ijk->ikj', C_array.conj())

    matrix_mult_sum = zeros(n_states, n_states)
    if fast:
        # C_array is an array of 2D arrays, where each 2D array only has one 
        # entry, i.e. don't have to do the full matrix multiplication each 
        # time for C@ρ@Cᶜ, i.e. using manual spare matrix multiplication
        for C,Cᶜ in tqdm(zip(C_array, C_conj_array), disable = not progress):
            idC = np.nonzero(C)
            idCᶜ = np.nonzero(Cᶜ)
            val = C[idC][0]*Cᶜ[idCᶜ][0]*ρ[idC[-1],idCᶜ[0]][0]
            matrix_mult_sum[idC[0][0],idCᶜ[-1][0]] += val   

    else:
        if nprocs > 1:
            with multiprocessing.Pool(processes = nprocs) as pool:
                results = pool.starmap(multi_C_ρ_Cconj, [(C,Cᶜ,ρ) for C,Cᶜ in
                                                    zip(C_array, C_conj_array)])
                matrix_mult_sum += np.sum(results)
        else:
            for idx in tqdm(range(C_array.shape[0]), disable = not progress):
                matrix_mult_sum[:,:] += C_array[idx]@ρ@C_conj_array[idx]

    Cprecalc = np.einsum('ijk,ikl', C_conj_array, C_array)

    a = -0.5 * (Cprecalc@ρ + ρ@Cprecalc)
    b = -1j*(hamiltonian@ρ - ρ@hamiltonian)
 
    if split_output:
        return b, matrix_mult_sum + a
    else:
        system = zeros(n_states, n_states)
        system += matrix_mult_sum + a +b
        return system