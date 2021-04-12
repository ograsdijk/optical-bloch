import pickle
import numpy as np
from pathlib import Path
from generate_hamiltonian import generate_B_hamiltonian_function
from states import CoupledBasisState
from utils import matrix_to_states, ni_range, find_exact_states, \
                    reduced_basis_hamiltonian

fname_c = "B_hamiltonians_symbolic_coupled_P_1to3.pickle"

script_dir = Path(__file__).parent.absolute()

path_coupled = script_dir.parent / "stored_data" / fname_c

with open(path_coupled, 'rb') as f:
    H_B = pickle.load(f)

H_B = generate_B_hamiltonian_function(H_B)

# generate coupled basis states
Jmin = 0
Jmax = 3
I_F = 1/2
I_Tl = 1/2
Ps = [-1,1]

QN_B = [CoupledBasisState(
                    F,mF,F1,J,I_F,I_Tl,P = P, Omega = 1, electronic_state='B'
                    )
        for J  in ni_range(Jmin, Jmax+1)
        for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
        for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
        for mF in ni_range(-F, F+1)
        for P in Ps
    ]

D,V = np.linalg.eigh(H_B)

# diagonalize the Hamiltonian
H_B_diag = V.conj().T @ H_B @ V

# new set of quantum numbers:
QN_B_diag = matrix_to_states(V, QN_B)

# define states to be included in the simulation
J = 1
F1 = 3/2
F = 1
excited_states_approx = [1*CoupledBasisState(
                    F,mF,F1,J,I_F,I_Tl, electronic_state='B', P = -1, Omega = 1
                    )
        for mF in ni_range(-F, F+1)
    ]

excited_states = find_exact_states(excited_states_approx, H_B_diag, QN_B_diag)

H_B_red = reduced_basis_hamiltonian(QN_B_diag, H_B_diag, excited_states)

print(H_B_red.shape)