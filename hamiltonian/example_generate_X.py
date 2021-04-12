import pickle
import numpy as np
from pathlib import Path
from generate_hamiltonian import generate_X_hamiltonian_function
from states import CoupledBasisState
from utils import matrix_to_states, ni_range, find_exact_states, \
                    reduced_basis_hamiltonian

fname_uc = "TlF_X_state_hamiltonian_J0to4.pickle"
fname_transform = "UC_to_C_J0to4.pickle"

script_dir = Path(__file__).parent.absolute()

path_uncoupled = script_dir.parent / "stored_data" / fname_uc
path_transform = script_dir.parent / "stored_data" / fname_transform

with open(path_uncoupled, 'rb') as f:
    H_X_uc = pickle.load(f)

with open(path_transform, 'rb') as f:
    S_transform = pickle.load(f)


H_X_uc = generate_X_hamiltonian_function(H_X_uc)

# generate coupled basis states
Jmin = 0
Jmax = 4
I_F = 1/2
I_Tl = 1/2

QN_X = [CoupledBasisState(
                F,mF,F1,J,I_F,I_Tl, electronic_state='X', P = (-1)**J, Omega = 0
                )
        for J  in ni_range(Jmin, Jmax+1)
        for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
        for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
        for mF in ni_range(-F, F+1)
    ]

E = np.array([0,0,0])
B = np.array([0,0,0.001])

H_X = S_transform.conj().T @ H_X_uc(E,B) @ S_transform

D, V = np.linalg.eigh(H_X)

# diagonalize the Hamiltonian
H_X_diag = V.conj().T @ H_X @ V

# new set of quantum numbers:
QN_X_diag = matrix_to_states(V, QN_X)

# define what states are to be included in the simulation
Js_g = [0,1,2,3]
ground_states_approx = [1*CoupledBasisState(
                F,mF,F1,J,I_F,I_Tl, electronic_state='X', P = (-1)**J, Omega = 0
                )
        for J  in Js_g
        for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
        for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
        for mF in ni_range(-F, F+1)
    ]

ground_states = find_exact_states(ground_states_approx, H_X_diag, QN_X_diag)

H_X_red = reduced_basis_hamiltonian(QN_X_diag, H_X_diag, ground_states)

print(H_X_red.shape)