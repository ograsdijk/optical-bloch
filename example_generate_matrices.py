import pickle
import numpy as np
from pathlib import Path
from couplings.utils import ED_ME_mixed_state
from hamiltonian.states import CoupledBasisState
from couplings.generate_coupling import optical_coupling_matrix, \
                                        microwave_coupling_matrix
from hamiltonian.generate_hamiltonian import generate_X_hamiltonian_function, \
                                                generate_B_hamiltonian_function
from hamiltonian.utils import matrix_to_states, ni_range, find_exact_states, \
                                reduced_basis_hamiltonian, \
                                find_state_idx_from_state, make_transform_matrix

######################################
### Generate X Hamiltonian matrix
######################################

fname_uc = "TlF_X_state_hamiltonian_J0to4.pickle"
fname_transform = "UC_to_C_J0to4.pickle"

script_dir = Path(__file__).parent.absolute()

path_uncoupled = script_dir / "stored_data" / fname_uc
path_transform = script_dir / "stored_data" / fname_transform

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

######################################
### Generate B Hamiltonian matrix
######################################

fname_c = "B_hamiltonians_symbolic_coupled_P_1to3.pickle"

path_coupled = script_dir / "stored_data" / fname_c

with open(path_coupled, 'rb') as f:
    H_B = pickle.load(f)

H_B = generate_B_hamiltonian_function(H_B)

# generate coupled basis states
Jmin = 1
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

print("Hamiltonian matrices shapes", H_X_red.shape, H_B_red.shape)

# #####################################
# ### Generate laser couplings
# #####################################

QN = ground_states + excited_states

#d efine ground states for laser driven transition
Js = [2]
ground_states_laser_approx =  [1*CoupledBasisState(
                F,mF,F1,J,I_F,I_Tl, electronic_state='X', P = (-1)**J, Omega = 0
                )
        for J  in Js
        for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
        for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
        for mF in ni_range(-F, F+1)
    ]

ground_states_laser = find_exact_states(
                                ground_states_laser_approx, H_X_diag, QN_X_diag
                                )
excited_states_laser = excited_states

H_laser_z = optical_coupling_matrix(
                                    QN, 
                                    ground_states_laser, 
                                    excited_states_laser, 
                                    pol_vec = np.array([0,0,1]), 
                                    reduced = False
                                    )
H_laser_x = optical_coupling_matrix(
                                    QN, 
                                    ground_states_laser, 
                                    excited_states_laser, 
                                    pol_vec = np.array([1,0,0]), 
                                    reduced = False
                                    )

# set small values to zero
H_laser_z[np.abs(H_laser_z) < 1e-3*np.max(np.abs(H_laser_z))] = 0
H_laser_x[np.abs(H_laser_x) < 1e-3*np.max(np.abs(H_laser_x))] = 0

# calculate the matrix element for the "main" transition so that coupling matrix 
# can be scaled to have appropriate rabi rate
# approximate form of main ground state
ground_main_approx = 1*CoupledBasisState(
        J=2,F1=5/2,F=2,mF=0,I1=1/2,I2=1/2,electronic_state='X', P = 1, Omega = 0
        )
ground_main_i = find_state_idx_from_state(
                                        H_X_diag,ground_main_approx, QN_X_diag
                                        )
ground_main = QN_X_diag[ground_main_i]

# approximate form of main excited state
excited_main_approx = 1*CoupledBasisState(
    J = 1,F1=3/2,F=1,mF=0,I1=1/2,I2=1/2, electronic_state='B', P = -1, Omega = 1
    )
excited_main_i = find_state_idx_from_state(
                                        H_B_diag,excited_main_approx, QN_B_diag
                                        )
excited_main = QN_B_diag[excited_main_i]

ME_main = ED_ME_mixed_state(
                        excited_main, ground_main, pol_vec = np.array([0,0,1])
                        )

print("ME_main =", ME_main, 'idx = ', ground_main_i, excited_main_i)
print("Laser coupling matrices shapes", H_laser_x.shape, H_laser_z.shape)

#############################################
### Generate microwave couplings J1-> J2
#############################################

# matrix element for the "main" transition so that coupling matrix can be scaled 
# to have appropriate rabi rate
# approximate form of main ground state
mu1_ground_approx = 1*CoupledBasisState(
    J=1,F1=3/2,F=2,mF=0,I1=1/2,I2=1/2,electronic_state='X', P = -1, Omega = 0
    )
mu1_ground_i = find_state_idx_from_state(H_X_diag,mu1_ground_approx, QN_X_diag)
mu1_ground = QN_X_diag[mu1_ground_i]

# approximate form of main excited state
mu1_excited_approx = 1*CoupledBasisState(
    J = 2,F1=5/2,F=3,mF=0,I1=1/2,I2=1/2, electronic_state='X', P = +1, Omega = 0
    )
mu1_excited_i = find_state_idx_from_state(H_X_diag,mu1_excited_approx, QN_X_diag)
mu1_excited = QN_X_diag[mu1_excited_i]

ME_mu1 = ED_ME_mixed_state(mu1_excited, mu1_ground, pol_vec = np.array([0,0,1]))

# energy difference between the states
omega_mu1 = np.real(H_X_diag[mu1_excited_i,mu1_excited_i] - H_X_diag[mu1_ground_i,mu1_ground_i])

J1_mu1 = 1
J2_mu1 = 2

H1_z = microwave_coupling_matrix(
                    J1_mu1, J2_mu1, ground_states, pol_vec = np.array([0,0,1])
                    )
H1_y = microwave_coupling_matrix(
                    J1_mu1, J2_mu1, ground_states, pol_vec = np.array([0,1,0])
                    )

# set very small elements to zero
H1_z[np.abs(H1_z) < 1e-3*np.max(np.abs(H1_z))] = 0
H1_y[np.abs(H1_y) < 1e-3*np.max(np.abs(H1_y))] = 0

# generate matrices for transforming to rotating frame
U1, D1 = make_transform_matrix(J1_mu1, J2_mu1, omega_mu1, ground_states)

print("Microwave J1 -> J2 coupling matrices shapes", H1_z.shape, H1_y.shape, D1.shape)

#############################################
### Generate microwave couplings J2-> J3
#############################################

# matrix element for the "main" transition so that coupling matrix can be scaled 
# to have appropriate rabi rate
# approximate form of main ground state
mu2_ground_approx = 1*CoupledBasisState(
    J=2,F1=5/2,F=3,mF=0,I1=1/2,I2=1/2,electronic_state='X', P = +1, Omega = 0
    )
mu2_ground_i = find_state_idx_from_state(H_X_diag,mu2_ground_approx, QN_X_diag)
mu2_ground = QN_X_diag[mu2_ground_i]

# approximate form of main excited state
mu2_excited_approx = 1*CoupledBasisState(
    J = 3,F1=7/2,F=4,mF=0,I1=1/2,I2=1/2, electronic_state='X', P = -1, Omega = 0
    )
mu2_excited_i = find_state_idx_from_state(H_X_diag,mu2_excited_approx, QN_X_diag)
mu2_excited = QN_X_diag[mu2_excited_i]

ME_mu2 = ED_ME_mixed_state(mu2_excited, mu2_ground, pol_vec = np.array([0,0,1]))

# energy difference between the states
omega_mu2 = np.real(H_X_diag[mu2_excited_i,mu2_excited_i] - H_X_diag[mu2_ground_i,mu2_ground_i])

J1_mu2 = 2
J2_mu2 = 3

H2_z = microwave_coupling_matrix(
                    J1_mu2, J2_mu2, ground_states, pol_vec = np.array([0,0,1])
                    )
H2_y = microwave_coupling_matrix(
                    J1_mu2, J2_mu2, ground_states, pol_vec = np.array([0,1,0])
                    )

# set very small elements to zero
H2_z[np.abs(H2_z) < 1e-3*np.max(np.abs(H2_z))] = 0
H2_y[np.abs(H2_y) < 1e-3*np.max(np.abs(H2_y))] = 0

#Generate matrices for transforming to rotating frame
U2, D2 = make_transform_matrix(J1_mu2, J2_mu2, omega_mu1+omega_mu2, ground_states)

print("Microwave J2 -> J3 coupling matrices shapes", H1_z.shape, H1_y.shape, D1.shape)
