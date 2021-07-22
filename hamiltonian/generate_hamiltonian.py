import sympy
import scipy
import pickle
import numpy as np
from pathlib import Path
from hamiltonian.states import CoupledBasisState
from hamiltonian.utils import reorder_evecs, ni_range, \
                            generate_coupled_ground_states, matrix_to_states, \
                            find_exact_states, reduced_basis_hamiltonian
from hamiltonian.constants_X import B_rot as B_rot_X
from hamiltonian.constants_B import B_rot as B_rot_B
from hamiltonian.constants_X import B_ϵ, α, D_TlF, μ_J, μ_Tl, μ_F
from hamiltonian.constants_B import D_rot, H_const, h1_Tl, h1_F, q, c_Tl, \
                                    c1p_Tl, μ_B, gL, gS


def generate_X_hamiltonian_function(components, c1 = 126030.0, c2 = 17890.0, c3 = 700.0, 
                        c4 = -13300.0):
    """Generate function that generates the Hamiltonian in units radian/s as a 
    function of E and B fields

    Args:
        components (dict): dict with symbolic matrices of the components making 
                    up the full TlF Hamiltonian
        c1 (float, optional): [description]. Defaults to 126030.0.
        c2 (float, optional): [description]. Defaults to 17890.0.
        c3 (float, optional): [description]. Defaults to 700.0.
        c4 (float, optional): [description]. Defaults to -13300.0.

    Returns:
        function: function that generates the Hamiltonian in units radian/s as a 
                    function of E and B fields
    """

    # Substitute values into hamiltonian
    variables = [
        sympy.symbols('Brot'),
        *sympy.symbols('c1 c2 c3 c4'),
        sympy.symbols('D_TlF'),
        *sympy.symbols('mu_J mu_Tl mu_F')
    ]
    
    lambdified_hamiltonians = {
        H_name : sympy.lambdify(variables, H_matrix)
        for H_name, H_matrix in components.items()
    }

    # constants import from constants.py
    H = {
        H_name : H_fn(
            B_rot_X,
            c1, c2, c3, c4,
            D_TlF,
            μ_J, μ_Tl, μ_F
        )
        for H_name, H_fn in lambdified_hamiltonians.items()
        }
    
    ham_func = lambda E,B: 2*np.pi*(H["Hff"] + \
        E[0]*H["HSx"]  + E[1]*H["HSy"] + E[2]*H["HSz"] + \
        B[0]*H["HZx"]  + B[1]*H["HZy"] + B[2]*H["HZz"])

    return ham_func

def generate_B_hamiltonian_function(components):
    """Generate function that generates the Hamiltonian in units radian/s

    Args:
        components (dict): dict with symbolic matrices of the components making 
                    up the full TlF Hamiltonian

    Returns:
        function: function that generates the Hamiltonian in units radian/s as a 
                    function of E and B fields
    """

    variables = [
        *sympy.symbols('Brot Drot H_const'),
        *sympy.symbols('h1_Tl h1_F'),
        sympy.symbols('q'),
        sympy.symbols('c_Tl'),
        sympy.symbols('c1p_Tl'),
        sympy.symbols('mu_B'),
        *sympy.symbols('gS gL')
    ]

    lambdified_hamiltonians = {
        H_name : sympy.lambdify(variables, H_matrix)
        for H_name, H_matrix in components.items()
    }

    H = {
        H_name : H_fn(
            B_rot_B, D_rot, H_const,
            h1_Tl, h1_F,
            q,
            c_Tl,
            c1p_Tl,
            μ_B,
            gS, gL
        )
        for H_name, H_fn in lambdified_hamiltonians.items()
    }

    ham =   H["Hrot"] + H["H_mhf_Tl"] + H["H_mhf_F"] + H["H_c_Tl"] + \
            H["H_cp1_Tl"] + H["H_LD"] + H["HZz"]*0.01

    return ham

def generate_diagonalized_hamiltonian(hamiltonian, keep_order = True, 
                                        return_V_ref = False):
    D, V = np.linalg.eigh(hamiltonian)
    if keep_order:
        V_ref = np.eye(V.shape[0])
        D, V = reorder_evecs(V,D,V_ref)

    hamiltonian_diagonalized = V.conj().T @ hamiltonian @ V
    if not return_V_ref or not keep_order:
        return hamiltonian_diagonalized, V
    else:
        return hamiltonian_diagonalized, V, V_ref

def generate_reduced_X_hamiltonian(ground_states_approx, Jmin = 0, Jmax = 4, 
                                    E = np.array([0,0,0]),
                                    B = np.array([0,0,0.001])):

    fname_X = "TlF_X_state_hamiltonian_J0to4.pickle"
    fname_transform = "UC_to_C_J0to4.pickle"

    script_dir = Path(__file__).parent.parent.absolute()
    path_X = script_dir / "stored_data" / fname_X
    path_transform = script_dir / "stored_data" / fname_transform

    with open(path_X, 'rb') as f:
        H_X_uc = pickle.load(f)

    with open(path_transform, 'rb') as f:
        S_transform = pickle.load(f)

    H_X_uc = generate_X_hamiltonian_function(H_X_uc)

    parity = lambda J: (-1)**J
    QN_X = generate_coupled_ground_states(ni_range(Jmin, Jmax + 1), 
                                            electronic_state = 'X',
                                            parity = parity, Ω = 0, I_Tl = 1/2, 
                                            I_F = 1/2)
    H_X = S_transform.conj().T @ H_X_uc(E,B) @ S_transform
    
    # diagonalize the Hamiltonian
    H_X_diag, V, V_ref_X = generate_diagonalized_hamiltonian(H_X, 
                                                            keep_order = True, 
                                                            return_V_ref = True)

    # new set of quantum numbers:
    QN_X_diag = matrix_to_states(V, QN_X)

    ground_states = find_exact_states(ground_states_approx, H_X_diag, QN_X_diag, 
                                        V_ref = V_ref_X)

    H_X_red = reduced_basis_hamiltonian(QN_X_diag, H_X_diag, ground_states)

    return ground_states, H_X_red

def generate_reduced_B_hamiltonian(excited_states_approx, Jmin = 1, Jmax = 3, 
                                    E = np.array([0,0,0]),
                                    B = np.array([0,0,0.001])):
    script_dir = Path(__file__).parent.parent.absolute()
    fname_B = "B_hamiltonians_symbolic_coupled_P_1to3.pickle"

    path_B = script_dir / "stored_data" / fname_B

    with open(path_B, 'rb') as f:
        H_B = pickle.load(f)

    # calculated B hamiltonian loaded from file is missing a factor 2π
    H_B = generate_B_hamiltonian_function(H_B)*2*np.pi

    # generate coupled basis states
    Ps = [-1,1]
    I_F = 1/2
    I_Tl = 1/2
    QN_B = [CoupledBasisState(
                        F,mF,F1,J,I_F,I_Tl,P = P, Omega = 1, electronic_state='B'
                        )
            for J  in ni_range(Jmin, Jmax+1)
            for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
            for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
            for mF in ni_range(-F, F+1)
            for P in Ps
        ]


    H_B_diag, V, V_ref_B = generate_diagonalized_hamiltonian(H_B,
                                                            keep_order = True, 
                                                            return_V_ref = True)

    # new set of quantum numbers:
    QN_B_diag = matrix_to_states(V, QN_B)

    excited_states = find_exact_states(excited_states_approx, H_B_diag, QN_B_diag, 
                                        V_ref=V_ref_B)

    H_B_red = reduced_basis_hamiltonian(QN_B_diag, H_B_diag, excited_states)
    return excited_states, H_B_red

def generate_total_hamiltonian(H_X_red, H_B_red, element_limit = 0.1):
    H_X_red[np.abs(H_X_red) < element_limit] = 0
    H_B_red[np.abs(H_B_red) < element_limit] = 0

    H_int = scipy.linalg.block_diag(H_X_red, H_B_red)
    V_ref_int = np.eye(H_int.shape[0])

    return H_int, V_ref_int