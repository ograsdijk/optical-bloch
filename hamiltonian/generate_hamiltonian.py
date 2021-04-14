import sympy
import numpy as np
from hamiltonian.utils import reorder_evecs
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