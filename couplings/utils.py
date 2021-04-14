import numpy as np
from tqdm import tqdm
from scipy.sparse import kron, eye
from sympy.physics.wigner import wigner_3j, wigner_6j

def threej_f(j1,j2,j3,m1,m2,m3):
    return complex(wigner_3j(j1,j2,j3,m1,m2,m3))

def sixj_f(j1,j2,j3,j4,j5,j6):
    return complex(wigner_6j(j1,j2,j3,j4,j5,j6))

def ED_ME_mixed_state(bra, ket, pol_vec = np.array([1,1,1]), reduced = False):
    """calculate electric dipole matrix elements between mixed states

    Args:
        bra (State): state object
        ket (State): state object
        pol_vec (np.ndarray, optional): polarization vector. 
                                        Defaults to np.array([1,1,1]).
        reduced (bool, optional): [description]. Defaults to False.

    Returns:
        complex: matrix element between bra and ket
    """
    ME = 0
    bra = bra.transform_to_omega_basis()
    ket = ket.transform_to_omega_basis()
    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += amp_bra.conjugate()*amp_ket*ED_ME_coupled(
                    basis_bra, basis_ket, pol_vec = pol_vec, rme_only = reduced
                    )

    return ME

def ED_ME_coupled(bra,ket, pol_vec = np.array([1,1,1]), rme_only = False):
    """calculate electric dipole matrix elements between coupled basis states

    Args:
        bra (CoupledBasisState): coupled basis state object
        ket (CoupledBasisState): coupled basis state object
        pol_vec (np.ndarray, optional): polarization vector. 
                                        Defaults to np.array([1,1,1]).
        rme_only (bool, optional): set True to return only reduced matrix 
                                    element, otherwise angular component is 
                                    included. Defaults to False.

    Returns:
        complex: electric dipole matrix element between bra en ket
    """

    # find quantum numbers for ground state
    F = bra.F
    mF = bra.mF
    J = bra.J
    F1 = bra.F1
    I1 = bra.I1
    I2 = bra.I2
    Omega = bra.Omega
    
    # find quantum numbers for excited state
    Fp = ket.F
    mFp = ket.mF
    Jp = ket.J
    F1p = ket.F1
    I1p = ket.I1
    I2p = ket.I2
    Omegap = ket.Omega
    
    # calculate the reduced matrix element
    q = Omega - Omegap
    ME = ((-1)**(F1+J+Fp+F1p+I1+I2) * np.sqrt((2*F+1)*(2*Fp+1)*(2*F1p+1)*(2*F1+1)) * sixj_f(F1p,Fp,I2,F,F1,1) 
          * sixj_f(Jp,F1p,I1,F1,J,1) * (-1)**(J-Omega) *np.sqrt((2*J+1)*(2*Jp+1)) * threej_f(J,1,Jp,-Omega, q, Omegap)
          * float(np.abs(q) < 2))
    
    # if we want the complete matrix element, calculate angular part
    if not rme_only:
        
        # calculate elements of the polarization vector in spherical basis
        p_vec = {}
        p_vec[-1] = -1/np.sqrt(2) * (pol_vec[0] + 1j *pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1/np.sqrt(2) * (pol_vec[0] - 1j *pol_vec[1])
        
        # calculate the value of p that connects the states
        p = mF-mFp
        p = p*int(np.abs(p) <= 1)
        # multiply RME by the angular part
        ME = ME * (-1)**(F-mF) * threej_f(F,1,Fp, -mF, p, mFp) * p_vec[p] * int(np.abs(p) <= 1)
    
    # return the matrix element
    return ME

def calculate_BR(excited_state, ground_states, tol = 1e-5):
    """
    Function that calculates branching ratios from the given excited state to the given ground states

    inputs:
    excited_state = state object representing the excited state that is spontaneously decaying
    ground_states = list of state objects that should span all the states to which the excited state can decay

    returns:
    BRs = list of branching ratios to each of the ground states
    """

    #Initialize container for matrix elements between excited state and ground states
    MEs = np.zeros(len(ground_states), dtype = complex)

    #loop over ground states
    for i, ground_state in enumerate(ground_states):
        MEs[i] = ED_ME_mixed_state(ground_state.remove_small_components(tol = tol),excited_state.remove_small_components(tol = tol))
    
    #Calculate branching ratios
    BRs = np.abs(MEs)**2/(np.sum(np.abs(MEs)**2))

    return BRs

def collapse_matrices(QN, ground_states, excited_states, gamma = 1, tol = 1e-4):
    """
    Function that generates the collapse matrix for given ground and excited states

    inputs:
    QN = list of states that defines the basis for the calculation
    ground_states = list of ground states that are coupled to the excited states
    excited_states = list of excited states that are coupled to the ground states
    gamma = decay rate of excited states
    tol = couplings smaller than tol/sqrt(gamma) are set to zero to speed up computation

    outputs:
    C_list = list of collapse matrices
    """
    #Initialize list of collapse matrices
    C_list = []

    #Start looping over ground and excited states
    for excited_state in tqdm(excited_states):
        j = QN.index(excited_state)
        BRs = calculate_BR(excited_state, ground_states)
        if np.sum(BRs) > 1:
            print(f"Warning: Branching ratio sum > 1, difference = {np.sum(BRs)-1:.2e}")
        for ground_state, BR in zip(ground_states, BRs):
            i = QN.index(ground_state)

            if np.sqrt(BR) > tol:
                #Initialize the coupling matrix
                H = np.zeros((len(QN),len(QN)), dtype = complex)
                H[i,j] = np.sqrt(BR*gamma)

                C_list.append(H)

    return C_list

def generate_sharp_superoperator(M, identity = None):
    """
    Given an operator M in Hilbert space, generates sharp superoperator M_L in Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)
    sharp = post-multiplies density matrix: |rho@A) = A_sharp @ |rho) 

    inputs:
    M = matrix representation of operator in Hilbert space

    outputs:
    M_L = representation of M in in Liouville space
    """

    if identity == None:
         identity = eye(M.shape[0], format = 'coo')

    M_L = kron(M.T,identity, format = 'csr')

    return M_L

def generate_flat_superoperator(M, identity = None):
    """
    Given an operator M in Hilbert space, generates flat superoperator M_L in Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)
    flat = pre-multiplies density matrix: |A@rho) = A_flat @ |rho)

    inputs:
    M = matrix representation of operator in Hilbert space

    outputs:
    M_L = representation of M in in Liouville space
    """
    if identity == None:
         identity = eye(M.shape[0], format = 'coo')

    M_L = kron(identity, M, format = 'csr')

    return M_L

def generate_superoperator(A,B):
    """
    Function that generates superoperator representing |A@rho@B) = np.kron(B.T @ A) @ |rho)

    inputs:
    A,B = matrix representations of operators in Hilbert space

    outpus:
    M_L = representation of A@rho@B in Liouville space
    """

    M_L = kron(B.T, A, format = 'csr')

    return M_L