import numpy as np
from hamiltonian.states import State

def matrix_to_states(V, QN, E = None):
    """Turn a matrix of eigenvectors into a list of state objects

    Args:
        V (np.ndarray): array with columns corresponding to eigenvectors
        QN (list): list of State objects
        E (list, optional): list of energies corresponding to the states. 
                            Defaults to None.

    Returns:
        list: list of eigenstates expressed as State objects
    """
    # find dimensions of matrix
    matrix_dimensions = V.shape
    
    # initialize a list for storing eigenstates
    eigenstates = []
    
    for i in range(0,matrix_dimensions[1]):
        # find state vector
        state_vector = V[:,i]

        # ensure that largest component has positive sign
        index = np.argmax(np.abs(state_vector))
        state_vector = state_vector * np.sign(state_vector[index])
        
        data = []
        
        # get data in correct format for initializing state object
        for j, amp in enumerate(state_vector):
            data.append((amp, QN[j]))
            
        # store the state in the list
        state = State(data)
        
        if E is not None:
            state.energy = E[i]
        
        eigenstates.append(state)
        
    
    # return the list of states
    return eigenstates

    """
    This function determines the index of the state vector most closely corresponding
    to an input state 
    
    H = Hamiltonian whose eigenstates the input state is compared to
    refernce_state = state whose index needs to be determined
    idx = index of the eigenstate that most closely corresponds to the input
    """

def find_state_idx_from_state(H, reference_state, QN, V_ref = None):
    """Determine the index of the state vector most closely corresponding to an
    input state

    Args:
        H (np.ndarray): Hamiltonian to compare to
        reference_state (State): state to find closest state in H to
        QN (list): list of state objects defining the basis for H

    Returns:
        int: index of closest state vector of H corresponding to reference_state
    """
    
    # determine state vector of reference state
    reference_state_vec = reference_state.state_vector(QN)
    
    # find eigenvectors of the given Hamiltonian
    E, V = np.linalg.eigh(H)
    
    if V_ref is not None:
        E, V = reorder_evecs(V,E,V_ref)
    
    overlaps = np.dot(np.conj(reference_state_vec),V)
    probabilities = overlaps*np.conj(overlaps)
    
    idx = np.argmax(probabilities)
    
    return idx


def find_exact_states(states_approx, H, QN, V_ref = None):
    """Find closest approximate eigenstates corresponding to states_approx

    Args:
        states_approx (list): list of State objects
        H (np.ndarray): Hamiltonian, diagonal in basis QN
        QN (list): list of State objects defining the basis for H

    Returns:
        list: list of eigenstates of H closest to states_approx
    """
    states = []
    for state_approx in states_approx:
        i = find_state_idx_from_state(H, state_approx, QN, V_ref)
        states.append(QN[i])

    return states

def reduced_basis_hamiltonian(basis_ori, H_ori, basis_red):
    """Generate Hamiltonian for a sub-basis of the original basis

    Args:
        basis_ori (list): list of states of original basis
        H_ori (np.ndarray): original Hamiltonian
        basis_red (list): list of states of sub-basis

    Returns:
        np.ndarray: Hamiltonian in sub-basis
    """

    #Determine the indices of each of the reduced basis states
    index_red = np.zeros(len(basis_red), dtype = int)
    for i, state_red in enumerate(basis_red):
        index_red[i] = basis_ori.index(state_red)

    #Initialize matrix for Hamiltonian in reduced basis
    H_red = np.zeros((len(basis_red),len(basis_red)), dtype = complex)

    #Loop over reduced basis states and pick out the correct matrix elements
    #for the Hamiltonian in the reduced basis
    for i, state_i in enumerate(basis_red):
        for j, state_j in enumerate(basis_red):
            H_red[i,j] = H_ori[index_red[i], index_red[j]]

    return H_red

def ni_range(x0, x1, dx=1):
    """
    Generating ranges
    """
    # sanity check arguments
    if dx==0:
        raise ValueError("invalid parameters: dx==0")
    if x0>x1 and dx>=0:
        raise ValueError("invalid parameters: x0>x1 and dx>=0")
    if x0<x1 and dx<=0:
        raise ValueError("invalid parameters: x0<x1 and dx<=0")
        
    # generate range list
    range_list = []
    x = x0
    while x < x1:
        range_list.append(x)
        x += dx
    return range_list

def make_transform_matrix(J1, J2, omega_mu, QN, I1 = 1/2, I2 = 1/2):
    """generate transofmration matrix rotating the Hamiltonian to the rotating
    frame

    Args:
        J1 (int): J of lower rotational state
        J2 (int): J of higher rotational state
        omega_mu (float): energy difference between states
        QN (list): list of State objects
        I1 (float, optional): spin. Defaults to 1/2.
        I2 (float, optional): sin. Defaults to 1/2.

    Returns:
        np.ndarray: unitary transformation matrix
    """
    
    # starting and ending indices of the part of the matrix that has exp(i*omega*t)
    J2_start = int((2*I1+1)*(2*I2+1)*(J2)**2)
    J2_end = int((2*I1+1)*(2*I2+1)*(J2+1)**2)
        
    # generate transformation matrices
    D = np.diag(np.concatenate((np.zeros((J2_start)), 
                                -omega_mu * np.ones((J2_end - J2_start)),
                                np.zeros((len(QN)-J2_end)))))
    
    U = lambda t: np.diag(np.concatenate((np.ones((J2_start)), 
                        np.exp(-1j*(omega_mu)*t) * np.ones((J2_end - J2_start)), 
                        np.ones(len(QN)-J2_end))))
    
    return U, D

def reorder_evecs(V_in,E_in,V_ref):
    """Reshuffle eigenvectors and eigenergies based on a reference

    Args:
        V_in (np.ndarray): eigenvector matrix to be reorganized
        E_in (np.ndarray): energy vector to be reorganized
        V_ref (np.ndarray): reference eigenvector matrix

    Returns:
        (np.ndarray, np.ndarray): energy vector, eigenvector matrix
    """
    # take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T),V_ref))
    
    # find which state has the largest overlap:
    index = np.argsort(np.argmax(overlap_vectors,axis = 1))
    # store energy and state
    E_out = E_in[index]
    V_out = V_in[:,index]   
    
    return E_out, V_out