import numpy as np
from couplings.utils import ED_ME_mixed_state

def optical_coupling_matrix(QN, ground_states, excited_states, 
                            pol_vec = np.array([0,0,1]), reduced = False
                            ):
    """generate optical coupling matrix for given ground and excited states

    Args:
        QN (list): list of basis states
        ground_states (list): list of ground states coupling to excited states
        excited_states (list): list of excited states
        pol_vec (np.ndarray, optional): polarization vector. Defaults to np.array([0,0,1]).
        reduced (bool, optional): [description]. Defaults to False.

    Returns:
        np.ndarray: optical coupling matrix
    """
    # initialize the coupling matrix
    H = np.zeros((len(QN),len(QN)), dtype = complex)

    # start looping over ground and excited states
    for ground_state in ground_states:
        i = QN.index(ground_state)
        for excited_state in excited_states:
            j = QN.index(excited_state)

            # calculate matrix element and add it to the Hamiltonian
            H[i,j] = ED_ME_mixed_state(
                                        ground_state, 
                                        excited_state, 
                                        pol_vec = pol_vec, 
                                        reduced = reduced
                                        )

    # make H hermitian
    H = H + H.conj().T

    return H

def microwave_coupling_matrix(J1, J2, QN, pol_vec = np.array([0,0,1])):
    """generate microwave coupling matrix between J1 and J2 rotational states,


    Args:
        J1 (int): one of the coupled rotational states
        J2 (int): one of the coupled rotational states
        QN (list): list of State objects
        pol_vec (np.ndarray, optional): polarization vector. Defaults to np.array([0,0,1]).

    Returns:
        np.ndarray: microwave coupling matrix
    """
    # number of states in system
    N_states = len(QN) 
    
    # initialize Hamiltonian
    H_mu = np.zeros((N_states,N_states), dtype = complex)
    
    
    # loop over states and calculate microwave matrix elements between them
    for i in range(0, N_states):
        state1 = QN[i].remove_small_components(tol = 0.001)
        
        for j in range(i, N_states):
            state2 = QN[j].remove_small_components(tol = 0.001)
            
            # check that the states have the correct values of J
            if (state1.find_largest_component().J == J1 and state2.find_largest_component().J == J2) or (state1.find_largest_component().J == J2 and state2.find_largest_component().J == J1):
                # calculate matrix element between the two states
                H_mu[i,j] = (ED_ME_mixed_state(state1, state2, reduced=False, pol_vec=pol_vec))
                
    # make H_mu hermitian
    H_mu = (H_mu + np.conj(H_mu.T)) - np.diag(np.diag(H_mu))
    
    
    # return the coupling matrix
    return H_mu

def generate_laser_D(H,QN, ground_main, excited_main, excited_states, Δ):
    # find transition frequency
    ig = QN.index(ground_main)
    ie = QN.index(excited_main)
    ω0 = (H[ie,ie] - H[ig,ig]).real

    # calculate the shift Δ = ω - ω₀
    ω = ω0 + Δ

    # shift matrix
    D = np.zeros(H.shape, H.dtype)
    for excited_state in excited_states:
        idx = QN.index(excited_state)
        D[idx,idx] -= ω

    return D