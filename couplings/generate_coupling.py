import numpy as np
import multiprocessing
from hamiltonian.utils import find_exact_states
from couplings.utils import ED_ME_mixed_state, multi_coupling_matrix, \
                            check_approx_state_exact_state

def optical_coupling_matrix(QN, ground_states, excited_states, 
                            pol_vec = np.array([0,0,1]), reduced = False,
                            nprocs = 1):
    """generate optical coupling matrix for given ground and excited states

    Args:
        QN (list): list of basis states
        ground_states (list): list of ground states coupling to excited states
        excited_states (list): list of excited states
        pol_vec (np.ndarray, optional): polarization vector. Defaults to np.array([0,0,1]).
        reduced (bool, optional): [description]. Defaults to False.
        nrpocs (int): # processes to use for multiprocessing

    Returns:
        np.ndarray: optical coupling matrix
    """
    if nprocs > 1:
        with multiprocessing.Pool(nprocs) as pool:
            result = pool.starmap(multi_coupling_matrix,
                [(QN, gs, excited_states, pol_vec, reduced) for gs in ground_states])
        H = np.sum(result, axis = 0)
    else:
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

def microwave_coupling_matrix(QN, ground_states, excited_states, pol_vec = np.array([0,0,1]), 
                                reduced = False, nprocs = 1):
    """generate microwave coupling matrix between J1 and J2 rotational states,


    Args:
        QN (list): list of State objects
        J1 (int): one of the coupled rotational states
        J2 (int): one of the coupled rotational states
        pol_vec (np.ndarray, optional): polarization vector. Defaults to np.array([0,0,1]).
        nprocs (int): # processes to use for multiprocessing

    Returns:
        np.ndarray: microwave coupling matrix
    """
    if nprocs > 1:
        with multiprocessing.Pool(nprocs) as pool:
            result = pool.starmap(multi_coupling_matrix,
                [(QN, gs, excited_states, pol_vec, reduced) for gs in ground_states])
        H = np.sum(result, axis = 0)
    
    else:
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

def generate_laser_D(H, QN, ground_main, excited_main, excited_states, Δ):
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

def generate_microwave_D(H, QN, ground_main, excited_main, excited_states, Δ):
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

def generate_coupling_field(ground_main_approx, excited_main_approx, 
                            ground_states_approx, excited_states_approx, 
                            H_rot, QN, V_ref, pol_main = np.array([0,0,1]), 
                            pol_vec = [],
                            relative_coupling = 1e-3,
                            absolute_coupling = 1e-6,
                            nprocs = 2):
    ground_states = find_exact_states(
                        ground_states_approx, H_rot, QN, V_ref = V_ref
                                    )
    excited_states = find_exact_states(
                        excited_states_approx, H_rot, QN, V_ref = V_ref
                                    )
    ground_main = find_exact_states([ground_main_approx], 
                                        H_rot, QN, V_ref = V_ref)[0]
    excited_main = find_exact_states([excited_main_approx], 
                                        H_rot, QN, V_ref = V_ref)[0]

    check_approx_state_exact_state(ground_main_approx, ground_main)
    check_approx_state_exact_state(excited_main_approx, excited_main)

    ME_main = ED_ME_mixed_state(
                        excited_main, ground_main, pol_vec = pol_main)

    assert_msg = f"main coupling element small, {ME_main:.2e}" + \
                  ", check states and/or polarization"
    assert np.abs(ME_main) > 1e-2, assert_msg

    couplings = []
    for pol in pol_vec:
        coupling = optical_coupling_matrix(
                                            QN, 
                                            ground_states, 
                                            excited_states, 
                                            pol_vec = pol, 
                                            reduced = False,
                                            nprocs = nprocs)

        coupling[np.abs(coupling) < relative_coupling*np.max(np.abs(coupling))] = 0
        coupling[np.abs(coupling) < absolute_coupling] = 0
        d = {'pol': pol, 'main_coupling': ME_main, 'field': coupling, 
            'ground_main': ground_main, 'excited_main': excited_main, 
            'ground_states': ground_states, 'excited_states': excited_states}
        couplings.append(d)
    return couplings