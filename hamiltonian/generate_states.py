import numpy as np
from hamiltonian.utils import ni_range
from hamiltonian.states import CoupledBasisState

def generate_states(Jlist, electronic, I_F = 1/2, I_Tl = 1/2):
    assert isinstance(Jlist, (list, tuple, np.ndarray)), \
                                                    "Supply list of J levels"
    parity = lambda J: (-1)**J
    states = [
                1*CoupledBasisState(F,mF,F1,J,I_F,I_Tl, 
                electronic_state=electronic, P = parity(J), Omega = 0)
                for J  in Jlist
                for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
                for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
                for mF in ni_range(-F, F+1)
            ]
    return states

def generate_ground_states_approx(Jlist, I_F = 1/2, I_Tl = 1/2):
    ground_states = generate_states(Jlist, 'X', I_F, I_Tl)
    return ground_states

# def generate_excited_states_approx(Jlist, I_F = 1/2, I_Tl = 1/2):
#     excited_states = generate_states(Jlist, 'B', I_F, I_Tl)
#     return excited_states

def generate_excited_states_approx(Jlist, Flist, F1list, P, 
                                    I_F = 1/2, I_Tl = 1/2):
    excited_states_approx = [
        1*CoupledBasisState(F,mF,F1,J,I_F,I_Tl, electronic_state='B', P = P, 
                            Omega = 1)
        for J,F1,F in zip(Jlist, F1list, Flist)
        for mF in ni_range(-F, F+1)
        ]
    return excited_states_approx