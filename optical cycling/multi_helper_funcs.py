import numpy as np
from scipy.integrate import solve_ivp

def multi_solve_ode(matrix_eq, tspan, y0, method = 'BDF', max_step = 1e-1):
    fun = lambda t, ρ: matrix_eq@ρ
    try:
        sol = solve_ivp(fun, tspan, y0, method, vectorized = True, max_step = max_step)
        return sol
    except Exception as e:
        print(e)
        return np.nan
    
def multi_solve_ode_integrate_excited(matrix_eq, tspan, y0, levels, method = 'BDF', max_step = 1e-1):
    fun = lambda t, ρ: matrix_eq@ρ
    try:
        sol = solve_ivp(fun, tspan, y0, method, vectorized = True, max_step = max_step)
        return np.trapz(np.sum(sol.y[levels].real, axis = 0), sol.t)
    except Exception as e:
        print(e)
        return np.nan