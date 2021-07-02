import multiprocessing
from symbolic.utils import multi_system_of_equations_to_lines
from symbolic.density_matrix import generate_density_matrix_symbolic

def system_of_equations_to_lines(system, nprocs = 1):
    n_states = system.shape[0]
    ρ = generate_density_matrix_symbolic(n_states)

    if nprocs > 1:
        with multiprocessing.Pool(processes = nprocs) as pool:
            result = pool.starmap(multi_system_of_equations_to_lines,
                                [(system, ρ, idx) for idx in range(n_states)])
        code_lines = [item for sublist in result for item in sublist]
    else:
        code_lines = []
        for idx in range(n_states):
            for idy in range(n_states):
                if system[idx,idy] != 0:
                    cline = str(system[idx,idy])
                    cline = f"du[{idx+1},{idy+1}] = " + cline
                    cline = cline.replace("(t)", "")
                    cline = cline.replace("(t)", "")
                    cline = cline.replace("I", "1im")
                    cline += '\n'
                    for i in range(system.shape[0]):
                        for j in range(system.shape[1]):
                            _ = str(ρ[i,j])
                            cline = cline.replace(_+"*", f"ρ[{i+1},{j+1}]*")
                            cline = cline.replace(_+" ", f"ρ[{i+1},{j+1}] ")
                            cline = cline.replace(_+"\n", f"ρ[{i+1},{j+1}]")
                            cline = cline.replace(_+")", f"ρ[{i+1},{j+1}])")
                    cline = cline.strip()
                    code_lines.append(cline)

    return code_lines