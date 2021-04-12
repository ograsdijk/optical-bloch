from sympy import zeros, Symbol
from symbolic.utils import recursive_subscript

def generate_density_matrix_symbolic(levels):
    ρ = zeros(levels,levels)
    levels = levels
    for i in range(levels):
        for j in range(i,levels):
            # \u03C1 is unicode for ρ, 
            if i == j:
                ρ[i,j] = Symbol(u'\u03C1{0},{1}'. \
                format(recursive_subscript(i), recursive_subscript(j)))
            else:
                ρ[i,j] = Symbol(u'\u03C1{0},{1}'. \
                format(recursive_subscript(i), recursive_subscript(j)))
                ρ[j,i] = Symbol(u'\u03C1{1},{0}'. \
                format(recursive_subscript(i), recursive_subscript(j)))
    return ρ