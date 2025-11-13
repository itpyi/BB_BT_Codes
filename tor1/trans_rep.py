### Get representation of translation operators in a logical space
import sympy as sp
import numpy as np
import galois
# Variables over F_2
x, y = sp.symbols('x y')

def in_ideal(generators, s):
    G = sp.groebner(generators, x, y, order='lex', domain=sp.GF(2))
    _, rem = G.reduce(s)
    return sp.Poly(rem, x, y, domain=sp.GF(2)).is_zero

def get_vec(basis, generators, s):
    N = len(basis)
    for i in range(1<<N):
        poly = s
        # print(bin(i), poly)
        for j in range(N):
            poly += sp.poly(basis[j], domain=sp.GF(2)) * ((i>>j)&1)
            # print((i>>j)^1)
        if in_ideal(generators, poly):
            return bin(i)
    return 0

def get_m_rep(basis, generators, m):
    N = len(basis)
    M = np.zeros((N, N), dtype=int)
    for i in range(N):
        s = sp.poly(basis[i], x, y, domain=sp.GF(2)) * sp.poly(m, x, y, domain=sp.GF(2))
        v_bin = get_vec(basis, generators, s)
        v_int = int(v_bin, 2)
        print(v_bin, v_int)
        for j in range(N):
            M[j, i] = (v_int >> j) & 1
    return M

if __name__ == "__main__":
    generators = [
        "x**6 + 1",
        "y**6 + 1",
        "(x**3 + y + y**2)*(y**3 + x + x**2)"]
    basis = [
        "x**4 + x**3*y + x*y**4 + x*y + x + y**4 + y**3 + 1",
        "x**4*y + x**3*y**2 + x*y**5 + x*y**2 + x*y + y**5 + y**4 + y",
        "x**4*y + x**4 + x**3 + x*y**3 + x*y + y**4 + y + 1",
        "x**4 + x**3*y**3 + x**3*y + x**3 + x*y**4 + x*y + x + y**4"]
    X = get_m_rep(basis, generators, x)
    X = galois.GF(2)(X)
    M = galois.GF(2)(np.eye(4, dtype=int))
    for i in range(6):
        M = M @ X
        print("X^{}: \n{}".format(i+1, M))