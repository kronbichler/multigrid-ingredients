#!/usr/bin/env python3


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def plot(u, init, smoother_its, mg_cycles):
    plt.figure()
    n = u.shape[0] + 1
    x = np.linspace(0, 1, n + 1)
    uplot = np.zeros(n + 1)
    uplot[1:-1] = u
    initplot = np.zeros(n + 1)
    initplot[1:-1] = init
    plt.plot(x, initplot, 'k:')
    plt.plot(x, uplot)
    plt.xlabel('x')
    plt.ylabel('iterative solution u^' + str(mg_cycles))
    plt.title('multigrid ' + str(mg_cycles) + ' cycles and ' \
              + str(smoother_its) + ' smoothing steps')
    plt.legend(['Initial condition', 'u^' + str(mg_cycles)], loc = 'upper right')
    plt.show()

def solve_mg(A, b, u, diags, min_level, its, omega):
    L = A.shape[0] - 1
    for l in range(L, min_level, -1):
        # Jacobi method
        for i in range(its):
            u[l] = omega * b[l] / diags[l] + \
                (u[l] - omega * (A[l] @ u[l]) / diags[l])
            
        b[l-1] = P[l].T @ (b[l] - A[l] @ u[l])
        u[l-1] = np.zeros_like(u[l-1])
    
    u[min_level] = sparse.linalg.spsolve(sparse.csr_matrix(A[min_level]), b[min_level])
    #for i in range(its):
    #    u[min_level] = omega * b[min_level] / diags[min_level] + \
    #        (u[min_level] - omega * (A[min_level] @ u[min_level]) / \
    #         diags[min_level])
    
    for l in range(min_level+1, L + 1):
        u[l] += P[l] @ u[l - 1]
    
        for i in range(its):
            u[l] += omega * (b[l] - A[l] @ u[l]) / diags[l]


# Main program and control of solution parameters

L            = 7
ns           = 2**np.arange(0, L+1)
min_level    = 1
omega        = 0.5
smoother_its = 3
mg_cycles    = 3


A     = np.array(ns, dtype=object)
b     = np.array(ns, dtype=object)
P     = np.array(ns, dtype=object)
diags = np.array(ns, dtype=object)
for l in range(ns.shape[0]):
    n = ns[l]
    dx = 1 / n
    e = np.ones(n - 1)
    data = np.array([-e, 2 * e, -e])
    A[l] = (1/(dx) * sparse.dia_matrix((data, np.array([-1, 0, 1])),
                                       shape=(n-1, n-1)))
    diags[l] = 1/dx * 2 * e
    b[l] = np.zeros(n-1)

    if n > 1:
        P[l] = np.zeros([n - 1, int(n/2) - 1])
        for j in range(P[l].shape[1]):
            P[l][2 * j, j] = 1/2
            P[l][2 * j + 1, j] = 1
            P[l][2 * j + 2, j] = 1/2

# Set initial condition by some fixed random numbers
my_random = np.random.RandomState(124512)
init = my_random.rand(n-1)

u = b.copy()

u[L] = init.copy()

for i in range(mg_cycles):
    solve_mg(A, b, u, diags, min_level, smoother_its, omega)


plot(u[L], init, smoother_its, mg_cycles)
print('Maximal error in vector: ', np.linalg.norm(u[L], np.inf))