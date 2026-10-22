import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
from scipy.integrate import quad

def eval_legendre(n,x):
    if n == 0:
        return np.array([1])
    if n == 1:
        return np.array([1,x])
    phi = np.zeros(n+1)
    phi[0]=1
    phi[1]=x
    for i in range(2, n+1):
        phi[i] = 1/(i) * ((2*(i-1) + 1) * x*phi[i-1] - (i-1)*phi[i-2])
    return phi

def eval_a(f, j, x, w,a,b):
    phi = eval_legendre(j,x)[j]
    numer = quad(phi * f(x) * w(x), a,b)
    denom = quad(phi ** 2 * w(x), a, b)
    return numer / denom

def eval_chebychev(n,x):
    if n == 0:
        return np.array([1])
    if n == 1:
        return np.array([1,x])
    T = np.zeros(n+1)
    T[0]=1
    T[1]=x
    for i in range(2, n+1):
        T[i] = 2 * x * T[i-1]-T[i-2]
    return T


def driver():
    #  function you want to approximate
    f = lambda x: 1/(1+x**2)

    # Interval of interest
    a = -1
    b = 1
    # weight function
    w = lambda x: 1.

    # order of approximation
    n = 3

    #  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a, b, N + 1)
    pval = np.zeros(N + 1)

    for kk in range(N + 1):
        pval[kk] = eval_legendre_expansion(f, a, b, w, n, xeval[kk])

    ''' create vector with exact values'''
    fex = np.zeros(N + 1)
    for kk in range(N + 1):
        fex[kk] = f(xeval[kk])

    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, pval, 'bs--', label='Expansion')
    plt.legend()
    plt.show()

    err = abs(pval - fex)
    plt.semilogy(xeval, err, 'ro--', label='error')
    plt.legend()
    plt.show()

    w2 = lambda x: 1/(np.sqrt(1-x**2))
    for kk in range(N + 1):
        pval[kk] = eval_chebychev_expansion(f, a, b, w2, n, xeval[kk])

    ''' create vector with exact values'''
    fex = np.zeros(N + 1)
    for kk in range(N + 1):
        fex[kk] = f(xeval[kk])

    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, pval, 'bs--', label='Expansion')
    plt.legend()
    plt.show()

    err = abs(pval - fex)
    plt.semilogy(xeval, err, 'ro--', label='error')
    plt.legend()
    plt.show()


def eval_chebychev_expansion(f, a, b, w, n, x):
    #   This subroutine evaluates the Legendre expansion

    #  Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab
    p = eval_chebychev(n,x)
    # initialize the sum to 0
    pval = 0.0
    for j in range(0, n + 1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_chebychev(j, x)[j]
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x) ** 2 * w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac, err = quad(phi_j_sq, -1, 1)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: phi_j(x) * f(x) * w(x) / norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj, err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval + aj * p[j]

    return pval

def eval_legendre_expansion(f, a, b, w, n, x):
    #   This subroutine evaluates the Legendre expansion

    #  Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab
    p = eval_legendre(n,x)
    # initialize the sum to 0
    pval = 0.0
    for j in range(0, n + 1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_legendre(j, x)[j]
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x) ** 2 * w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac, err = quad(phi_j_sq, -1, 1)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: phi_j(x) * f(x) * w(x) / norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj, err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval + aj * p[j]

    return pval


if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
