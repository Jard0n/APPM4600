#!/usr/bin/env python
# coding: utf-8

# In[44]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.special as special



def driver2():
    f = lambda x: np.sin(x)
    a=0
    b=np.pi
    n=4
    ntest = 2**np.arange(0,n+1)
    I_ex = 2
    I_simp = rombergSimp(a,b,n,f)
    I_trap = rombergTrap(a,b,n,f)
    err_simp = abs(I_ex-np.diag(I_simp))
    err_trap = abs(I_ex-np.diag(I_trap))
    

    plt.semilogy(ntest,err_simp,'o-')
    plt.semilogy(ntest,err_trap,'o-')
    plt.xlabel('n')
    plt.ylabel('absolute error')
    plt.legend(('Romberg Simpson', 'Romberg Trapezoid'))
    plt.show()

def driver3():
    a = 0
    b = 1
    n = 4
    I_simp = SimpsonQuad(a,b,n,g)
    print(f"Transformed Integral Approx: {I_simp}")

def driver4():
    tol = 1e-10
    for x in [1.0, 2.0, 3.5, 5.0, 10.0]:
        exact = special.gamma(x)
        print(f"Scipy Gamma: {exact}")
        print(f"x = {x},  T = {choose_T(x, tol)}")
        for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            approx = gamma_trap(x, h, tol)
            print(f"h = {h} Gamma = {approx}  err = {abs(approx - exact)}")
            for n in [2, 4, 8, 16, 32, 64]:
                t, w = np.polynomial.laguerre.laggauss(n)
                approx = np.sum(w * t **(x-1))
                print(f"n = {n} Gamma = {approx}  err = {abs(approx)}")
    

def integrand(t, x):
    out = np.empty_like(t)
    pos = t > 0
    out[pos] = t[pos]**(x - 1) * np.exp(-t[pos])
    if x > 1:
        out[~pos] = 0.0
    elif x == 1:
        out[~pos] = 1.0
    return out

def choose_T(x, tol):
    T = max(x, 1.0)
    while T**(x - 1) * np.exp(-T) > tol:   # tail estimate
        T += 1.0
    return T

def TrapQuad(a, b, n, f):
    t = np.linspace(a, b, n + 1)
    y = f(t)
    h = (b - a) / n
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def gamma_trap(x, h, tol):
    T = choose_T(x, tol)
    n = int(np.ceil(T / h))
    return TrapQuad(0.0, T, n, lambda t: integrand(t, x))

def TrapQuad(a,b,n,f):
    h = (b-a)/n
    x = np.linspace(a,b,n + 1)
    y=f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    
def SimpsonQuad(a, b, n, f):
    x = np.linspace(a, b, n + 1)       # n intervals -> n+1 points
    h = (b - a) / n
    y = f(x)
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

def rombergSimp(a,b,nrefine,f):
# number of panels: 1, 2, 4, ..., 2^nrefine
    ntest = 2**np.arange(0,nrefine+1)

    nn = len(ntest)

    Rom = np.zeros((nn,nn))

# first column: composite trapezoidal rule with n panels
    for j in range(0,nn):
        Rom[j,0] = SimpsonQuad(a,b,ntest[j],f)

# remaining columns: Richardson extrapolation
    for j in range(1,nn):
        for i in range(j,nn):
            Rom[i,j] = Rom[i,j-1] + (Rom[i,j-1]-Rom[i-1,j-1])/(4**(j+1)-1)

    return Rom

def rombergTrap(a,b,nrefine,f):
# number of panels: 1, 2, 4, ..., 2^nrefine
    ntest = 2**np.arange(0,nrefine+1)

    nn = len(ntest)

    Rom = np.zeros((nn,nn))

# first column: composite trapezoidal rule with n panels
    for j in range(0,nn):
        Rom[j,0] = TrapQuad(a,b,ntest[j],f)

# remaining columns: Richardson extrapolation
    for j in range(1,nn):
        for i in range(j,nn):
            Rom[i,j] = Rom[i,j-1] + (Rom[i,j-1]-Rom[i-1,j-1])/(4**j-1)

    return Rom




driver2()
driver3()
driver4()




# In[ ]:





# In[ ]:




