import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm
import newtonNONLinear

def SlackerNewton(x0, tol, Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)

        if norm(inv(J) * F) < 1:
            J = evalJ(x0)
            Jinv = inv(J)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)

       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]

       x0 = x1

    xstar = x1
    ier = 1
    return[xstar,ier,its]


def evalF(x):
    # vector function that you want to find the roots of

    F = np.zeros(2)

    F[0] = 4 * x[0] ** 2 + x[1] ** 2 - 4
    F[1] = x[0] + x[1] - math.sin(x[0] - x[1])

    return F


def evalJ(x):
    # Jacobian of the vector function you want to find the roots of

    J = np.array([[8 * x[0], 2 * x[1]],
                  [1 - math.cos(x[0] - x[1]), 1 + math.cos(x[0] - x[1])]])

    return J

x = [1,0]

t = time.time()
for j in range(50):
    [xstar,ier,its] = LazyNewton(x,10 ** -10, 100)
elapsed = time.time()-t
print(f"lazy newton took {elapsed} seconds to complete 50 runs")
print(f'lazy newton took {its} iterations')

t = time.time()
for j in range(50):
    [xstar,ier,its] = SlackerNewton(x,10 ** -10, 100)
elapsed = time.time()-t
print(f"slacker newton took {elapsed} seconds to complete 50 runs")
print(f'slacker newton took {its} iterations')