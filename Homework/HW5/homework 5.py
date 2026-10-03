import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm
import newtonNONLinear as nt


f = lambda x,y: 3 * x ** 2 - y ** 2
g = lambda x,y: 3 * x * y ** 2 - x ** 3 - 1
j = lambda x,y: [[6*x,-2*y],[3*x ** 2 - 3 *y **2, 6 * x * y]]
tol = 10 ** -10

x = np.array([1,1])
xs = [x]
for i in range(100):
    x = np.array([x[0]-(1/6 * f(x[0], x[1]) + 1/18*g(x[0], x[1])), x[1] - 1/6 * g(x[0], x[1])])
    xs.append(x)
    if i > 3 and norm(xs[-1] - xs[-2]) < tol:
        print(i)
        break

print(xs[-5:-1])


def Newton(x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = j(x0[0],x0[1])
        Jinv = inv(J)
        F = [f(x0[0],x0[1]), g(x0[0],x0[1])]

        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]

[xstar, ier, its] = Newton([1,1], tol, 100000)
print(xstar, its)