from numpy.linalg import norm
import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv
import numpy as np
import time
jax.config.update('jax_enable_x64', True)


#problem 1

f = lambda x: jnp.array([3 * x[0] ** 2 + 4 * x[1] ** 2 - 1, x[1] ** 3 - 8 * x[0] ** 3 - 1])
j = jax.jacobian(f)

g = lambda x: x - jnp.matmul(jnp.array([[.016, -.17],[.52,-.26]]), f(x))

def fixedSys(x0, f, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        try:
            x1 = f(x0)
        except:
            ier = 1
            print('overflow divergence')
            return [x0, ier, its]

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def Newton(x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        Jx = j(x0)
        Jinv = jnp.linalg.inv(Jx)
        Fx = f(x0)

        x1 = x0 - Jinv.dot(Fx)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def LazyNewton(x0, tol, Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    Jx = j(x0)
    Jinv = inv(Jx)
    for its in range(Nmax):

        F = f(x0)
        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            ier = 0
            return [xstar, ier, its]

        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, its]


def Broyden(x0, tol, Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''

    '''In Broyden 
    x = [F(xk)-F(xk-1)-hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''

    '''initialize with 1 newton step'''

    A0 = j(x0)

    v = f(x0)
    A = inv(A0)

    s = -A.dot(v)
    xk = x0 + s
    for its in range(Nmax):
        '''(save v from previous step)'''
        w = v
        ''' create new v'''
        v = f(xk)
        '''y_k = F(xk)-F(xk-1)'''
        y = v - w;
        '''-A_{k-1}^{-1}y_k'''
        z = -A.dot(y)
        ''' p = s_k^tA_{k-1}^{-1}y_k'''
        p = -np.dot(s, z)
        u = np.dot(s, A)
        ''' A = A_k^{-1} via Morrison formula'''
        tmp = s + z
        tmp2 = np.outer(tmp, u)
        A = A + 1. / p * tmp2
        ''' -A_k^{-1}F(x_k)'''
        s = -A.dot(v)
        xk = xk + s
        if (norm(s) < tol):
            alpha = xk
            ier = 0
            return [alpha, ier, its]
    alpha = xk
    ier = 1
    return [alpha, ier, its]

def readout(x0):

    [xstar, ier, its] = fixedSys(x0, g, 10 ** -4, 100)
    jx = j(x0)
    j_inv = jnp.linalg.inv(jx)

    if ier:
        print(f'fixed point does not converge with input {x0}')
    else:
        print(f'fixed point converges to {xstar} in {its} iterations with input {x0}')
    print(f' the inverse jacobian at input {x0} = {j_inv}')

    [xstar, ier, its] = Newton(x0, 10 ** -4, 100)
    if ier:
        print(f'newton does not converge with input {x0}')
    else:
        print(f'newton converges to {xstar} in {its} iterations with input {x0}')
    print(f' the inverse jacobian at input {x0} = {j_inv}')


x = [jnp.array([-.5,.25]),jnp.array([-1.5,1.25]),jnp.array([-3.0,3.0])]
print('PROBLEM 1')
for thing in x:
    readout(thing)
print('\n')

def readout2(x0):

    t = time.time()
    [xstar, ier, its] = LazyNewton(x0, 10 ** -4, 100)
    t2 = time.time() - t
    if ier:
        print(f'Lazy Newton does not converge with input {x0}')
    else:
        print(f'Lazy Newton converges to {xstar} in {its} iterations and {t2} seconds with input {x0}')
    t = time.time()
    [xstar, ier, its] = Broyden(x0, 10 ** -4, 100)
    t2 = time.time() - t
    if ier:
        print(f'Broyden does not converge with input {x0}')
    else:
        print(f'Broyden converges to {xstar} in {its} iterations and {t2} seconds with input {x0}')
    t = time.time()
    [xstar, ier, its] = Newton(x0, 10 ** -4, 100)
    t2 = time.time() - t
    if ier:
        print(f'Newton does not converge with input {x0}')
    else:
        print(f'Newton converges to {xstar} in {its} iterations and {t2} seconds with input {x0}')


f = lambda x: jnp.array([x[0] ** 2 + x[1] ** 2 - 4,jnp.exp(x[0]) + x[1] - 1])
j = jax.jacobian(f)
x = [jnp.array([1.0,1.0]),jnp.array([1.0,-1.0]),jnp.array([0.0,0.0])]
print('PROBLEM 2')
for thing in x:
    readout2(thing)
print('\n')


f = lambda x: jnp.array([x[0]+jnp.cos(x[0]*x[1]*x[2])-1,
                         (1-x[0]) ** .25 + x[1] + .05 * x[2] ** 2 - .15 * x[2] - 1,
                         -x[0] ** 2 - .1 * x[1] ** 2 + .01 * x[2] + x[3] - 1])
g = lambda x: f(x)[0] ** 2 + f(x)[1] ** 2 + f(x)[2] ** 2
j = jax.jacobian(f)


def SteepestDescent(x, tol, Nmax):
    for its in range(Nmax):
        g1 = g(x)
        z = np.transpose(j(x)).dot(f(x))
        z0 = norm(z)

        # below is finding the step-length using the quadratic min.
        if z0 == 0:
            print("zero gradient")
        z = z / z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3 * z
        g3 = g(dif_vec)

        while g3 >= g1:
            alpha3 = alpha3 / 2
            dif_vec = x - alpha3 * z
            g3 = g(dif_vec)

        if alpha3 < tol:
            print("no likely improvement")
            ier = 0
            return [x, g1, ier, its]

        alpha2 = alpha3 / 2
        dif_vec = x - alpha2 * z
        g2 = g(dif_vec)

        h1 = (g2 - g1) / alpha2
        h2 = (g3 - g2) / (alpha3 - alpha2)
        h3 = (h2 - h1) / alpha3

        alpha0 = 0.5 * (alpha2 - h1 / h3)
        dif_vec = x - alpha0 * z
        g0 = g(dif_vec)

        if g0 <= g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval = g3
        #####
        x = x - alpha * z

        if abs(gval - g1) < tol:
            ier = 0
            return [x, gval, ier, its]

    print('max iterations exceeded')
    ier = 1
    return [x, g1, ier, its]

def SteepestNetwon(x,tol,Nmax):
    [x,g,ier,its] = SteepestDescent(x,5 * 10 ** -2, Nmax)
    [x,ier,its2] =  Newton(x,tol,Nmax-ier)
    return [x,ier,its + its2]


def readout3(x0):
    t = time.time()
    [xstar, g, ier, its] = SteepestDescent(x0, 10 ** -6, 100)
    t2 = time.time() - t
    if ier:
        print(f'Steepest Descent does not converge with input {x0}')
    else:
        print(f'Steepest Descent converges to {xstar} in {its} iterations and {t2} seconds with input {x0}')
    t = time.time()
    [xstar, ier, its] = Newton(x0, 10 ** -6, 100)
    t2 = time.time() - t
    if ier:
        print(f'Newton does not converge with input {x0}')
    else:
        print(f'Newton converges to {xstar} in {its} iterations and {t2} seconds with input {x0}')
    t = time.time()
    [xstar, ier, its] = SteepestNetwon(x0, 10 ** -6, 100)
    t2 = time.time() - t
    if ier:
        print(f'Steepest Newton does not converge with input {x0}')
    else:
        print(f'Steepest Newton converges to {xstar} in {its} iterations and {t2} seconds with input {x0}')

x = jnp.array([0.0,0.0,0.0])
t = time.time()
[xstar, ier, its] = Newton(x,10 ** -6, 100)
t2 = time.time() - t
print('PROBLEM 3')
readout3(x)