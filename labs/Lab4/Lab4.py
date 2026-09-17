"""
 This script uses the bisection method to approximate the root of a
 scalar function.
"""

#############################################
"""
Copyright (C) 2025  Adrianna M. Gillman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
#############################################

# import libraries
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import newton_example
import bisection_example


def driver():
    # use routines
    f = lambda x: jnp.exp(x ** 2 + 7 * x - 30) - 1
    a = 2
    b = 4.5

    #    f = lambda x: np.sin(x)
    #    a = 0.1
    #    b = np.pi+0.1

    tol = 1e-11

    [p, astar, ier, it] = bisection(f, a, b, tol)
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))
    #print(f"the newton's iteration table is {p}")
    print(f'it took {it} iterations \n \n')

    [p, astar, ier, it] = newton_example.newton(f, jax.grad(f), 4.5, tol, 1000, 0)
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))
    # print(f"the newton's iteration table is {p}")
    print(f'it took {it} iterations \n \n')

    '''[astar, ier] = bisection_example.bisection(f, a, b, tol)
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))
    # print(f"the newton's iteration table is {p}")
    print(f'it took {it} iterations \n \n')'''


# define routines
def bisection(f, a, b, tol):
    #    Inputs:
    #     f,a,b       - function and endpoints of initial interval
    #      tol  - bisection stops when interval length < tol

    #    Returns:
    #      astar - approximation of root
    #      ier   - error message
    #            - ier = 1 => Failed
    #            - ier = 0 == success

    #     first verify there is a root we can find in the interval
    df = jax.grad(f)
    ddf = jax.grad(df)
    fa = f(a)
    fb = f(b);
    if (fa * fb > 0):
        ier = 1
        astar = a
        return [astar, ier]

    #   verify end points are not a root
    if (fa == 0):
        astar = a
        ier = 0
        return [astar, ier]

    if (fb == 0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5 * (a + b)
    while (abs(d - a) > tol):
        fd = f(d)
        dfd = df(d)
        ddfd = ddf(d)
        if abs(1 - (dfd ** 2 - fd * ddfd) / dfd ** 2) < 1:
            astar = d
            ier = 2
            return newton_example.newton(f,df,d,tol,1000, count)
            return [astar, ier]
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa * fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count = count + 1
    #      print('abs(d-a) = ', abs(d-a))

    astar = d
    ier = 0
    print('count = ', count)
    return [astar, ier]

driver()


