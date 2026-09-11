"""
 This script explores the use of the fixed point method.  
 Two functions are considered that have different properties.
 I like to use this code before I talk about convergence analysis
 for the fixed point method as motivation.
"""
import math

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
import math
    
def driver():


# test functions 
     f1 = lambda x: (10 / (x + 4)) ** .5
# fixed point is alpha1 = 1.4987....

     f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09... 

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar[-1])
     print('f1(xstar):',f1(xstar[-1]))
     print('Error message reads:',ier)
     print(len(xstar))
     #print(xstar)
     error_vec = abs((np.array(xstar)-xstar[-1]))
     alpha = np.log(error_vec[-2] / error_vec[-3]) / np.log(error_vec[-3] / error_vec[-4])
     print('alpha equals', alpha)

     new_xstar = aitken(xstar, tol, Nmax)
     print(new_xstar)


'''#test f2 
     x0 = 0.0
     [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar[-1])
     print('f2(xstar):',f2(xstar[-1]))
     print('Error message reads:',ier)
'''


# define routines
def fixedpt(f,x0,tol,Nmax):
    xstar = []
    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)

       xstar.append(x0)
       if (abs(x1-x0) <tol):
          ier = 0
          return [xstar,ier]
       x0 = x1

    ier = 1
    print(len(xstar))
    return [xstar, ier]

def aitken(seq, tol, Nmax):
    n = len(seq)
    output = np.zeros((n, 1))
    for i in range(n, n-2):
        output[i] = seq[i] - ((seq[i+1] - seq[i]) ** 2) / (seq[i + 2] - 2 * seq[i + 1] + seq[i])
        if abs(output[i] - output[i-1]) < tol or i >= Nmax:
            print(i)
            break
    return output

driver()
`