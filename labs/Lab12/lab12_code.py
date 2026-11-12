"""
 This script provides you the opprotunity play with different linear algebra solvers.
You will learn about the performance of the different solvers and how to build
test some test matrices to challenge the different methods. 
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


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 100
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)

     t = time.time()
     x = scila.solve(A, b)
     t = time.time() - t
     print(f'For N = {N}, Time for Scipy.solve: {t}')
     '''test = np.matmul(A, x)
     r = la.norm(test - b)'''

     t1 = time.time()
     LU, P, info = scila.lapack.dgetrf(A)
     t1 = time.time() - t1
     print(f'For N = {N}, Time for LU Factorization: {t1}')
     t2 = time.time()
     x = scila.lu_solve((LU, P), b)
     t2 = time.time() - t2
     print(f'For N = {N}, Time for LU solve: {t2}')
     print(f'Total for LU method: {t2 + t1}')
     
     #print(r)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 3
     M = 2
     A = create_rect(N,M)     
     b = np.random.rand(N,1)

     Q, R = scila.qr(A)
     c = np.matmul(Q.T, b)
     x = scila.solve(Q, c)
     #print(x)

     L = A.T@A
     print(scila.det(L))
     R = A.T@b
     x = scila.solve(L, R)
     #print(x)





     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
