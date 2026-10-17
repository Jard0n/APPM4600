#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 1 / (1+ (10 * x) ** 2)
pltx = np.linspace(-1,1, 1001)
n=20
i = np.linspace(1,n, n)
h = 2/(n-1)
x = np.array(-1 + (i-1) * h)
plt.plot(pltx,f(pltx))
plt.plot(x,f(x),'o')
V = np.zeros([n,n])
for p in range(n):
    for q in range(n):
        V[p, q] = x[p] ** q
inv = np.linalg.inv(V)
c = np.matmul(inv,f(x))
#print(c)

def p(x):
    output = 0
    for i in range(len(c)):
        output += c[i] * x ** (i)
    return output
plt.plot(pltx,p(pltx))
#print(V)
plt.show()


# In[49]:


def phi(z):
    output = 1
    for i in range(n):
        output *= (z-x[i])
    return output

def w(j):
    output = 1
    for i in range(n):
        if i != j:
            output /= (x[j]-x[i])
    return output

def p(z):
    
    output = 0
    for i in range(n):
        for y in z:
            if y != x[i]:
                output += w(i) * f(x[i])/(y-x[i])
    return output * phi(z)

plt.plot(pltx,p(pltx))
plt.plot(pltx,f(pltx))
plt.plot(x,f(x),'o')
plt.show()


# In[51]:


cheb = np.cos((2 * i - 1) * np.pi / 2 * n) 
plt.plot(cheb,p(cheb))
plt.plot(pltx,f(pltx))
plt.plot(x,f(x),'o')
plt.show()


# In[ ]:




