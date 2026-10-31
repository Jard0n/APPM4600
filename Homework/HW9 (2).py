#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 500)
f = np.sin(x)

P6  = x - x**3/6 + x**5/120
R33 = (x - 7*x**3/60) / (1 + x**2/20)          # also equals R_{4,2}
R24 = x / (1 + x**2/6 + 7*x**4/360)

plt.semilogy(x, abs(f-P6),  label='Maclaurin P6')
plt.semilogy(x, abs(f-R33), label='Padé M=N=3 and M=4,N=2')
plt.semilogy(x, abs(f-R24), label='Padé M=2,N=4')
plt.xlabel('x'); plt.ylabel('|error|'); plt.legend(); plt.grid(True)
plt.show()


# In[ ]:




