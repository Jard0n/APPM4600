import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

#question 1
f = lambda x: np.exp(x) * np.cos(x)
I_ex=-12.070346316389632
I_trap = [0]
I_simp = [0]
error = [np.inf]
h=[]
a=0
b=np.pi
n=2


def TrapQuad(a,b,n,f):
    h = (b-a)/n
    x = np.linspace(a,b,n + 1)
    y=f(x)
    return [h,h * (np.sum(y) - 0.5 * (y[0] + y[-1]))]

def SimpsonQuad(a, b, n, f):
    x = np.linspace(a, b, n + 1)       # n intervals -> n+1 points
    h = (b - a) / n
    y = f(x)
    return [h,(h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))]

while (error[-1] > 10 ** -10) and n < 10 ** 10:
    n*=2
    I_trap = (TrapQuad(a,b,n,f)[1])
    error.append(abs(I_ex-I_trap))
    h.append(TrapQuad(a,b,n,f)[0])

slope, intercept = np.polyfit(np.log(h), np.log(error[1:]), 1)
plt.loglog(h,error[1:])
plt.title('Trapezoidal Rule')
plt.annotate(f'slope = {slope}', xy=(h[-2], error[-1]))
plt.show()


n=2
error = [np.inf]
h=[]

while (error[-1] > 10 ** -10) and n < 10 ** 10:
    n *= 2
    I_simp = (SimpsonQuad(a, b, n, f))[1]
    error.append(abs(I_ex-I_simp))
    h.append(SimpsonQuad(a,b,n,f)[0])

slope, intercept = np.polyfit(np.log(h), np.log(error[1:]), 1)
plt.loglog(h,error[1:])
plt.title('Simpson Rule')
plt.annotate(f'slope = {slope}', xy=(h[-2], error[-1]))
plt.show()

#question 2
f = lambda x: 1 / (1+x ** 2)
a=-5
b=5
n=1291
I_trap = TrapQuad(a,b,n,f)[1]
n=107
I_simp = SimpsonQuad(a,b,n,f)[1]

print(f"Result of Trapezoidal Rule: {I_trap} "
      f"\nResult of Simpson Rule: {I_simp} "
      f"\nResult of Scipy Quad (10^-6): {integrate.quad(f,a,b)[0]}, taking {integrate.quad(f,a,b, full_output=1)[2]['neval']} evaluations"
      f"\nResult of Scipy Quad (10^-4): {integrate.quad(f,a,b,epsabs=10**-4)}, taking {integrate.quad(f,a,b,epsabs=10**-4,full_output=1)[2]['neval']} evaluations")