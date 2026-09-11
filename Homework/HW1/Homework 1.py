import matplotlib.pyplot as plt
import numpy as np
import math
x = np.arange(1.920,2.08,.001)
p1 = lambda t: (t ** 9 - 18 * x ** 8 + 144 * x ** 7 - 672 * x ** 6 + 2016 * x ** 5
                - 4032 * x ** 4 + 5376 * x ** 3 - 4608 * x ** 2 + 2304 *  x - 512)
plt.plot(x,p1(x))
p2 = lambda t: (t-2) ** 9
plt.plot(x, p2(x))
plt.legend(("polynomial", "(x-2) ^ 9"))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


x = np.pi
y = 10 ** 6
delta = np.float_power(10, np.arange(0,-16,-1))
f1 = np.cos(x + delta) - np.cos(x)
f2 = np.cos(x + delta) + np.cos(x + np.pi)
f3 = np.cos(y + delta) - np.cos(y)
f4 = np.cos(y + delta) + np.cos(y + np.pi)
difference1 = np.abs(f2 - f1)
difference2 = np.abs(f3 - f4)
plt.semilogx(delta, difference1)
plt.semilogx(delta, difference2)
plt.show()
print(difference2)
