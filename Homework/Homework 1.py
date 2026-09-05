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


