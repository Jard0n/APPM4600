import scipy
import math
import matplotlib.pyplot as plt
import numpy as np
import bisection_example
import newton_example
import fixedpt_example

#question 2
f = lambda x: scipy.special.erf(x/1.519)-0.429
df = lambda x: 0.7428 * math.exp(-.0433 * x ** 2)
x = np.linspace(0,1,100)
plt.plot(x,f(x))
plt.plot(x, x * 0, '--')


[astar, ier] = bisection_example.bisection(f,0,1,10 ** -11)
print(f'The approximate root from bisection is {astar} meters')

[p, astar, ier, count] = newton_example.newton(f, df, 0, 10 ** -11, 1000, 0)
print(f'the approximate root from newton is {astar} meters \n it took {count} iterations of newton')

#question 3
a = lambda x: x * (1 + (7 - x ** 5) / x ** 2) ** 3
[xstar, ier] = fixedpt_example.fixedpt(a,1,10 ** -10, 100)
print(f'a: error message {ier}')

b = lambda x: x - (x ** 5 - 7) / x ** 2
[xstar, ier] = fixedpt_example.fixedpt(b,1,10 ** -10, 100)
print(f'b: error message {ier}')

c = lambda x: x - (x ** 5 - 7) / (5 * x ** 4)
[xstar, ier] = fixedpt_example.fixedpt(c,1,10 ** -10, 100)
print(f'c: error message {ier}')

d = lambda x: x - (x ** 5 - 7) / 12
[xstar, ier] = fixedpt_example.fixedpt(d,1,10 ** -10, 100)
print(f'd: error message {ier}')

plt.show()