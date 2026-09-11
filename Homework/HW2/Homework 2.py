import math
import matplotlib.pyplot as plt
import numpy as np
import bisection_example
import fixedpt_example

#problem 3
x = 9.999999995 * 10 ** -10
y = math.e ** x
print (f'original: {y - 1}')
y2 = x + x ** 2 / 2 + x ** 3 / 6
print(f'taylor series: {y2:.17}')

#problem 4
f = lambda x: (math.sin(x) - 2 * x + 1)
[astar, ier] = bisection_example.bisection(f, 0, 2, 10 ** -8)
print(f'bisection result: {astar}')

#problem 5
f = lambda x: x ** 3 + x - 4
[xstar, iter] = bisection_example.bisection(f, 1, 4, 10 ** -3)

#problem 6
f = lambda x: (x - 4 * np.sin(2 * x) - 3)
x = np.linspace(-2.5, 10, 100)
plt.plot(x, f(x))
plt.plot(x, np.zeros((100,1)))
plt.show()

g = lambda x: -1 * np.sin(2 * x) + 5 / 4 * x -3 / 4
[xstar, ier] = fixedpt_example.fixedpt(g, -.8, 10 ** -10, 1000)
print(f'fixed point root 1: {xstar}')
[xstar, ier] = fixedpt_example.fixedpt(g, 1.8, 10 ** -10, 1000)
print(f'fixed point root 2: {xstar}')