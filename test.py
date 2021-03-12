import math
import matplotlib.pyplot as plt
from numpy import log as ln

eps = 0.22
h = 0.008
p = 2.7 * 10 ** -2
ak = lambda x: 1 / x
c_p = 9.7667
T = 10

n_k = lambda k: (
        math.pow(1 / (h - eps), 2) * (
            c_p + 2 * p * math.pow(ak(k), -2) * math.pow(ln(k), 2))
)

ks = list(range(1, T + 1))

ns = []

for k in ks:
    ns.append(n_k(k))

plt.plot(ks, ns)
plt.show()
