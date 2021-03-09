import numpy as np
import matplotlib.pyplot as plt

pl = [8,8,10,10,10,16,22,24,26,32,30,28,22,18,16,16,20,24,28,34,38,30,22,12]


def monte_carlo(values, iterations):
    all_samples = []

    for j in range(iterations):
        one_sample = []
        for i in  values:
            x = np.random.normal(i, i*(1/3), 1)
            one_sample.append(x)
        all_samples.append(one_sample)

    return all_samples

results = monte_carlo(pl,1000)

#plot first sample
#plt.plot(range(24),results[0])

#plot first five samples
'''
fig, axs = plt.subplots(1, 5, figsize=(9, 3), sharey=True)
axs[0].plot(range(24),results[0])
axs[1].plot(range(24),results[1])
axs[2].plot(range(24),results[2])
axs[3].plot(range(24),results[3])
axs[4].plot(range(24),results[4])
'''

#get average of all samples
res = np.array(results)
res.mean(axis=0)

#plot averages
plt.plot(range(24),res.mean(axis=0))


size = 1000
costs = []
for i in range(0,size):
    load = np.random.normal(30, 30*(1/3))
    if load <= 20:
        # generate alone
        costs.append(load*1.20)
    else:
        # purchase electrictiy from retail
        costs.append(20*1.2 + 2*(load-20))

sum(costs)/size

plt.hist(test, bins=20)
plt.title(f'{size} samples')