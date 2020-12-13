import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
### Matplotlib Settings
###############################################################################

settings = {
    'text.usetex': True,
    'font.weight' : 'normal',
    'font.size'   : 14
}
plt.rcParams.update(**settings)

# resolution for plots
dpi = 300

# path to csv files
path = '../3_results/'

# saving path
saving_path = '../4_plots/'

###############################################################################
### Plots
###############################################################################

objective_values = np.loadtxt(
     fname=path + 'objective_values_no_sensitivity.csv',
     delimiter=','
)

bounds_differences = np.loadtxt(
     fname=path + 'bounds_differences_no_sensitivity.csv',
     delimiter=','
)

iterations = range(1, len(objective_values)+1)

plt.plot(iterations, bounds_differences)
plt.grid()
plt.yscale("log")
plt.show()