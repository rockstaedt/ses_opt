###############################################################################
### Plotting
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.lines import Line2D
import json
from pathlib import Path
import os

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

current_path = Path.cwd()

# path to csv files
results_path = os.path.join(current_path.parent, '3_results')

# saving path
saving_path = os.path.join(current_path.parent, '4_plots')

###############################################################################
### Computation time all tasks, all sample sizes
###############################################################################

sample_sizes = [10, 100, 1000, 10000]

computation_times = {'task_'+str(i): [] for i in range(1,4)}

for task in range(1,4):
    for sample_size in sample_sizes:
        df = pd.read_csv(
            os.path.join(
                results_path,
                'stochastic',
                'task_'+str(task),
                str(sample_size),
                'computation_times.csv'
            )
        )
        computation_times['task_'+str(task)].append(df.iloc[1,1])

plt.plot(
    sample_sizes,
    computation_times['task_1'],
    label='Task 1',
    marker='x')
plt.plot(
    sample_sizes,
    computation_times['task_2'],
    label='Task 2',
    marker='x')
plt.plot(
    sample_sizes,
    computation_times['task_3'],
    label='Task 3',
    marker='x')
plt.legend()
plt.xlabel('Sample Size')
plt.ylabel('Computation time in [s]')
plt.xscale('log')

plt.savefig(
    os.path.join(saving_path, 'computation_times.png'),
    dpi=dpi,
    bbox_inches='tight'
)


