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

from model_options import output

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

sample_sizes = [10, 100, 1000, 10000, 100000]

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

fig1 = plt.figure(1)
ax1 = fig1.gca()

ax1.plot(
    sample_sizes,
    computation_times['task_1'],
    label='Case 1',
    marker='x',
    color='darkred')
ax1.plot(
    sample_sizes,
    computation_times['task_2'],
    label='Case 2',
    marker='x')
ax1.plot(
    sample_sizes,
    computation_times['task_3'],
    label='Case 3',
    marker='x',
    color='darkgreen')
ax1.legend()
ax1.set_xlabel('Sample size')
ax1.set_ylabel('Computation time in [s]')
ax1.set_xscale('log')
ax1.set_yscale('log')

if output:
    fig1.savefig(
        os.path.join(saving_path, 'computation_times.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Sensitivity plot for sample size = 10000
###############################################################################

from model_options import sensitivity_analysis
sensitivity_analysis = True
from parameters import stor_levels_max
from parameters import test_size

sample_size = 10000

sense_path = os.path.join(
    results_path,
    'stochastic',
    'task_4',
    str(sample_size),
    'sensitivity analysis',
    'testing'
)

generation_mix = ['p1', 'p2', 'pg']

stor_level_to_mean = {}
stor_level_to_var = {}
stor_level_to_gen_mix = {
    stor_level: {generation: [] for generation in generation_mix}
    for stor_level in stor_levels_max
}
stor_level_to_gen_mix_mean = {stor_level: {} for stor_level in stor_levels_max}

for stor_level in stor_levels_max:
    # mean and variance
    path = os.path.join(
        sense_path,
        'mean_var_' + str(stor_level) + '_' + str(test_size) + '.csv'
    )
    df = pd.read_csv(path)
    stor_level_to_mean[stor_level] = df.iloc[0,0]
    stor_level_to_var[stor_level] = df.iloc[0,1]

    # generation mix
    path = os.path.join(
        sense_path,
        'testing_results_' + str(stor_level) + '_' + str(test_size) + '.json'
    )
    with open(path) as f:
        # returns JSON object as a dictionary
        results = json.load(f)

    for sample, dic_var in results.items():
        pg = 0
        p1 = 0
        p2 = 0
        for var, dic_values in dic_var.items():
            if var == 'pg':
                for hour, value in dic_values.items():
                    pg += float(value)
            if var == 'p1':
                for hour, value in dic_values.items():
                    p1 += float(value)
            if var == 'p2':
                for hour, value in dic_values.items():
                    p2 += float(value)
        stor_level_to_gen_mix[stor_level]['pg'].append(pg)
        stor_level_to_gen_mix[stor_level]['p1'].append(p1)
        stor_level_to_gen_mix[stor_level]['p2'].append(p2)

    stor_level_to_gen_mix_mean[stor_level]['pg'] = np.mean(
        stor_level_to_gen_mix[stor_level]['pg']
    )
    stor_level_to_gen_mix_mean[stor_level]['p1'] = np.mean(
        stor_level_to_gen_mix[stor_level]['p1']
    )
    stor_level_to_gen_mix_mean[stor_level]['p2'] = np.mean(
        stor_level_to_gen_mix[stor_level]['p2']
    )

means = [stor_level_to_mean[stor_level] for stor_level in stor_levels_max]
stor_levels_string = [str(stor_level) for stor_level in stor_levels_max]

fig2 = plt.figure(2)
ax2 = fig2.gca()

ax2.bar(stor_levels_string, means)

ax2.set_xlabel('Storage capacity in kWh')
ax2.set_ylabel('Objective value in \$')
ax2.set_ylim([85,100])

if output:
    fig2.savefig(
        os.path.join(saving_path, 'sense_analyse.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Bound Development
###############################################################################

datau = pd.read_csv(
    os.path.join(
        results_path,
        'stochastic',
        'task_3',
        '10000',
        'objective_values_4.csv'
    ),
    header = None
)
datal = pd.read_csv(
    os.path.join(
        results_path,
        'stochastic',
        'task_3',
        '10000',
        'lower_bounds_4.csv'
    ),
    header = None
)
xu = datau.values
xl = datal.values

b = np.arange(2,7)

fig3 = plt.figure(3, figsize=(12, 7))
ax3 = fig3.gca()

# Create an axes instance
ax3 = fig3.add_subplot(111)

#ax.plot(b, xu[0],  b, xl[0])
ax3.plot(b, xu[0][2:],  b, xl[0][2:])
ax3.set_xlabel('Iterations',fontsize = 15)
ax3.set_ylabel('Value',fontsize = 15)
ax3.legend(['upper bound','lower bound'],fontsize=15)

if output:
    fig3.savefig(
        os.path.join(saving_path, 'bound_evolution.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Variance
###############################################################################

means1 = []
varis1 = []

for i in [10,100,1000,10000,100000]:
    data_mv = pd.read_csv(
        os.path.join(
            results_path,
            'stochastic',
            'task_1',
            str(i),
            'testing',
            'mean_var_4_1000.csv'
        ),
        header = None)
    x = data_mv.values
    means1.append(float(x[1][0]))
    varis1.append(float(x[1][1]))

means2 = []
varis2 = []

for i in [10,100,1000,10000,100000]:
    data_mv = pd.read_csv(
        os.path.join(
            results_path,
            'stochastic',
            'task_2',
            str(i),
            'testing',
            'mean_var_4_1000.csv'
        ),
        header = None)
    x = data_mv.values
    means2.append(float(x[1][0]))
    varis2.append(float(x[1][1]))

means3 = []
varis3 = []

for i in [10,100,1000,10000,100000]:
    data_mv = pd.read_csv(
        os.path.join(
            results_path,
            'stochastic',
            'task_3',
            str(i),
            'testing',
            'mean_var_4_1000.csv'
        ),
        header = None)
    x = data_mv.values
    means3.append(float(x[1][0]))
    varis3.append(float(x[1][1]))

sz = [10,100,1000,10000,100000]
b = np.arange(1,6)

means1 = np.array(means1)
varis1 = np.sqrt(np.array(varis1))
yplus1 = means1+varis1
yminus1 = means1-varis1

means2 = np.array(means2)
varis2 = np.sqrt(np.array(varis2))
yplus2 = means2+varis2
yminus2 = means2-varis2

means3 = np.array(means3)
varis3 = np.sqrt(np.array(varis3))
yplus3 = means3+varis3
yminus3 = means3-varis3

fig4, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12, 7),sharex=True)

ax1.bar(b,varis1,color='darkred',label='Case 1')
ax1.set_xticks(b,sz)
ax1.set_ylim([2.5,3.5])
ax1.legend()


ax2.bar(b,varis2,label='Case 2')
ax2.set_xticks(b,sz)
ax2.set_ylim([2.5,3.5])
ax2.set_ylabel('Standard Deviation',fontsize=15)
ax2.legend()


ax3.bar(b,varis3,color='darkgreen',label='Case 3')
ax3.set_ylim([2.5,3.5])
ax3.set_xlabel('Sample Size', fontsize=15)
ax3.legend()

plt.xticks(b,sz)

if output:
    fig4.savefig(
        os.path.join(saving_path, 'standard_deviations.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Mean
###############################################################################

fig5 = plt.figure(figsize=(12,7))
ax5 = fig5.gca()
ax5.plot(b,means1,color='darkred',label='Case 1')
ax5.plot(b,means2,label='Case 2')
ax5.plot(b,means3,color='darkgreen',label='Case 3')


ax5.set_xlabel('Sample Size',fontsize=15)
ax5.set_ylabel('Objective Value',fontsize=15)
ax5.set_xticks(b,sz)
ax5.grid()
ax5.legend(fontsize=13)

if output:
    fig5.savefig(
        os.path.join(saving_path, 'means.png'),
        dpi=dpi,
        bbox_inches='tight'
    )