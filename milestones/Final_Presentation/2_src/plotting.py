###############################################################################
### Plotting
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
import csv
import json
from pathlib import Path
import os
import math

from model_options import output
from modeling_helper.utilities import get_loads

###############################################################################
### Matplotlib Settings
###############################################################################

settings = {
    'text.usetex': True,
    'font.weight': 'normal',
    'font.size': 14
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
### Computation time
###############################################################################

sample_sizes = [100, 1000, 10000]

computation_times = {'task_'+str(i): [] for i in [1,2]}
computation_times['task_4_av_sampling'] = []
computation_times['task_4_lhc_sampling'] = []

for task in [1,2,4]:
    for sample_size in sample_sizes:
        if task != 4:
            df = pd.read_csv(
                os.path.join(
                    current_path.parent.parent,
                    'Milestone_4',
                    '3_results',
                    'stochastic',
                    'task_'+str(task),
                    str(sample_size),
                    'computation_times.csv'
                )
            )
            computation_times['task_'+str(task)].append(df.iloc[1,1])
        else:
            for sampling in ['av_sampling', 'lhc_sampling']:
                df = pd.read_csv(
                    os.path.join(
                        current_path.parent.parent,
                        'Milestone_4',
                        '3_results',
                        'stochastic',
                        'task_'+str(task),
                        sampling,
                        str(sample_size),
                        'computation_times.csv'
                    )
                )
                computation_times[f'task_{task}_{sampling}'].append(
                    df.iloc[1,1]
                )

fig_comp_time, ax1 = plt.subplots(1, figsize=(9,6))

ax1.plot(
    sample_sizes,
    computation_times['task_4_av_sampling'],
    '--',
    label='Antithetic Variates',
    marker='x',
    color='darkorange')
ax1.plot(
    sample_sizes,
    computation_times['task_4_lhc_sampling'],
    '--',
    label='Latin Hypercube',
    marker='x',
    color = 'forestgreen')
ax1.plot(
    sample_sizes,
    computation_times['task_1'],
    label='Antithetic Variates with multiprocessing',
    marker='x',
    color='darkorange')
ax1.plot(
    sample_sizes,
    computation_times['task_2'],
    label='Latin Hypercube with multiprocessing',
    marker='x',
    color = 'forestgreen')

ax1.legend()
ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Computation time in [s]')
ax1.set_xticks(sample_sizes)
ax1.get_xaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

if output:
    fig_comp_time.savefig(
        os.path.join(saving_path, 'computation_times.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Mean Variance - MS 4
###############################################################################

sample_sizes = ['10', '100', '1,000', '10,000', '100,000']

sampling_methods = ['crude', 'av', 'lhc']

sampling_method_to_results_path = {
    'crude': os.path.join(
        current_path.parent.parent,
        'Milestone_3',
        '3_results',
        'stochastic',
        'task_3'
    ),
    'av': os.path.join(
        current_path.parent.parent,
        'Milestone_4',
        '3_results',
        'stochastic',
        'task_1'
    ),
    'lhc': os.path.join(
        current_path.parent.parent,
        'Milestone_4',
        '3_results',
        'stochastic',
        'task_2'
    )
}

sampling_method_to_file = {
    'crude': 'mean_var_4_1000.csv',
    'av': 'mean_var_0.8_1000.csv',
    'lhc': 'mean_var_0.8_1000.csv'
}

stds = {
    sampling_method: [] for sampling_method in sampling_methods
}
means = {
    sampling_method: [] for sampling_method in sampling_methods
}

for sampling_method in sampling_methods:
    for sample_size in sample_sizes:
        df = pd.read_csv(
            os.path.join(
                sampling_method_to_results_path[sampling_method],
                sample_size.replace(',', ''),
                'testing',
                sampling_method_to_file[sampling_method]
            )
        )
        means[sampling_method].append(df.iloc[0,0])
        stds[sampling_method].append(math.sqrt(df.iloc[0,1]))

fig_mean_ms4, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,6))

x = np.arange(len(sample_sizes))

width = 0.3

ax1.bar(
    x - width,
    means['crude'],
    width,
    color='blue'
)
ax1.bar(
    x,
    means['av'],
    width,
    color='darkorange'
)
ax1.bar(
    x + width,
    means['lhc'],
    width,
    color='forestgreen'
)

ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Mean total costs in [\$]')
ax1.set_xticks(x)
ax1.set_xticklabels(sample_sizes)
ax1.set_ylim((93, 95.5))

ax2.bar(
    x - width,
    stds['crude'],
    width,
    label='Crude Monte Carlo',
    color='blue'
)
ax2.bar(
    x,
    stds['av'],
    width,
    label='Antithetic Variates',
    color='darkorange'
)
ax2.bar(
    x + width,
    stds['lhc'],
    width,
    label='Latin Hypercube',
    color='forestgreen'
)

ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Standard deviation in [\$]')
ax2.set_xticks(x)
ax2.set_xticklabels(sample_sizes)
ax2.set_ylim((2, 4))


ax2.legend(bbox_to_anchor=(1,1), loc="upper left")

if output:
    fig_mean_ms4.savefig(
        os.path.join(saving_path, 'mean_std_ms4.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Mean Variance - Final Presentation
###############################################################################

sample_sizes = ['10', '100', '1,000', '10,000', '100,000']

sampling_methods = ['crude', 'av', 'lhc']

sampling_method_to_results_path = {
    'crude': os.path.join(
        results_path,
        'stochastic',
        'multiprocessing',
        'task_1_c'
    ),
    'av': os.path.join(
        results_path,
        'stochastic',
        'multiprocessing',
        'task_1_b'
    ),
    'lhc': os.path.join(
        results_path,
        'stochastic',
        'multiprocessing',
        'no_task'
    )
}

stds = {
    sampling_method: [] for sampling_method in sampling_methods
}
means = {
    sampling_method: [] for sampling_method in sampling_methods
}

for sampling_method in sampling_methods:
    for sample_size in sample_sizes:
        df = pd.read_csv(
            os.path.join(
                sampling_method_to_results_path[sampling_method],
                sample_size.replace(',', ''),
                'testing',
                'mean_var_0.6_1000.csv'
            )
        )
        means[sampling_method].append(df.iloc[0,0])
        stds[sampling_method].append(math.sqrt(df.iloc[0,1]))

fig_mean_final, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,6))

x = np.arange(len(sample_sizes))

width = 0.3

ax1.bar(
    x - width,
    means['crude'],
    width,
    color='blue'
)
ax1.bar(
    x,
    means['av'],
    width,
    color='darkorange'
)
ax1.bar(
    x + width,
    means['lhc'],
    width,
    color='forestgreen'
)

ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Mean total costs in [\$]')
ax1.set_xticks(x)
ax1.set_xticklabels(sample_sizes)
ax1.set_ylim((102, 104))

ax2.bar(
    x - width,
    stds['crude'],
    width,
    label='Crude Monte Carlo',
    color='blue'
)
ax2.bar(
    x,
    stds['av'],
    width,
    label='Antithetic Variates',
    color='darkorange'
)
ax2.bar(
    x + width,
    stds['lhc'],
    width,
    label='Latin Hypercube',
    color='forestgreen'
)

ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Standard deviation in [\$]')
ax2.set_xticks(x)
ax2.set_xticklabels(sample_sizes)
ax2.set_ylim((1, 3.5))


ax2.legend(bbox_to_anchor=(1,1), loc="upper left")

if output:
    fig_mean_final.savefig(
        os.path.join(saving_path, 'mean_std_final.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Mean Variance - MS 3
###############################################################################

sample_sizes = ['10', '100', '1,000', '10,000', '100,000']

tasks = ['no_battery', 'battery']

task_to_results_path = {
    'no_battery': os.path.join(
        current_path.parent.parent,
        'Milestone_3',
        '3_results',
        'stochastic',
        'task_2'
    ),
    'battery': os.path.join(
        current_path.parent.parent,
        'Milestone_3',
        '3_results',
        'stochastic',
        'task_3'
    )
}

stds = {
    task: [] for task in tasks
}
means = {
    task: [] for task in tasks
}

for task in tasks:
    for sample_size in sample_sizes:
        df = pd.read_csv(
            os.path.join(
                task_to_results_path[task],
                sample_size.replace(',', ''),
                'testing',
                'mean_var_4_1000.csv'
            )
        )
        means[task].append(df.iloc[0,0])
        stds[task].append(math.sqrt(df.iloc[0,1]))

fig_mean_ms3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,6))

x = np.arange(len(sample_sizes))

width = 0.3

ax1.bar(
    x - width/2,
    means['no_battery'],
    width,
    color='slategray'
)
ax1.bar(
    x + width/2,
    means['battery'],
    width,
    color='yellowgreen'
)

ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Mean total costs in [\$]')
ax1.set_xticks(x)
ax1.set_xticklabels(sample_sizes)
ax1.set_ylim((93.5, 97))

ax2.bar(
    x - width/2,
    stds['no_battery'],
    width,
    label='Without ESR',
    color='slategray'
)
ax2.bar(
    x + width/2,
    stds['battery'],
    width,
    label='With ESR',
    color='yellowgreen'
)

ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Standard deviation in [\$]')
ax2.set_xticks(x)
ax2.set_xticklabels(sample_sizes)
ax2.set_ylim((2, 4))


ax2.legend(bbox_to_anchor=(1,1), loc="upper left")

if output:
    fig_mean_ms3.savefig(
        os.path.join(saving_path, 'mean_std_ms3.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Comparison stochastic vs. deterministic
###############################################################################

labels = ['Deterministic', 'Stochastic']

label_to_file = {
    'Deterministic': os.path.join(
        results_path,
        'deterministic',
        'task_1',
        'testing',
        'mean_var_0.6_1000.csv'
    ),
    'Stochastic': os.path.join(
        results_path,
        'stochastic',
        'multiprocessing',
        'task_1_b',
        '10000',
        'testing',
        'mean_var_0.6_1000.csv'
    )
}

means = []

for label in labels:
    df = pd.read_csv(label_to_file[label])
    means.append(df.iloc[0,0])


fig_comp, ax1 = plt.subplots(1, figsize=(6,6))

ax1.bar(
    'Deterministic',
    means[0],
    color='slategray'
)

ax1.bar(
    'Stochastic',
    means[1],
    color='yellowgreen'
)

ax1.set_ylabel('Mean total costs in [\$]')
ax1.set_ylim((101, 106))

if output:
    fig_comp.savefig(
        os.path.join(saving_path, 'comparison_stoch_det_means.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Generation mix
###############################################################################

approaches = ['stochastic', 'deterministic']

approach_to_file = {
    'deterministic': os.path.join(
        results_path,
        'deterministic',
        'task_1',
        'testing',
        'testing_results_0.6_1000.json'
    ),
    'stochastic': os.path.join(
        results_path,
        'stochastic',
        'multiprocessing',
        'task_1_b',
        '10000',
        'testing',
        'testing_results_0.6_1000.json'
    )
}

variables = ['p1', 'pg', 'p2']

approach_to_variable_to_values = {
    approach: {
        variable: [] for variable in variables
    } for approach in approaches
}

for approach in approaches:
    with open(approach_to_file[approach]) as f:
        # returns JSON object as a dictionary
        results = json.load(f)

    for sample in range(len(results)):
        for variable in variables:
            hour_to_value = results[str(sample)][variable]
            value_sum = 0
            for hour, value in hour_to_value.items():
                value_sum += value
            approach_to_variable_to_values[approach][variable].append(
                value_sum
            )

approach_to_means = {
    approach: [] for approach in approaches
}

for approach in approaches:
    for variable in variables:
        approach_to_means[approach].append(
            np.mean(
                approach_to_variable_to_values[approach][variable]
            )
        )

fig_comp_power, ax1 = plt.subplots(1, figsize=(6,6))

labels = ['$p_{FW}$', '$p_G$', '$p_{RT}$']

x = np.arange(len(labels))

width = 0.3

ax2 = ax1.twinx()

ax1.bar(
    x[:2] - width/2,
    approach_to_means['deterministic'][:2],
    width,
    color='slategray',
    label='Deterministic'
)
ax1.bar(
    x[:2] + width/2,
    approach_to_means['stochastic'][:2],
    width,
    color='yellowgreen',
    label='Stochastic'
)
ax2.bar(
    x[2] - width/2,
    approach_to_means['deterministic'][2],
    width,
    color='slategray',
    label='Deterministic'
)
ax2.bar(
    x[2] + width/2,
    approach_to_means['stochastic'][2],
    width,
    color='yellowgreen',
    label='Stochastic'
)
ax2.axvline(x = 1.5, color='k', linestyle='--', linewidth='1')

ax1.set_xlabel('Variable')
ax1.set_ylabel('Mean power in [kW]')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim((230, 280))

ax2.set_ylim((9, 13))
ax2.legend(bbox_to_anchor=(1,1), loc="upper left")

if output:
    fig_comp_power.savefig(
        os.path.join(saving_path, 'comparison_stoch_det_power.png'),
        dpi=dpi,
        bbox_inches='tight'
    )

###############################################################################
### Load vector
###############################################################################

loads = np.array(get_loads())
xs = np.arange(len(loads))
std = np.sqrt(loads*(1/3))

fig_load = plt.figure(figsize=(11,7))

plt.plot(xs[1:], loads[1:], linewidth=3, label='Mean', color='darkgreen')
plt.fill_between(
    xs[1:],
    loads[1:]+std[1:],
    loads[1:]-std[1:],
    alpha=0.3,
    color='lime',
    label = 'Standard deviation'
)
plt.legend(loc=2)
plt.xlabel('Hours')
plt.ylabel('Load in [kW]')

if output:
    fig_load.savefig(
        os.path.join(saving_path, 'load_values.png'),
        dpi=dpi,
        bbox_inches='tight'
    )