import json
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import os
from pathlib import Path
import pandas as pd
import concurrent.futures

from modeling_helper.utilities import *
from modeling_helper.printing import *
from parameters import *
from model_options import *

opt = SolverFactory('gurobi')

current_path = Path.cwd()

# ******************************************************************************
# Model Options
# ******************************************************************************

# Model options are defined in the file 'model_options.py'.

# ******************************************************************************
# Parameters
# ******************************************************************************

# Seet for randomness
seed = 17

# Get test samples.
TEST_SAMPLES = get_monte_carlo_samples(LOADS, samples=test_size, seed=seed)

# Define a list of approaches.
APPROACHES = ['stochastic', 'deterministic']

# Define storage level for testing
stor_level_max = 4

# All other parameteres are defined in the file 'parameters.py'

# ******************************************************************************
# Testing
# ******************************************************************************

# Create dictionary for objective values.
objective_values = {approach: [] for approach in APPROACHES}

# Create dictionary for results
results_dic = {approach: {} for approach in APPROACHES}

for approach in APPROACHES:

    print_title(approach)

    # Get path to result files
    if approach == 'stochastic':
        deterministic = False
    else:
        deterministic = True
    path = get_path_by_task(
        up_down_time,
        ramping,
        esr,
        deterministic,
        sensitivity_analysis,
        sample_size,
        current_path)

    print_caption(f'Solution for {stor_level_max} kWh')

    # Open result model json file
    try:
        with open(
            os.path.join(path, f'results_master_{stor_level_max}.json')
            ) as f:
            # returns JSON object as a dictionary
            parameter = json.load(f)
    except:
        print(
            'Attention! File '
            + os.path.join(path, f'results_master_{stor_level_max}.json')
            + ' not found.'
        )
        break
    print(f'Solving for all {test_size} test samples')

    # **************************************************************************
    # Model
    # **************************************************************************

    model = pyo.ConcreteModel()

    # **************************************************************************
    # Sets
    # **************************************************************************

    # Hour sets
    model.H = pyo.RangeSet(1,len(HOURS)-1)
    model.H_all = pyo.RangeSet(0, len(HOURS)-1)

    # **************************************************************************
    # Parameter
    # **************************************************************************

    # Create a parameter for the load values.
    model.load_values = pyo.Param(model.H_all, mutable=True)

    # **************************************************************************
    # Variables
    # **************************************************************************

    # Fixed model problem variables
    model.u = pyo.Var(model.H_all)
    model.p1 = pyo.Var(model.H_all)
    for h in model.H_all:
        model.u[h].fix(parameter['u'][str(h)])
        model.p1[h].fix(parameter['p1'][str(h)])

    # Electricity produced by generator
    model.pg = pyo.Var(model.H_all, within=pyo.NonNegativeReals)
    # Initialization for p1
    model.pg[0].fix(0)

    # Electrictiy bought from retailer
    model.p2 = pyo.Var(model.H_all, within=pyo.NonNegativeReals)
    # Initialization for p1
    model.p2[0].fix(0)

    if esr:
        # Net injection by storage
        model.stor_net_i = pyo.Var(model.H, bounds=(-p_w_max, p_i_max))
        # Initialization for net injection
        model.stor_net_i[0].fix(0)

        # Storage level
        model.stor_level = pyo.Var(
            model.H_all,
            within=pyo.NonNegativeReals,
            bounds=(0, stor_level_max)
        )
        # Initialization for net injection
        model.stor_level[0].fix(0)

    # **************************************************************************
    # Objective function
    # **************************************************************************

    model.OBJ = pyo.Objective(
        expr=sum(
            c1*model.u[h] + l1*model.p1[h]
            + c2*model.pg[h] + l2*model.p2[h] for h in model.H
        )
    )

    # **************************************************************************
    # Constraints
    # **************************************************************************

    if up_down_time:
        # Minimum uptime constraint
        def min_uptime(model, H):
            # Apply minimum uptime constraint.
            # The end value of the range function needs to be increased
            # to be included.
            V = list(range(H, min([H-1 + uptime, len(HOURS)-1]) + 1))
            # For the last hour, the ouput of the range function is 0
            # because range(24,24). To include hour 24 into the list,
            # check for length and put hour 24 into V.
            if len(V) == 0:
                V = [H]
            # Return the sum of all hours in V to apply the constraint
            # for all hours in V.
            return sum(
                model.u[H] - model.u[H-1] for v in V
                ) <= sum(model.u[v] for v in V)
        model.min_uptime = pyo.Constraint(model.H, rule=min_uptime)

        # Minimum downtime constraint
        def min_downtime(model, H):
            # Apply minimum downtime constraint.
            # The end value of the range function needs to be increased
            # to be included.
            V = list(range(H, min([H-1 + downtime, len(HOURS)-1]) + 1))
            # For the last hour, the ouput of the range function is 0
            # because range(24,24). To include hour 24 into the list,
            # check for length and put hour 24 into V.
            if len(V) == 0:
                V = [H]
            # Return the sum of all hours in V to apply the constraint
            # for all hours in V.
            return sum(
                model.u[H-1] - model.u[H] for v in V
                ) <= sum(1 - model.u[v] for v in V)
        model.min_downtime = pyo.Constraint(model.H, rule=min_downtime)

    if esr:
        # Load must be covered by production, purchasing electrictiy or
        # by net injection of storage system.
        def con_load(model, H):
            return (model.pg[H] + model.p1[H] + model.p2[H]
                + model.stor_net_i[H]) >= model.load_values[H]
        model.con_load = pyo.Constraint(model.H, rule=con_load)
    else:
        # Load must be covered by production or purchasing electrictiy.
        def con_load(model, H):
            return (model.pg[H] + model.p1[H]
                    + model.p2[H]) >=  model.load_values[H]
        model.con_load = pyo.Constraint(model.H, rule=con_load)

    # maximum capacity of generator
    def con_max(model, H):
        return model.pg[H] <= pmax*model.u[H]
    model.con_max = pyo.Constraint(model.H, rule=con_max)

    if ramping:
        # Ramping constraint of generator
        def con_ramping(model, H):
            return (
                -ramping_constraint,
                model.pg[H] - model.pg[H-1],
                ramping_constraint
            )
        model.con_ramping = pyo.Constraint(model.H, rule=con_ramping)

    if esr:
        # Storage Balance
        def stor_balance(model, H):
            return model.stor_level[H] == (
                model.stor_level[H-1] - model.stor_net_i[H]
            )
        model.stor_balance = pyo.Constraint(model.H, rule=stor_balance)

    # Use concurrent package to enable multiprocessing to solve test samples in
    # paralell.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            solve_sample,
            TEST_SAMPLES,
            list(range(len(TEST_SAMPLES))),
            [len(TEST_SAMPLES)] * len(TEST_SAMPLES),
            [model] * len(TEST_SAMPLES),
            [opt] * len(TEST_SAMPLES),
            [True] * len(TEST_SAMPLES)
        )

    # Results are stored in a map object and have to be unpacked
    # into a dict.
    for i, result in enumerate(results):
        results_dic[approach][i] = result
        objective_values[approach].append(
            results_dic[approach][i]['objective_value']
        )

    # **************************************************************************
    # Export
    # **************************************************************************

    if output:
        saving_path = os.path.join(path, 'testing')
        # Make sure that folders exist
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        print(f'Exporting results to: {saving_path}')

        #-----------------------------------------------------------------------
        # Model results
        #-----------------------------------------------------------------------

        with open(
            os.path.join(
                saving_path,
                f'testing_results_{stor_level_max}_{test_size}.json'
            ), 'w') as outfile:
            json.dump(results_dic[approach], outfile)

        #-----------------------------------------------------------------------
        # Objective values
        #-----------------------------------------------------------------------

        # Save objectives
        np.array(objective_values[approach]).tofile(
            os.path.join(
                saving_path,
                f'objective_values_{stor_level_max}_{test_size}.csv'),
            sep=','
        )

        mean = np.mean(objective_values[approach])
        variance = np.var(objective_values[approach])
        pd.DataFrame({'Mean': [mean], 'Variance': [variance]}).to_csv(
            os.path.join(
                saving_path,
                f'mean_var_{stor_level_max}_{test_size}.csv'),
            sep=','
        )

        print()
        print_caption('End')
