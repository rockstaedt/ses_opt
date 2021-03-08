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

multiprocessing_testing = False

# ******************************************************************************
# Parameters
# ******************************************************************************

# Seet for randomness
seed = 17

# Get test samples.
TEST_SAMPLES = get_monte_carlo_samples(LOADS, sample_size=test_size, seed=seed)

# Define a list of approaches.
APPROACHES = ['stochastic', 'deterministic']

# Define charge target for ev
charge_target = 0.6

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
        mc_sampling=mc_sampling,
        av_sampling=av_sampling,
        deterministic=deterministic,
        sample_size=sample_size,
        multiprocessing=multiprocessing,
        current_path=current_path
    )

    print_caption(f'Solution for {charge_target*100} %')

    # Open result model json file
    try:
        with open(
            os.path.join(path, f'results_master_{charge_target}.json')
            ) as f:
            # returns JSON object as a dictionary
            parameter = json.load(f)
    except:
        print(
            'Attention! File '
            + os.path.join(path, f'results_master_{charge_target}.json')
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

    # Set of energy storage resources.
    model.ESR = pyo.Set(initialize=esr_types)

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

    # Net injection by storage with bounds of maximum charge and discharge.
    # Create function to get the bounds for corresponding ESR.
    def get_net_bounds(model, ESR, H):
        return (-esr_to_p_w_max[ESR], esr_to_p_i_max[ESR])
    model.stor_net_i = pyo.Var(
        model.ESR,
        model.H_all,
        bounds=get_net_bounds
    )

    # Storage level in kWh
    # Create function to get max storage level for corresponding ESR.
    def get_stor_levels(model, ESR, H):
        if 'ev' not in ESR:
            return (0, esr_to_stor_level_max[ESR])
        else:
            if H < plug_in_hour or H > plug_out_hour:
                return (0,0)
            else:
                return (
                    min_soc*esr_to_stor_level_max[ESR],
                    max_soc*esr_to_stor_level_max[ESR]
                )
    model.stor_level = pyo.Var(
        model.ESR,
        model.H_all,
        within=pyo.NonNegativeReals,
        bounds=get_stor_levels
    )
    # Initialization of storage level and net injections for all ESR.
    for esr_type in esr_types:
        if 'ev' in esr_type:
            for h in model.H_all:
                if h < plug_in_hour or h > plug_out_hour:
                    # Fix stor level and net injection values to zero before ev
                    # is plugged in and after ev is plugged out.
                    model.stor_level[esr_type, h].fix(0)
                    model.stor_net_i[esr_type, h].fix(0)
            model.stor_level[esr_type, plug_out_hour].fix(
                charge_target * esr_to_stor_level_max[esr_type]
            )
        else:
            model.stor_level[esr_type, 0].fix(esr_to_stor_level_zero[esr_type])

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

    # Load must be covered by production, purchasing electrictiy or by the
    # sum of net injections over all esr.
    def con_load(model, H):
        return (model.pg[H] + model.p1[H] + model.p2[H]
            + sum(model.stor_net_i[esr, H] for esr in model.ESR)
            ) >= model.load_values[H]
    model.con_load = pyo.Constraint(model.H, rule=con_load)

    # Maximum capacity of generator
    def con_max(model, H):
        return model.pg[H] <= pmax*model.u[H]
    model.con_max = pyo.Constraint(model.H, rule=con_max)

    # Ramping constraint of generator
    def con_ramping(model, H):
        return (
            -ramping_constraint, model.pg[H] - model.pg[H-1], ramping_constraint
        )
    model.con_ramping = pyo.Constraint(model.H, rule=con_ramping)

    # Storage Balance
    def stor_balance(model, ESR, H):
        if 'ev' not in ESR:
            return model.stor_level[ESR, H] == (
                model.stor_level[ESR, H-1] - model.stor_net_i[ESR, H])
        else:
            if H > plug_in_hour and H <= plug_out_hour:
                return model.stor_level[ESR, H] == (
                    model.stor_level[ESR, H-1] - model.stor_net_i[ESR, H]
                )
            if H == plug_in_hour:
                return model.stor_level[ESR, H] == (
                    esr_to_stor_level_zero[ESR] - model.stor_net_i[ESR, H]
                )
            else:
                return pyo.Constraint.Skip
    model.stor_balance = pyo.Constraint(model.ESR, model.H, rule=stor_balance)

    # **************************************************************************
    # Initialization
    # **************************************************************************

    if multiprocessing_testing:
        # Use concurrent package to enable multiprocessing to solve test samples
        # in parallel.
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
    else:
        # Solve model per iteration.
        results = map(
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
                f'testing_results_{charge_target}_{test_size}.json'
            ), 'w') as outfile:
            # Check for variables containing a tuple key.
            for sample in results_dic[approach]:
                for var in results_dic[approach][sample]:
                    # Key 'objective_value' does not contain any keys.
                    if var != 'objective_value':
                        first_key = list(
                            results_dic[approach][sample][var].keys()
                        )[0]
                        if type(first_key) == tuple:
                            # Reset all keys for that variable.
                            results_dic[approach][sample][var] = reset_tuple_key(
                                results_dic[approach][sample][var]
                            )
            json.dump(results_dic[approach], outfile)

        #-----------------------------------------------------------------------
        # Objective values
        #-----------------------------------------------------------------------

        # Save objectives
        np.array(objective_values[approach]).tofile(
            os.path.join(
                saving_path,
                f'objective_values_{charge_target}_{test_size}.csv'),
            sep=','
        )

        mean = np.mean(objective_values[approach])
        variance = np.var(objective_values[approach])
        pd.DataFrame({'Mean': [mean], 'Variance': [variance]}).to_csv(
            os.path.join(
                saving_path,
                f'mean_var_{charge_target}_{test_size}.csv'),
            sep=',',
            index=False
        )

    print()
    print_caption('End')
