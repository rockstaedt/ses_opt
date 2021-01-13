import json
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import os
from pathlib import Path

from modeling_helper.utilities import *
from modeling_helper.printing import *
from parameters import *
from model_options import *

opt = SolverFactory('gurobi')

current_path = Path.cwd()

# **************************************************************************
# Model Options
# **************************************************************************

# Model options are defined in the file 'model_options.py'.

# **************************************************************************
# Parameters
# **************************************************************************

# Seet for randomness
seed = 17

# Get test samples.
TEST_SAMPLES = get_monte_carlo_samples(LOADS, samples=test_size, seed=seed)

# Define a list of approaches.
APPROACHES = ['stochastic', 'deterministic']

# All other parameteres are defined in the file 'parameters.py'

# **************************************************************************
# Testing
# **************************************************************************

# Create dictionary for approach and for storage capacities to store objective
# value.
objective_values = {
    approach: [] for approach in APPROACHES
}

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

    print_caption(f'Solution for {stor_levels_max[0]} kWh')
    # Init result dictionary.
    results_dic = {}

    # Open result model json file
    try:
        with open(
            os.path.join(path, f'results_master_{stor_levels_max[0]}.json')
            ) as f:
            # returns JSON object as a dictionary
            parameter = json.load(f)
    except:
        print(
            'Attention! File '
            + os.path.join(path, f'results_master_{stor_levels_max[0]}.json')
            + ' not found.'
        )
        break
    print(f'Solving for all {test_size} test samples')

    for i, sample in enumerate(TEST_SAMPLES):

        print_status(i)

        # ******************************************************************
        # Model
        # ******************************************************************

        model = pyo.ConcreteModel()

        # ******************************************************************
        # Sets
        # ******************************************************************

        # hour set
        model.H = pyo.RangeSet(0,len(HOURS)-1)

        # ******************************************************************
        # Variables
        # ******************************************************************

        # Fixed model problem variables
        model.u = pyo.Var(model.H)
        model.p1 = pyo.Var(model.H)

        # Electricity produced by generator
        model.pg = pyo.Var(model.H, within=pyo.NonNegativeReals)

        # Electrictiy bought from retailer
        model.p2 = pyo.Var(model.H, within=pyo.NonNegativeReals)

        if esr:
            # Net injection by storage with bounds of maximum charge and
            # discharge.
            model.stor_net_i = pyo.Var(model.H)

            # Storage level
            model.stor_level = pyo.Var(model.H, within=pyo.NonNegativeReals)

        # ******************************************************************
        # Objective function
        # ******************************************************************

        model.OBJ = pyo.Objective(
            expr=sum(
                c1*model.u[h] + l1*model.p1[h]
                + c2*model.pg[h] + l2*model.p2[h] for h in model.H
            )
        )

        # ******************************************************************
        # Constraints
        # ******************************************************************

        # Init of forward contract
        def p1_zero(model):
            return model.p1[0] == 0
        model.p1_zero = pyo.Constraint(rule=p1_zero)

        if up_down_time:
            # Minimum uptime constraint
            def min_uptime(model, H):
                # In order to avoid applying this constraint for hour 0,
                # check for index.
                if H != 0:
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
                else:
                    # Initialize unit commitment in hour 0.
                    return model.u[H] == 0
            model.min_uptime = pyo.Constraint(model.H, rule=min_uptime)

            # Minimum downtime constraint
            def min_downtime(model, H):
                # In order to avoid applying this constraint for hour 0, check
                # for index.
                if H != 0:
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
                else:
                    # Initialize unit commitment in hour 0.
                    return model.u[H] == 0
            model.min_downtime = pyo.Constraint(model.H, rule=min_downtime)

        # Take first random vector from samples
        pl = sample
        if esr:
            # Load must be covered by production, purchasing electrictiy or
            # by net injection of storage system.
            def con_load(model, H):
                return (model.pg[H] + model.p1[H] + model.p2[H]
                    + model.stor_net_i[H]) >= pl[H]
            model.con_load = pyo.Constraint(model.H, rule=con_load)
        else:
            # Load must be covered by production or purchasing electrictiy.
            def con_load(model, H):
                return model.pg[H] + model.p1[H] + model.p2[H] >= pl[H]
            model.con_load = pyo.Constraint(model.H, rule=con_load)

        # maximum capacity of generator
        def con_max(model, H):
            return model.pg[H] <= pmax*model.u[H]
        model.con_max = pyo.Constraint(model.H, rule=con_max)

        if ramping:
            # Ramping constraint of generator
            def con_ramping(model, H):
                if H != 0:
                    return (
                        -ramping_constraint,
                        model.pg[H] - model.pg[H-1],
                        ramping_constraint
                    )
                else:
                    return model.pg[H] == 0
            model.con_ramping = pyo.Constraint(model.H, rule=con_ramping)

        if esr:
            # Constraint for net injection by storage
            def max_net_i(model, H):
                if H != 0:
                    return (-p_w_max, model.stor_net_i[H], p_i_max)
                else:
                    return model.stor_net_i[H] == 0
            model.max_net_i = pyo.Constraint(model.H, rule=max_net_i)

            # Maximum Storage Level
            def max_storage(model, H):
                return model.stor_level[H] <= stor_levels_max[0]
            model.max_storage = pyo.Constraint(model.H, rule=max_storage)

            # Storage Balance
            def stor_balance(model, H):
                if H != 0:
                    return model.stor_level[H] == (
                        model.stor_level[H-1] - model.stor_net_i[H]
                    )
                else:
                    return model.stor_level[H] == 0
            model.stor_balance = pyo.Constraint(model.H, rule=stor_balance)

        # ensure variable u is equal to the solution of the model problem
        def dual_con1(model, H):
            return model.u[H] == parameter['u'][str(H)]
        model.dual_con1 = pyo.Constraint(model.H, rule=dual_con1)

        # ensure variable p1 is equal to the solution of the model problem
        def dual_con2(model, H):
            return model.p1[H] == parameter['p1'][str(H)]
        model.dual_con2 = pyo.Constraint(model.H, rule=dual_con2)

        # solve model
        solve_model(opt, model)

        results_dic[i] = get_results(model)

        objective_values[approach].append(pyo.value(model.OBJ))

    print()
    print()
    print_caption('End')

        # with open(
        #     f'{path}testing_{approach}_results_{l2}.json', 'w'
        # ) as outfile:
        #     json.dump(results_dic, outfile)

# # save objectives
# path_save_test = os.path.join(current_path, '3_results', app)
# with open(f'{path}testing_objectives.json', 'w') as outfile:
#     json.dump(objective_values, outfile)

# print(np.mean(np.array(objective_values['Stochastic']['0.3'])))
# print(np.mean(np.array(objective_values['Deterministic']['0.3'])))
