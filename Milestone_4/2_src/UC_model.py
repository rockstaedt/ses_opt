import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time as tm
import numpy as np
import json
import pandas as pd
import os
from pathlib import Path
import concurrent.futures

from modeling_helper.utilities import *
from modeling_helper.printing import *
from parameters import *
from model_options import *

time_start_all = tm.time()

###############################################################################
### Model Options
###############################################################################

# The model options are defined in the file 'model_options.py'

###############################################################################
### Parameters
###############################################################################

# Seet for randomness
seed = 12

# For the deterministic approach, the normal load vector with its mean values
# is used. For the stochastic approach, a monte carlo sample is created.
if deterministic:
    SAMPLES = np.array([LOADS])
    sample_size = 1
else:
    SAMPLES = get_monte_carlo_samples(
        LOADS,
        samples=sample_size,
        seed=seed
    )

# All other parameteres are defined in the file 'parameters.py'

###############################################################################
### L-shape method
###############################################################################

# Solver for MIP
opt = pyo.SolverFactory('gurobi')

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

def objective(u, p1, pg, p2):
    """
    This function calculates the objective value for all hours in hours. This is
    also the upper bound of the decomposition.
    """
    return sum(
        c1*u[h] + l1*p1[h] + c2*pg[h] + l2*p2[h] for h in HOURS
    )

def master_prob(u, p1, alpha):
    """
    This function calculates the lower bound of the decomposition.
    """
    return sum(c1*u[h] + l1*p1[h] + alpha[h] for h in HOURS)

# Dataframe for computation times
times_dic = {'stor_level': [], 'time': [], 'iterations': []}

# Loop over all storage capacities and solve the L-shape method.
for stor_level_max in stor_levels_max:

    if deterministic:
        print_title(
            f'Solve L-Shape method for {stor_level_max} kWh - Deterministic'
        )
    else:
        print_title(
            f'Solve L-Shape method for {stor_level_max} kWh - Stochastic'
        )

    #---------------------------------------------------------------------------
    # Helper variables
    #---------------------------------------------------------------------------

    # List of the objective values = upper bound
    objective_values = []

    # List of lower bound values
    lower_bounds = []

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # Master problem
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    master = pyo.ConcreteModel()

    # **************************************************************************
    # Sets
    # **************************************************************************

    # Hour sets
    master.H = pyo.RangeSet(1, len(HOURS)-1)
    master.H_all = pyo.RangeSet(0, len(HOURS)-1)

    # **************************************************************************
    # Variables
    # **************************************************************************

    # Unit commitment for generator
    master.u = pyo.Var(master.H_all, within=pyo.Binary)

    # Initialization for u
    master.u[0].fix(0)

    # Electricity purchased with the forward contract
    master.p1 = pyo.Var(master.H_all, within=pyo.NonNegativeReals)

    # Initialization for p1
    master.p1[0].fix(0)

    # Value function for second stage problem
    master.alpha = pyo.Var(master.H_all)

    # Initialization for p1
    master.alpha[0].fix(0)

    # **************************************************************************
    # Objective function
    # **************************************************************************

    def master_obj(master):
        return sum(
            c1*master.u[h] + l1*master.p1[h] + master.alpha[h] for h in master.H
        )
    master.OBJ = pyo.Objective(rule=master_obj)

    # **************************************************************************
    # Constraints
    # **************************************************************************

    # Alpha down (-500) is an arbitrary selected bound
    def alphacon1(master, H):
        return master.alpha[H] >= -500
    master.alphacon1 = pyo.Constraint(master.H, rule=alphacon1)

    if up_down_time:
        # Minimum uptime constraint
        def min_uptime(master, H):
            # Apply minimum uptime constraint.
            # The end value of the range function needs to be increased to
            # be included.
            V = list(range(H, min([H-1 + uptime, len(HOURS)-1]) + 1))
            # For the last hour, the ouput of the range function is 0
            # because range(24,24). To include hour 24 into the list,
            # check for length and put hour 24 into V.
            if len(V) == 0:
                V = [H]
            # Return the sum of all hours in V to apply the constraint for
            # all hours in V.
            return sum(
                master.u[H] - master.u[H-1] for v in V
                ) <= sum(master.u[v] for v in V)
        master.min_uptime = pyo.Constraint(master.H, rule=min_uptime)

        # Minimum downtime constraint
        def min_downtime(master, H):
            # Apply minimum downtime constraint.
            # The end value of the range function needs to be increased to
            # be included.
            V = list(range(H, min([H-1 + downtime, len(HOURS)-1]) + 1))
            # For the last hour, the ouput of the range function is 0
            # because range(24,24). To include hour 24 into the list,
            # check for length and put hour 24 into V.
            if len(V) == 0:
                V = [H]
            # Return the sum of all hours in V to apply the constraint for
            # all hours in V.
            return sum(
                master.u[H-1] - master.u[H] for v in V
                ) <= sum(1 - master.u[v] for v in V)
        master.min_downtime = pyo.Constraint(master.H, rule=min_downtime)

    #---------------------------------------------------------------------------
    # Initialization of master problem
    #---------------------------------------------------------------------------

    # Save current time to get the time of calculating
    time_start = tm.time()

    # Init iteration counter
    iteration = 0

    print_caption('Initialization')

    print('Solving master problem...')

    solve_model(opt, master)

    results_master = get_results(master)

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # Sub problem
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    sub = pyo.ConcreteModel()

    # **************************************************************************
    # Sets
    # **************************************************************************

    # Hour sets
    sub.H = pyo.RangeSet(1, len(HOURS)-1)
    sub.H_all = pyo.RangeSet(0, len(HOURS)-1)

    # **************************************************************************
    # Parameter
    # **************************************************************************

    # Create a parameter for the load values.
    sub.load_values = pyo.Param(sub.H_all, mutable=True)

    # **************************************************************************
    # Variables
    # **************************************************************************

    # First stage variables

    # no need for declaration of variable types because that is determined by
    # corresponding variables of master problem
    sub.u = pyo.Var(sub.H_all)
    sub.p1 = pyo.Var(sub.H_all)

    # Second stage variables

    # Electricity produced by generator
    sub.pg = pyo.Var(sub.H_all, within=pyo.NonNegativeReals)

    # Initialization of pg
    sub.pg[0].fix(0)

    # Electrictiy bought from retailer
    sub.p2 = pyo.Var(sub.H_all, within=pyo.NonNegativeReals)

    # Initialization of pg
    sub.p2[0].fix(0)

    if esr:
        # Net injection by storage with bounds of maximum charge and discharge.
        sub.stor_net_i = pyo.Var(sub.H_all,  bounds=(-p_w_max, p_i_max))

        # Initialization of storage net injection
        sub.stor_net_i[0].fix(0)

        # Storage level
        sub.stor_level = pyo.Var(
            sub.H_all,
            within=pyo.NonNegativeReals,
            bounds=(0, stor_level_max)
        )

        # Initialization of storage level
        sub.stor_level[0].fix(0)


    # **************************************************************************
    # Objective function
    # **************************************************************************

    sub.OBJ = pyo.Objective(
        expr=sum(c2*sub.pg[h] +l2*sub.p2[h] for h in sub.H)
    )

    # **************************************************************************
    # Constraints
    # **************************************************************************

    if esr:
        # Load must be covered by production, purchasing electrictiy or by net
        # injection of storage system.
        def con_load(sub, H):
            return (sub.pg[H] + sub.p1[H] + sub.p2[H]
                + sub.stor_net_i[H]) >= sub.load_values[H]
        sub.con_load = pyo.Constraint(sub.H, rule=con_load)
    else:
        # Load must be covered by production or purchasing electrictiy.
        def con_load(sub, H):
            return sub.pg[H] + sub.p1[H] + sub.p2[H] >= sub.load_values[H]
        sub.con_load = pyo.Constraint(sub.H, rule=con_load)

    # maximum capacity of generator
    def con_max(sub, H):
        return sub.pg[H] <= pmax*sub.u[H]
    sub.con_max = pyo.Constraint(sub.H, rule=con_max)

    # ensure variable u is equal to the solution of the master problem
    def dual_con1(sub, H):
        return sub.u[H] == results_master['u'][H]
    sub.dual_con1 = pyo.Constraint(sub.H_all, rule=dual_con1)

    # ensure variable p1 is equal to the solution of the master problem
    def dual_con2(sub, H):
        return sub.p1[H] == results_master['p1'][H]
    sub.dual_con2 = pyo.Constraint(sub.H_all, rule=dual_con2)

    if ramping:
        # Ramping constraint of generator
        def con_ramping(sub, H):
            return (
                -ramping_constraint, sub.pg[H] - sub.pg[H-1], ramping_constraint
            )
        sub.con_ramping = pyo.Constraint(sub.H, rule=con_ramping)

    if esr:
        # Storage Balance
        def stor_balance(sub, H):
            return sub.stor_level[H] == sub.stor_level[H-1] - sub.stor_net_i[H]
        sub.stor_balance = pyo.Constraint(sub.H, rule=stor_balance)

    #---------------------------------------------------------------------------
    # Initialization of sub problem
    #---------------------------------------------------------------------------

    # enable calculation of dual variables in pyomo
    sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    print(f'Solving sub problem for samples size = {sample_size}')

    # Use concurrent package to enable multiprocessing to solve sample in
    # paralell.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            solve_sample,
            SAMPLES,
            list(range(len(SAMPLES))),
            [len(SAMPLES)] * len(SAMPLES),
            [sub] * len(SAMPLES),
            [opt] * len(SAMPLES)
        )

    # Results are stored in a map object and have to be unpacked into a dict.
    results_sub = {}
    for i, result in enumerate(results):
        results_sub[i] = result

    # check if upper and lower bound are converging
    converged, upper_bound, lower_bound = convergence_check(
        objective,
        master_prob,
        results_master,
        results_sub,
        samples=SAMPLES,
        epsilon=epsilon
    )

    print_convergence(converged)

    objective_values.append(upper_bound)

    lower_bounds.append(lower_bound)

    # Optimize until upper and lower bound are converging
    while not converged:
        iteration += 1

        print_caption(f'Iteration {iteration}')

        def cut(master, H):
            return (
                sum(
                    c2*results_sub[i]['pg'][H] + l2*results_sub[i]['p2'][H]
                    + results_sub[i]['dual_con1'][H]*(
                        master.u[H] - results_master['u'][H])
                    + results_sub[i]['dual_con2'][H]*(
                        master.p1[H] - results_master['p1'][H])
                    for i, sample in enumerate(SAMPLES)
                )/sample_size
                <= master.alpha[H]
            )

        setattr(master, f'cut_{iteration}', pyo.Constraint(master.H, rule=cut))
        print(f'Added cut_{iteration}')

        print('Solving master problem...')
        solve_model(opt, master)
        results_master = get_results(master)

        # update dual constraint in sub problem
        sub.dual_con1.reconstruct()
        sub.dual_con2.reconstruct()

        print(f'Solving sub problem for samples size = {sample_size}')

        # Use concurrent package to enable multiprocessing to solve sample in
        # paralell.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(
                solve_sample,
                SAMPLES,
                list(range(len(SAMPLES))),
                [len(SAMPLES)] * len(SAMPLES),
                [sub] * len(SAMPLES),
                [opt] * len(SAMPLES)
            )

        # Results are stored in a map object and have to be unpacked
        # into a dict.
        results_sub = {}
        for i, result in enumerate(results):
            results_sub[i] = result

        converged, upper_bound, lower_bound = convergence_check(
            objective,
            master_prob,
            results_master,
            results_sub,
            samples=SAMPLES,
            epsilon=epsilon
        )

        print_convergence(converged)

        objective_values.append(upper_bound)

        lower_bounds.append(lower_bound)

    print_caption('End')

    times_dic['iterations'].append(iteration)

    ############################################################################
    ### Results & Exports
    ############################################################################

    time_end = tm.time()
    times_dic['stor_level'].append(str(stor_level_max))
    times_dic['time'].append(time_end - time_start)
    print('Computation time:')
    print(f'\t{round(time_end - time_start, 2)}s')

    current_path = Path.cwd()

    path = get_path_by_task(
        up_down_time,
        ramping,
        esr,
        deterministic,
        sensitivity_analysis,
        sample_size,
        current_path)

    # Make sure that folders exist
    if not os.path.exists(path):
        os.makedirs(path)

    #---------------------------------------------------------------------------
    # Model results
    #---------------------------------------------------------------------------

    if output:
        # Save only the sub results of the last iteration and last sample.
        with open(
            os.path.join(path, f'results_sub_{stor_level_max}.json'),
            'w') as outfile:
            last_key = list(results_sub.keys())[-1]
            json.dump(results_sub[last_key], outfile)

        with open(
            os.path.join(path, f'results_master_{stor_level_max}.json'),
            'w') as outfile:
            json.dump(results_master, outfile)

    #---------------------------------------------------------------------------
    # Upper and lower bound
    #---------------------------------------------------------------------------

    if output:
        # Export upper and lower bounds
        np.array(objective_values).tofile(
            os.path.join(path, f'objective_values_{stor_level_max}.csv'),
            sep = ','
        )
        np.array(lower_bounds).tofile(
            os.path.join(path, f'lower_bounds_{stor_level_max}.csv'),
            sep = ','
        )

#---------------------------------------------------------------------------
    # Computation time
#---------------------------------------------------------------------------

time_end_all = tm.time()
times_dic['stor_level'].append('TOTAL')
times_dic['time'].append(time_end_all - time_start_all)
times_dic['iterations'].append(0)
if sensitivity_analysis:
    print('Computation time for sensitivity analysis:')
    print(f'\t{round(time_end_all - time_start_all, 2)}s')

if output:
    # save computation times as csv
    df_time = pd.DataFrame(times_dic)
    df_time.to_csv(
        os.path.join(path, f'computation_times.csv'),
        index=False)
