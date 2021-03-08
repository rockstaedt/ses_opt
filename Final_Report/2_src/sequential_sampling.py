### first try on sequential sampling###
'''
while running:
- draw n samples
- solve UC_model for these n samples -> x*
- testing.py for candidate solution x_hut
- testing.py for x*
- calculate optimality gap
- if gap < threshold => stop
- else increase n, move on to next candidate solution


gaps = []
sample_size = 2
for x_cand in [10,100,1000,10000]:
    
    samples = get_samples(sample_size)
    x* = UC_model(samples)

    for i in [x*,x_cand]:
        testing(samples,i)

    gap = obj(x_cand) - obj(x*)

    gaps.append(gap)

    sample_size += 1
'''

import json
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import os
from pathlib import Path
import pandas as pd
import concurrent.futures
import time as tm

from modeling_helper.utilities import *
from modeling_helper.printing import *
from parameters import *
from model_options import *

sample_size = 2
gaps = []

for x_cand in [10,100,1000,10000]:
    ###############################################################################
    ### Sampling Set n
    ###############################################################################


    ###Sample Size###

    
    test_size = sample_size

    # Seed for randomness
    seed = 12

    if deterministic:
        SAMPLES = np.array([LOADS])
        sample_size = 1
    else:
        if mc_sampling:
            SAMPLES = get_monte_carlo_samples(
                LOADS,
                sample_size=sample_size,
                seed=seed
            )
        elif av_sampling:
            SAMPLES = get_av_samples(
                LOADS,
                sample_size=sample_size,
                seed=seed
            )
        elif lhc_sampling:
            SAMPLES = get_lhc_samples(
                LOADS,
                sample_size=sample_size,
                seed=seed
            )

    ###############################################################################
    ### Solving UC_Model for n to get x*
    ###############################################################################

    ###############################################################################
    ### L-shape method
    ###############################################################################
    time_start_all = tm.time()
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
    times_dic = {'time': [], 'iterations': []}

    if deterministic:
        print_title(
            f'Solve L-Shape method for {charge_target*100} % - Deterministic'
        )
    else:
        print_title(
            f'Solve L-Shape method for {charge_target*100} % - Stochastic'
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

    # Set of energy storage resources.
    sub.ESR = pyo.Set(initialize=esr_types)

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

    # Net injection by storage with bounds of maximum charge and discharge.
    # Create function to get the bounds for corresponding ESR.
    def get_net_bounds(sub, ESR, H):
        return (-esr_to_p_w_max[ESR], esr_to_p_i_max[ESR])
    sub.stor_net_i = pyo.Var(sub.ESR, sub.H_all, bounds=get_net_bounds)

    # Storage level in kWh
    # Create function to get max storage level for corresponding ESR.
    def get_stor_levels(sub, ESR, H):
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
    sub.stor_level = pyo.Var(
        sub.ESR,
        sub.H_all,
        within=pyo.NonNegativeReals,
        bounds=get_stor_levels
    )
    # Initialization of storage level and net injections for all ESR.
    for esr_type in esr_types:
        if 'ev' in esr_type:
            for h in sub.H_all:
                if h < plug_in_hour or h > plug_out_hour:
                    # Fix stor level and net injection values to zero before ev is
                    # plugged in and after ev is plugged out.
                    sub.stor_level[esr_type, h].fix(0)
                    sub.stor_net_i[esr_type, h].fix(0)
            sub.stor_level[esr_type, plug_out_hour].fix(
                charge_target * esr_to_stor_level_max[esr_type]
            )
        else:
            sub.stor_level[esr_type, 0].fix(esr_to_stor_level_zero[esr_type])

    # **************************************************************************
    # Objective function
    # **************************************************************************

    sub.OBJ = pyo.Objective(
        expr=sum(c2*sub.pg[h] +l2*sub.p2[h] for h in sub.H)
    )

    # **************************************************************************
    # Constraints
    # **************************************************************************

    # Load must be covered by production, purchasing electrictiy or by the
    # sum of net injections over all esr.
    def con_load(sub, H):
        return (sub.pg[H] + sub.p1[H] + sub.p2[H]
            + sum(sub.stor_net_i[esr, H] for esr in sub.ESR)
            ) >= sub.load_values[H]
    sub.con_load = pyo.Constraint(sub.H, rule=con_load)

    # Maximum capacity of generator
    def con_max(sub, H):
        return sub.pg[H] <= pmax*sub.u[H]
    sub.con_max = pyo.Constraint(sub.H, rule=con_max)

    # Ensure variable u is equal to the solution of the master problem.
    def dual_con1(sub, H):
        return sub.u[H] == results_master['u'][H]
    sub.dual_con1 = pyo.Constraint(sub.H_all, rule=dual_con1)

    # Ensure variable p1 is equal to the solution of the master problem.
    def dual_con2(sub, H):
        return sub.p1[H] == results_master['p1'][H]
    sub.dual_con2 = pyo.Constraint(sub.H_all, rule=dual_con2)

    # Ramping constraint of generator
    def con_ramping(sub, H):
        return (
            -ramping_constraint, sub.pg[H] - sub.pg[H-1], ramping_constraint
        )
    sub.con_ramping = pyo.Constraint(sub.H, rule=con_ramping)

    # Storage Balance
    def stor_balance(sub, ESR, H):
        if 'ev' not in ESR:
            return sub.stor_level[ESR, H] == (
                sub.stor_level[ESR, H-1] - sub.stor_net_i[ESR, H])
        else:
            if H > plug_in_hour and H <= plug_out_hour:
                return sub.stor_level[ESR, H] == (
                    sub.stor_level[ESR, H-1] - sub.stor_net_i[ESR, H]
                )
            if H == plug_in_hour:
                return sub.stor_level[ESR, H] == (
                    esr_to_stor_level_zero[ESR] - sub.stor_net_i[ESR, H]
                )
            else:
                return pyo.Constraint.Skip

    sub.stor_balance = pyo.Constraint(sub.ESR, sub.H, rule=stor_balance)

    #---------------------------------------------------------------------------
    # Initialization of sub problem
    #---------------------------------------------------------------------------

    # enable calculation of dual variables in pyomo
    sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    print(f'Solving sub problem for samples size = {sample_size}')

    if multiprocessing:
        # Use concurrent package to enable multiprocessing to solve test samples
        # in parallel.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(
                solve_sample,
                SAMPLES,
                list(range(len(SAMPLES))),
                [len(SAMPLES)] * len(SAMPLES),
                [sub] * len(SAMPLES),
                [opt] * len(SAMPLES)
            )
    else:
        # Solve model per iteration.
        results = map(
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

        if multiprocessing:
            # Use concurrent package to enable multiprocessing to solve test samples
            # in parallel.
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(
                    solve_sample,
                    SAMPLES,
                    list(range(len(SAMPLES))),
                    [len(SAMPLES)] * len(SAMPLES),
                    [sub] * len(SAMPLES),
                    [opt] * len(SAMPLES)
                )
        else:
            # Solve model per iteration.
            results = map(
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

    ############################################################################
    ### Results & Exports
    ############################################################################

    current_path = Path.cwd()

    path = get_path_by_task(
        mc_sampling=mc_sampling,
        av_sampling=av_sampling,
        deterministic=deterministic,
        sample_size=sample_size,
        multiprocessing=True,
        current_path=current_path)

    # Make sure that folders exist
    if not os.path.exists(path):
        os.makedirs(path)

    #---------------------------------------------------------------------------
    # Model results
    #---------------------------------------------------------------------------

    if output:
        # Save only the sub results of the last iteration and last sample.
        with open(
            os.path.join(path, f'results_sub_{charge_target}.json'),
            'w') as outfile:
            last_key = list(results_sub.keys())[-1]
            # Check for variables containing a tuple key.
            for variable in results_sub[last_key]:
                first_key = list(results_sub[last_key][variable].keys())[0]
                if type(first_key) == tuple:
                    # Reset all keys for that variable.
                    results_sub[last_key][variable] = reset_tuple_key(
                        results_sub[last_key][variable]
                    )
            json.dump(results_sub[last_key], outfile)

        with open(
            os.path.join(path, f'results_master_{charge_target}.json'),
            'w') as outfile:
            json.dump(results_master, outfile)

    #---------------------------------------------------------------------------
    # Upper and lower bound
    #---------------------------------------------------------------------------

    if output:
        # Export upper and lower bounds
        np.array(objective_values).tofile(
            os.path.join(path, f'objective_values_{charge_target}.csv'),
            sep = ','
        )
        np.array(lower_bounds).tofile(
            os.path.join(path, f'lower_bounds_{charge_target}.csv'),
            sep = ','
        )

    #---------------------------------------------------------------------------
        # Computation time
    #---------------------------------------------------------------------------

    time_end_all = tm.time()
    times_dic['time'].append(time_end_all - time_start_all)
    print(f'Computation time: {round(time_end_all - time_start_all,2)}')
    times_dic['iterations'].append(iteration)

    if output:
        # save computation times as csv
        df_time = pd.DataFrame(times_dic)
        df_time.to_csv(
            os.path.join(path, f'computation_times.csv'),
            index=False)










    obj_dict = {'obj_x_star':0,'obj_x_cand':0}


    for nn,sizes in enumerate([sample_size,x_cand]):
        ############################################################################
        ### Testing
        ############################################################################

        # ******************************************************************************
        # Model Options
        # ******************************************************************************

        # Model options are defined in the file 'model_options.py'.

        multiprocessing_testing = False

        # ******************************************************************************
        # Parameters
        # ******************************************************************************

        # Seet for randomness
        #seed = 12

        # Get test samples.
        TEST_SAMPLES = get_monte_carlo_samples(LOADS, sample_size=test_size, seed=seed)

        # Define a list of approaches.
        APPROACHES = ['stochastic']

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
                sample_size=sizes,
                multiprocessing=True,
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

            if nn == 0:
                obj_dict['obj_x_star'] = mean
            
            if nn == 1:
                obj_dict['obj_x_cand'] = mean

            print()
            print_caption('End')
            
    #print(obj_dict)

    gap =  obj_dict['obj_x_cand'] - obj_dict['obj_x_star']
    gaps.append(gap)
    print(gap)

    sample_size = sample_size + 1

print('gaps:')
print(gaps)