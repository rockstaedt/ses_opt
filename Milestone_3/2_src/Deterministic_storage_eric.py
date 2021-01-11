import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time as tm
import numpy as np
import json
import pandas as pd
import math

import opt_helper

###############################################################################
### Model Options
###############################################################################

# enables sensitivity analysis regarding the elecitricity price of the real
# time contract from 15 to 35 in steps of 5
sensitivity_analysis = False

# enables the output of csv files, saved into '3_results'
csv_output = False

###############################################################################
### Parameters
###############################################################################

# fixed generator costs in $/h
c1 = 2.12*10**-5

# linear generator costs in $/kWh
c2 = 0.128

# maximum capacity of generator
pmax = 12

# minimum uptime of generator in hours
uptime = 3

# minimum downtime of generator in hours
downtime = 4

# ramping constraint of generator in kW
ramping_constraint = 5

# electricity price forward contract in $/kWh
l1 = 0.25

# maximum charging power of storage in kW
p_w_max = 10

# maxium discharging power of storage in kW
p_i_max = p_w_max

# maximum storage level in kWh
stor_level_max = 5

# overall storage efficiency
eff = 1

# charging efficiency
eff_w = math.sqrt(eff)

# discharging efficiency
eff_i = eff_w

# electricity price real time contract in $/kWh
# this parameter is part of the sensitivity analysis that is why the
# parameter is defined as a list
if sensitivity_analysis:
    l2s = [0.15, 0.20, 0.25, 0.30, 0.35]
    # stopp a new time here to get the time period of the sensitivity analysis
    time_start_sens = tm.time()
else:
    l2s = [0.3]

# load values in kW
LOADS = opt_helper.get_loads()

# hours
HOURS = list(range(0, len(LOADS)))

# arbitrary value for convergence check
epsilon = 0.0001

###############################################################################
### Benders decomposition
###############################################################################

# solver for MIP
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

# dataframe for computation times
times_dic = {'l2': [], 'time': []}

# loop over all real time prices and solve the L-shape method
for l2 in l2s:

    opt_helper.print_sens_step(f'Solve benders decomposition for {l2} $/kWh')

    #---------------------------------------------------------------------------
    # Helper variables
    #---------------------------------------------------------------------------

    # list for the differences of the bounds
    bounds_difference = []

    # list for the objective values
    objective_values = []

    # list for lower bound values
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

    # hour set
    master.H = pyo.RangeSet(0,24)

    # **************************************************************************
    # Variables
    # **************************************************************************

    # unit commitment for generator
    master.u = pyo.Var(master.H, within=pyo.Binary)

    # electricity purchased with the forward contract
    master.p1 = pyo.Var(master.H, within=pyo.NonNegativeReals)

    # value function for second stage problem
    master.alpha = pyo.Var(master.H)

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

    # alpha down (-500) is an arbitrary selected bound
    def alphacon1(master, H):
        return master.alpha[H] >= -500
    master.alphacon1 = pyo.Constraint(master.H, rule=alphacon1)

    # initialization for forward contract
    def p1_zero(master):
        return master.p1[0] == 0
    master.p1_zero = pyo.Constraint(rule=p1_zero)

    # minimum uptime constraint
    def min_uptime(master, H):
        # In order to avoid applying this constraint for hour 0, check for index.
        if H != 0:
            # Apply minimum uptime constraint.
            # The end value of the range function needs to be increased to
            # be included.
            V = list(range(H, min([H-1 + uptime, len(HOURS)-1]) + 1))
            # For the last hour, the ouput of the range function is 0 because
            # range(24,24). To include hour 24 into the list, check for length
            # and put hour 24 into V.
            if len(V) == 0:
                V = [H]
            # Return the sum of all hours in V to apply the constraint for all
            # hours in V.
            return sum(
                master.u[H] - master.u[H-1] for v in V
                ) <= sum(master.u[v] for v in V)
        else:
            # Initialize unit commitment in hour 0.
            return master.u[H] == 0
    master.min_uptime = pyo.Constraint(master.H, rule=min_uptime)

    # minimum downtime constraint
    def min_downtime(master, H):
        # In order to avoid applying this constraint for hour 0, check for index.
        if H != 0:
            # Apply minimum downtime constraint.
            # The end value of the range function needs to be increased to
            # be included.
            V = list(range(H, min([H-1 + downtime, len(HOURS)-1]) + 1))
            # For the last hour, the ouput of the range function is 0 because
            # range(24,24). To include hour 24 into the list, check for length
            # and put hour 24 into V.
            if len(V) == 0:
                V = [H]
            # Return the sum of all hours in V to apply the constraint for all
            # hours in V.
            return sum(
                master.u[H-1] - master.u[H] for v in V
                ) <= sum(1 - master.u[v] for v in V)
        else:
            # Initialize unit commitment in hour 0.
            return master.u[H] == 0
    master.min_downtime = pyo.Constraint(master.H, rule=min_downtime)

    #---------------------------------------------------------------------------
    # Initialization of master problem
    #---------------------------------------------------------------------------

    # save current time to get the time of calculating
    time_start = tm.time()

    # initialize iteration counter
    iteration = 0

    opt_helper.print_caption('Initialization')

    print('Solving master problem...')

    opt_helper.solve_model(opt, master)

    results_master = opt_helper.get_results(master, write=True)

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # Sub problem
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    sub = pyo.ConcreteModel()

    # **************************************************************************
    # Sets
    # **************************************************************************

    # hour set
    sub.H = pyo.RangeSet(0,24)

    # **************************************************************************
    # Variables
    # **************************************************************************

    # First stage variables

    # no need for declaration of variable types because that is determined by
    # corresponding variables of master problem
    sub.u = pyo.Var(sub.H)
    sub.p1 = pyo.Var(sub.H)

    # Second stage variables

    # electricity produced by generator
    sub.pg = pyo.Var(sub.H, within=pyo.NonNegativeReals)

    # electrictiy bought from retailer
    sub.p2 = pyo.Var(sub.H, within=pyo.NonNegativeReals)

    # injection (discharging) status of storage
    sub.u_stor_i = pyo.Var(sub.H, within=pyo.Binary)

    # withdrawal (charging) status of storage
    sub.u_stor_w = pyo.Var(sub.H, within=pyo.Binary)

    # power injection of storage
    sub.p_stor_i = pyo.Var(sub.H, within=pyo.NonNegativeReals)

    # power withdrawahl of storage
    sub.p_stor_w = pyo.Var(sub.H, within=pyo.NonNegativeReals)

    # storage level
    sub.stor_level = pyo.Var(sub.H, within=pyo.NonNegativeReals)

    # **************************************************************************
    # Objective function
    # **************************************************************************

    sub.OBJ = pyo.Objective(
        expr=sum(c2*sub.pg[h] +l2*sub.p2[h] for h in sub.H)
    )

    # **************************************************************************
    # Constraints
    # **************************************************************************

    # load must be covered by production, purchasing electrictiy or by storage
    def con_load(sub, H):
        return (sub.pg[H] + sub.p1[H] + sub.p2[H] +
            sub.p_stor_i[H] - sub.p_stor_w[H]) >= LOADS[H]
    sub.con_load = pyo.Constraint(sub.H, rule=con_load)

    # maximum capacity of generator
    def con_max(sub, H):
        return sub.pg[H] <= pmax*sub.u[H]
    sub.con_max = pyo.Constraint(sub.H, rule=con_max)

    # ramping constraint of generator
    def con_ramping(sub, H):
        if H != 0:
            return (
                -ramping_constraint,
                sub.pg[H] - sub.pg[H-1],
                ramping_constraint
            )
        else:
            return sub.pg[H] == 0
    sub.con_ramping = pyo.Constraint(sub.H, rule=con_ramping)

    # ensure variable u is equal to the solution of the master problem
    def dual_con1(sub, H):
        return sub.u[H] == results_master['u'][H]
    sub.dual_con1 = pyo.Constraint(sub.H, rule=dual_con1)

    # ensure variable p1 is equal to the solution of the master problem
    def dual_con2(sub, H):
        return sub.p1[H] == results_master['p1'][H]
    sub.dual_con2 = pyo.Constraint(sub.H, rule=dual_con2)

    # Storage can only withdraw or inject power in one hour.
    def consistency_storage(sub, H):
        if H != 0:
            return sub.u_stor_i[H] + sub.u_stor_w[H] <= 1
        else:
            return sub.u_stor_i[H] + sub.u_stor_w[H] == 0
    sub.consistency_storage = pyo.Constraint(sub.H, rule=consistency_storage)

    def test(sub, H):
        if sub.u_stor_i[H] >= 1:
            return

    # Maximum charging power
    def max_charge(sub, H):
        return sub.p_stor_w[H] <= sub.u_stor_w[H] * p_w_max
    sub.max_charge = pyo.Constraint(sub.H, rule=max_charge)

    # Maximum discharging power
    def max_discharge(sub, H):
        return sub.p_stor_i[H] <= sub.u_stor_i[H] * p_i_max
    sub.max_discharge = pyo.Constraint(sub.H, rule=max_discharge)

    # Maximum Storage Level
    def max_storage(sub, H):
        return sub.stor_level[H] <= stor_level_max
    sub.max_storage = pyo.Constraint(sub.H, rule=max_storage)

    # Storage Balance
    def stor_balance(sub, H):
        if H != 0:
            return sub.stor_level[H] == (
                sub.stor_level[H-1]
                + 1/(eff_w)*sub.p_stor_w[H]
                - eff_i*sub.p_stor_i[H]
            )
        else:
            return sub.stor_level[H] == 0
    sub.stor_balance = pyo.Constraint(sub.H, rule=stor_balance)

    #---------------------------------------------------------------------------
    # Initialization of sub problem
    #---------------------------------------------------------------------------

    print('Solving sub problem Binary...')

    # Solve sub problem with binary status variable for storage but without
    # calculating the dual variable. This is due to the fact that Pyomo can
    # not calculate the dual variables if the problem contains binary variables.
    opt_helper.solve_model(opt, sub)
    results_sub = {}
    results_sub[0] = opt_helper.get_results(sub, dual=False, write=False)

    sub2 = pyo.ConcreteModel()

    sub2.H = pyo.RangeSet(0,24)

    sub2.u = pyo.Var(sub2.H)
    sub2.p1 = pyo.Var(sub2.H)
    sub2.pg = pyo.Var(sub2.H, within=pyo.NonNegativeReals)
    sub2.p2 = pyo.Var(sub2.H, within=pyo.NonNegativeReals)
    sub2.p_stor_i = pyo.Var(sub2.H, within=pyo.NonNegativeReals)
    sub2.p_stor_w = pyo.Var(sub2.H, within=pyo.NonNegativeReals)
    sub2.stor_level = pyo.Var(sub2.H, within=pyo.NonNegativeReals)
    sub2.u_stor_i = pyo.Var(sub2.H)
    sub2.u_stor_w = pyo.Var(sub2.H)

    sub2.OBJ = pyo.Objective(
        expr=sum(c2*sub2.pg[h] +l2*sub2.p2[h] for h in sub2.H)
    )

    def dual_con1(sub2, H):
        return sub2.u[H] == results_master['u'][H]
    sub2.dual_con1 = pyo.Constraint(sub2.H, rule=dual_con1)

    def dual_con2(sub2, H):
        return sub2.p1[H] == results_master['p1'][H]
    sub2.dual_con2 = pyo.Constraint(sub2.H, rule=dual_con2)

    # Create constraints to set the variables of the subproblem to the
    # obtained results
    def set_u_stor_i(sub2, H):
        return sub2.u_stor_i[H] == results_sub[0]['u_stor_i'][H]
    sub2.set_u_stor_i = pyo.Constraint(sub2.H, rule=set_u_stor_i)

    def set_u_stor_w(sub2, H):
        return sub2.u_stor_w[H] == results_sub[0]['u_stor_w'][H]
    sub2.set_u_stor_w = pyo.Constraint(sub2.H, rule=set_u_stor_w)

    def set_p_stor_i(sub2, H):
        return sub2.p_stor_i[H] == results_sub[0]['p_stor_i'][H]
    sub2.set_p_stor_i = pyo.Constraint(sub2.H, rule=set_p_stor_i)

    def set_p_stor_w(sub2, H):
        return sub2.p_stor_w[H] == results_sub[0]['p_stor_w'][H]
    sub2.set_p_stor_w = pyo.Constraint(sub2.H, rule=set_p_stor_w)

    def set_stor_level(sub2, H):
        return sub2.stor_level[H] == results_sub[0]['stor_level'][H]
    sub2.set_stor_level = pyo.Constraint(sub2.H, rule=set_stor_level)

    def set_pg(sub2, H):
        return sub2.pg[H] == results_sub[0]['pg'][H]
    sub2.set_pg = pyo.Constraint(sub2.H, rule=set_pg)

    def set_p2(sub2, H):
        return sub2.p2[H] == results_sub[0]['p2'][H]
    sub2.set_p2 = pyo.Constraint(sub2.H, rule=set_p2)

    # enable calculation of dual variables in pyomo
    sub2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    print('Solving sub problem without Binary...')

    opt_helper.solve_model(opt, sub2)
    results_sub = {}
    results_sub[0] = opt_helper.get_results(sub2, dual=True, write=False)

    # check if upper and lower bound are converging
    converged, upper_bound, lower_bound = opt_helper.convergence_check(
        objective,
        master_prob,
        results_master,
        results_sub,
        samples=[0],
        epsilon=epsilon
    )

    opt_helper.print_convergence(converged)

    bounds_difference.append(abs(upper_bound - lower_bound))

    objective_values.append(upper_bound)

    lower_bounds.append(lower_bound)

    # optimize until upper and lower bound are converging
    while not converged:
        iteration += 1

        opt_helper.print_caption(f'Iteration {iteration}')

        def cut(master, H):
            return (
                c2*results_sub[0]['pg'][H] + l2*results_sub[0]['p2'][H]
                + results_sub[0]['dual_con1'][H]*(
                    master.u[H] - results_master['u'][H])
                + results_sub[0]['dual_con2'][H]*(
                    master.p1[H] - results_master['p1'][H])
                <= master.alpha[H]
            )

        setattr(master, f'cut_{iteration}', pyo.Constraint(master.H, rule=cut))
        print(f'Added cut_{iteration}')

        print('Solving master problem...')
        opt_helper.solve_model(opt, master)
        results_master = opt_helper.get_results(master)

        # update dual constraint in sub problem
        sub.dual_con1.reconstruct()
        sub.dual_con2.reconstruct()

        print('Solving sub problem for all samples...')
        results_sub = {}
        opt_helper.solve_model(opt, sub)
        results_sub[0] = opt_helper.get_results(sub, dual=False, write=False)

        sub2.dual_con1.reconstruct()
        sub2.dual_con2.reconstruct()
        sub2.set_u_stor_i.reconstruct()
        sub2.set_u_stor_w.reconstruct()
        sub2.set_p_stor_i.reconstruct()
        sub2.set_p_stor_w.reconstruct()
        sub2.set_stor_level.reconstruct()
        sub2.set_pg.reconstruct()
        sub2.set_p2.reconstruct()

        results_sub = {}
        opt_helper.solve_model(opt, sub2)
        results_sub[0] = opt_helper.get_results(sub2, dual=False, write=False)

        converged, upper_bound, lower_bound = opt_helper.convergence_check(
            objective,
            master_prob,
            results_master,
            results_sub,
            samples=[0],
            epsilon=epsilon
        )

        opt_helper.print_convergence(converged)

        bounds_difference.append(abs(upper_bound - lower_bound))

        objective_values.append(upper_bound)

        lower_bounds.append(lower_bound)

    ############################################################################
    ### Results
    ############################################################################

    opt_helper.print_caption('End Results')

    path = '../3_results/'

    with open(f'{path}deterministic_results_sub_{l2}.json', 'w') as outfile:
        json.dump(results_sub[0], outfile)

    with open(f'{path}deterministic_results_master_{l2}.json', 'w') as outfile:
        json.dump(results_master, outfile)

    time_end = tm.time()
    times_dic['l2'].append(str(l2))
    times_dic['time'].append(time_end - time_start)
    print('Computation time:')
    print(f'\t{round(time_end - time_start, 2)}s')

    ############################################################################
    ### Exports
    ############################################################################

    if sensitivity_analysis:
        path += 'sensitivity analysis/deterministic_'
        prefix = f'_sensitivity_{l2}.csv'
    else:
        path = 'deterministic_'
        prefix = '_no_sensitivity.csv'

    if csv_output:
        # export objective values, difference between bound into
        # '3_results' as CSV
        np.array(objective_values).tofile(
            path + 'objective_values' + prefix,
            sep = ','
        )
        np.array(upper_bound).tofile(
            path + 'upper_bounds' + prefix,
            sep = ','
        )
        np.array(lower_bound).tofile(
            path + 'lower_bounds' + prefix,
            sep = ','
        )
        np.array(bounds_difference).tofile(
            path + 'bounds_differences' + prefix,
            sep = ','
        )

if sensitivity_analysis:
    time_end_sens = tm.time()
    times_dic['l2'].append('ALL')
    times_dic['time'].append(time_end_sens - time_start_sens)
    print('Computation time for sensitivity analysis:')
    print(f'\t{round(time_end_sens - time_start_sens, 2)}s')

if csv_output:
    # save computation times as csv
    df_time = pd.DataFrame(times_dic)
    df_time.to_csv(
        '../3_results/deterministic_computation_times.csv',
        index=False
    )
