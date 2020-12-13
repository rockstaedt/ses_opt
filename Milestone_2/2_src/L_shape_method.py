import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time as tm

from helper import *

###############################################################################
### Model Options
###############################################################################

# enables sensitivity analysis regarding the elecitricity price of the real
# time contract from 15 to 35 in steps of 5
sensitivity_analysis = True

# sample size for monte carlo simulation
sample_size = 5

###############################################################################
### Parameters
###############################################################################

# fixed generator costs in $/h
c1 = 2.12*10**-5

# linear generator costs in $/kWh
c2 = 0.128

# maximum capacity of generator
pmax = 12

# electricity price forward contract in $/kWh
l1 = 0.25

# electricity price real time contract in $/kWh
l2 = 0.3

# load values in kW
LOADS = [8,8,10,10,10,16,22,24,26,32,30,28,22,18,16,16,20,24,28,34,38,30,22,12]

# monte carlo samples
SAMPLES = get_monte_carlo_samples(LOADS, samples=sample_size)

# hours
HOURS = list(range(0, len(LOADS)))

# arbitrary value for convergence check
epsilon = 0.0001

###############################################################################
### Benders decomposition
###############################################################################

# solver for MIP
opt = pyo.SolverFactory('gurobi')

# list for the differences of the bounds
bounds_difference = []

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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Master problem
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

master = pyo.ConcreteModel()

# *****************************************************************************
# Sets
# *****************************************************************************

# hour set
master.H = pyo.RangeSet(0, 23)

# *****************************************************************************
# Variables
# *****************************************************************************

# unit commitment for generator
master.u = pyo.Var(master.H, within=pyo.Binary)

# electricity purchased with the forward contract
master.p1 = pyo.Var(master.H, within=pyo.NonNegativeReals)

# value function for second stage problem
master.alpha = pyo.Var(master.H)

# *****************************************************************************
# Objective function
# *****************************************************************************

def master_obj(master):
    return sum(
        c1*master.u[h] + l1*master.p1[h] + master.alpha[h] for h in master.H
    )
master.OBJ = pyo.Objective(rule=master_obj)

# *****************************************************************************
# Constraints
# *****************************************************************************

# alpha down (-500) is an arbitrary selected bound
def alphacon1(master, H):
    return master.alpha[H] >= -500
master.alphacon1 = pyo.Constraint(master.H, rule=alphacon1)

#------------------------------------------------------------------------------
# Initialization of master problem
#------------------------------------------------------------------------------

# save current time to get the time of calculating
time_start = tm.time()

# initialize iteration counter
iteration = 0

print_caption('Initialization')

print('Solving master problem...')

solve_model(opt, master)

results_master = get_results(master)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Sub problem
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

sub = pyo.ConcreteModel()

# *****************************************************************************
# Sets
# *****************************************************************************

# hour set
sub.H = pyo.RangeSet(0,23)

# *****************************************************************************
# Variables
# *****************************************************************************

# First stage variables

# no need for declaration of variable types because that is determined by
# corresponding variables of master problem
sub.u = pyo.Var(sub.H)
sub.p1 = Var(sub.H)

# Second stage variables

# electricity produced by generator
sub.pg = pyo.Var(sub.H, within=pyo.NonNegativeReals)

# electrictiy bought from retailer
sub.p2 = pyo.Var(sub.H, within=pyo.NonNegativeReals)

# *****************************************************************************
# Objective function
# *****************************************************************************

sub.OBJ = pyo.Objective(
    expr=sum(c2*sub.pg[h] +l2*sub.p2[h] for h in sub.H)
)

# *****************************************************************************
# Constraints
# *****************************************************************************

# load must be covered by production or purchasing electrictiy
# take first random vector from samples
pl = SAMPLES[0, :]
def con_load(sub, H):
    return sub.pg[H] + sub.p1[H] + sub.p2[H] >= pl[H]
sub.con_load = pyo.Constraint(sub.H, rule=con_load)

# maximum capacity of generator
def con_max(sub, H):
    return sub.pg[H] <= pmax*sub.u[H]
sub.con_max = pyo.Constraint(sub.H, rule=con_max)

# ensure variable u is equal to the solution of the master problem
def dual_con1(sub, H):
    return sub.u[H] == results_master['u'][H]
sub.dual_con1 = pyo.Constraint(sub.H, rule=dual_con1)

# ensure variable p1 is equal to the solution of the master problem
def dual_con2(sub, H):
    return sub.p1[H] == results_master['p1'][H]
sub.dual_con2 = pyo.Constraint(sub.H, rule=dual_con2)

#------------------------------------------------------------------------------
# Initialization of sub problem
#------------------------------------------------------------------------------

# enable calculation of dual variables in pyomo
sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

print('Solving sub problem for all samples...')

results_sub = {}
for i, sample in enumerate(SAMPLES):
    print_status(i)
    # filter for first sample because that is set in the initialization of
    # the model
    if i != 0:
        # get new load sample
        pl = sample
        # update constraint
        sub.con_load.reconstruct()
    # solve model
    solve_model(opt, sub)
    results_sub[i] = get_results(sub, dual=True)

print('Done.')

# check if upper and lower bound are converging
converged, diff = convergence_check(
    objective,
    master_prob,
    results_master,
    results_sub,
    samples=SAMPLES,
    epsilon=epsilon
)

print_convergence(converged)

bounds_difference.append(diff)

# optimize until upper and lower bound are converging
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

    print('Solving sub problem for all samples...')
    results_sub = {}
    for i, sample in enumerate(SAMPLES):
        # no if statement here because constraint con load is reconstructed
        # with the first sample in samples
        print_status(i)
        pl = sample
        # update constraint
        sub.con_load.reconstruct()
        # solve model
        solve_model(opt, sub)
        results_sub[i] = get_results(sub, dual=True)

    converged, diff = convergence_check(
        objective,
        master_prob,
        results_master,
        results_sub,
        samples=SAMPLES,
        epsilon=epsilon
    )

    print_convergence(converged)

    bounds_difference.append(diff)


print_caption('End Results')

print('Solutions for master problem')
results_master = get_results(master, write=True)

print('Solutions for sub problem')
results_sub = get_results(sub, dual=True, write=True)

objective_value = objective(
        results_master['u'], results_master['p1'],
        results_sub['pg'], results_sub['p2']
)
print()
print(f'Objective value: {objective_value}')

time_end = tm.time()
print()
print(time_end - time_start)