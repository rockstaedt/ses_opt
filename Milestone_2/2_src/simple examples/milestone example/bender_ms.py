import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Var, value
import time as tm
import numpy as np

from bender_helper_ms import *

###############################################################################
### Model Options
###############################################################################

epsilon = 0.0001

opt = pyo.SolverFactory('gurobi')

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
pl = [8,8,10,10,10,16,22,24,26,32,30,28,22,18,16,16,20,24,28,34,38,30,22,12]


###############################################################################
# Monte Carlo sampling
###############################################################################
# covariance matrix of pl
cov_pl = np.diagflat(np.array(pl)*1/3)
# 1000 random samples from normal distribution
samples = np.random.multivariate_normal(pl, cov_pl, size= 1000)


###############################################################################
### Benders decomposition
###############################################################################

def objective(u, p1, pg, p2):
    return sum(c1*u[h] + l1*p1[h] + c2*pg[h] + l2*p2[h] for h in range(0, 24))

def master_prob(u,p1, alpha):
    return sum(c1*u[h] + l1*p1[h] + alpha[h] for h in range(0, 24))

# master problem

master = pyo.ConcreteModel()

# Hour Set
master.H = pyo.RangeSet(0, 23)

master.u = pyo.Var(master.H, within=pyo.Binary)
master.p1 = pyo.Var(master.H, within=pyo.NonNegativeReals)
master.alpha = pyo.Var(master.H)

def master_obj(master):
    return sum(
        c1*master.u[h] + l1*master.p1[h] + master.alpha[h] for h in master.H
    )
master.OBJ = pyo.Objective(rule=master_obj)

# alpha down (-500) is an arbitrary selected bound
def alphacon1(master, H):
    return master.alpha[H] >= -500
master.alphacon1 = pyo.Constraint(master.H, rule=alphacon1)

# First iteration
start_time = tm.time()

iteration = 0
print('######################################################################')
print('Initialization')
print('######################################################################')
print()

solve_model(opt, master)

print('Solutions for master problem')

results_master = get_results(master)

# subproblem

sub = pyo.ConcreteModel()

#Hour Set
sub.H = pyo.RangeSet(0,23)

# no need for declaration of variable types because that is determined by
# corresponding variables of master problem
sub.u = pyo.Var(sub.H)
sub.p1 = Var(sub.H)

sub.pg = pyo.Var(sub.H, within=pyo.NonNegativeReals)
sub.p2 = pyo.Var(sub.H, within=pyo.NonNegativeReals)

sub.OBJ = pyo.Objective(
    expr=sum(c2*sub.pg[h] +l2*sub.p2[h] for h in sub.H)
)

def con_load(sub, H):
    return sub.pg[H] + sub.p1[H] + sub.p2[H] >= pl[H]
sub.con_load = pyo.Constraint(sub.H, rule=con_load)

def con_max(sub, H):
    return sub.pg[H] <= pmax*sub.u[H]
sub.con_max = pyo.Constraint(sub.H, rule=con_max)

def dual_con1(sub, H):
    return sub.u[H] == results_master['u'][H]
sub.dual_con1 = pyo.Constraint(sub.H, rule=dual_con1)

def dual_con2(sub, H):
    return sub.p1[H] == results_master['p1'][H]
sub.dual_con2 = pyo.Constraint(sub.H, rule=dual_con2)

sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

solve_model(opt, sub)

print('Solutions for sub problem')

results_sub = get_results(sub, dual=True)

converged = convergence_check(
    objective,
    master_prob,
    results_master['u'],
    results_master['p1'],
    results_sub['pg'],
    results_sub['p2'],
    results_master['alpha'],
    epsilon=epsilon
)

while not converged:
    iteration += 1
    print()
    print('--> Not converging. Next iteration.')
    print()
    print('######################################################################')
    print(f'Iteration = {iteration}')
    print('######################################################################')
    print()

    def cut(master, H):
        return (
            c2*results_sub['pg'][H] + l2*results_sub['p2'][H]
            + results_sub['dual_con1'][H]*(
                master.u[H] - results_master['u'][H])
            + results_sub['dual_con2'][H]*(master.p1[H] - results_master['p1'][H])
            <= master.alpha[H]
        )

    setattr(master, f'cut_{iteration}', pyo.Constraint(master.H, rule=cut))
    print(f'Added cut_{iteration}')

    solve_model(opt, master)
    print('Solutions for master problem')
    results_master = get_results(master, write=False)

    # update dual constraint in sub problem
    sub.dual_con1.reconstruct()
    sub.dual_con2.reconstruct()
    solve_model(opt, sub)
    print('Solutions for sub problem')
    results_sub = get_results(sub, dual=True, write=False)

    converged = convergence_check(
        objective,
        master_prob,
        results_master['u'],
        results_master['p1'],
        results_sub['pg'],
        results_sub['p2'],
        results_master['alpha'],
        epsilon=epsilon
    )


print('######################################################################')
print('######################################################################')
print('END RESULTS')
print()

print('Solutions for master problem')
results_master = get_results(master)

print('Solutions for sub problem')
results_sub = get_results(sub, dual=True)

time_end = tm.time()
print(time_end - start_time)