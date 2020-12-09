import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Var, value

from bender_helper import *

epsilon = 0.0001

opt = pyo.SolverFactory('gurobi')

def objective(x,y):
    return -1/4*x - y

def master_prob(x, alpha):
    return -1/4*x + alpha

# master problem

master = pyo.ConcreteModel()

master.x = pyo.Var()

master.alpha = pyo.Var()

def master_obj(master):
    return -1/4*master.x + master.alpha
master.OBJ = pyo.Objective(rule=master_obj)

def xcon(master):
    return (0, master.x, 16)
master.xcon = pyo.Constraint(rule=xcon)

def alphacon1(master):
    return -25 <= master.alpha
master.alphacon1 = pyo.Constraint(rule=alphacon1)

# First iteration
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

sub.y = pyo.Var(within=pyo.NonNegativeReals)

sub.x = pyo.Var()

sub.OBJ = pyo.Objective(
    expr=-sub.y
)

def con1(model):
    return sub.y - sub.x <= 5
sub.con1 = pyo.Constraint(rule=con1)

def con2(model):
    return sub.y - 0.5*sub.x <= 15/2
sub.con2 = pyo.Constraint(rule=con2)

def con3(model):
    return sub.y + 0.5*sub.x <= 35/2
sub.con3 = pyo.Constraint(rule=con3)

def con4(model):
    return -sub.y + sub.x <= 10
sub.con4 = pyo.Constraint(rule=con4)

def dual_con(model):
    return sub.x == results_master['x']
sub.dual_con = pyo.Constraint(rule=dual_con)

sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

solve_model(opt, sub)

print('Solutions for sub problem')

results_sub = get_results(sub, dual=True)

converging = convergence_check(
    objective,
    master_prob,
    results_master['x'],
    results_sub['y'],
    results_master['alpha'],
    epsilon=epsilon
)

while not converging:
    iteration += 1
    print()
    print('--> Not converging. Next iteration.')
    print()
    print('######################################################################')
    print(f'Iteration = {iteration}')
    print('######################################################################')
    print()

    def cut(master):
        return -results_sub['y'] + results_sub['lambda']*(
            master.x - results_master['x']) <= master.alpha
    setattr(master, f'cut_{iteration}', pyo.Constraint(rule=cut))
    print(f'Added cut_{iteration}')

    solve_model(opt, master)
    print('Solutions for master problem')
    results_master = get_results(master)

    # update dual constraint in sub problem
    sub.dual_con.reconstruct()
    solve_model(opt, sub)
    print('Solutions for sub problem')
    results_sub = get_results(sub, dual=True)

    converging = convergence_check(
    objective,
    master_prob,
    results_master['x'],
    results_sub['y'],
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