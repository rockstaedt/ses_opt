
import pyomo.environ as pyo
from pyomo.core import Var, value

def solve_model(solver, model):
    solver.solve(model)

def get_results(model, dual=False, write=True):
    dic = {}
    for v in model.component_objects(Var, active=True):
        dic[str(v)] = getattr(model, str(v)).value
        if write:
            print('\tVariable', v, getattr(model, str(v)).value)
    if dual:
        for c in model.component_objects(pyo.Constraint, active=True):
            if str(c) == 'dual_con':
                dic['lambda'] = model.dual[c[None]]
                if write:
                    print('\tDual Variable lambda', model.dual[c[None]])
    return dic

def convergence_check(objective, master_prob, x, y, alpha,
                      epsilon=0.001):
    upper_bound = objective(x, y)
    lower_bound = master_prob(x, alpha)
    return not upper_bound-lower_bound > epsilon