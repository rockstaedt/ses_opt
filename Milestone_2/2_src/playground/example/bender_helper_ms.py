
import pyomo.environ as pyo
from pyomo.core import Var, value

def solve_model(solver, model):
    solver.solve(model)

def get_results(model, dual=False, write=True):
    dic = {}
    for v in model.component_objects(Var, active=True):
        varobject = getattr(model, str(v))
        dic2 = {}
        for index in varobject:
            dic2[index] = varobject[index].value
            if write:
                print('\tVariable', v, index, varobject[index].value)
        dic[str(v)] = dic2
    if dual:
        for c in model.component_objects(pyo.Constraint, active=True):
            if str(c) == 'dual_con1' or str(c) == 'dual_con2':
                if len(model.dual) != 0:
                    dic2 = {}
                    for index in c:
                        dic2[index] = model.dual[c[index]]
                        if write:
                            print("\tLambda ", c, index, model.dual[c[index]])
                else:
                    dic2 = {}
                    for i in range(0, 24):
                        dic2[i] = 0
                        if write:
                            print('\tLambda', c, i, 0)
                dic[str(c)] = dic2
    return dic

def convergence_check(objective, master_prob, u, p1, pg, p2, alpha,
                      epsilon=0.001):
    upper_bound = objective(u, p1, pg, p2)
    print(f'Upper Bound: {upper_bound}')
    lower_bound = master_prob(u, p1, alpha)
    print(f'Lower Bound: {lower_bound}')
    diff = abs(upper_bound - lower_bound)
    print(f'Difference: {diff}')
    return not abs(upper_bound-lower_bound) > epsilon