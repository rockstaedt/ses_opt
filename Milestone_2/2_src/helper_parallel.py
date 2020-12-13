import pyomo.environ as pyo
from pyomo.core import Var, value
import numpy as np

np.random.seed(42)

def get_monte_carlo_samples(values:list, samples=1000):
    """
    This function creates a monte carlo sample of size samples. For every
    element in values, a sample is drawn from a normal distribution with the
    elements value as mean and one third from the elements value as deviation.
    """
    # covariance matrix of pl
    cov_pl = np.diagflat(np.array(values)*1/3)

    # return random samples from normal distribution
    return np.random.multivariate_normal(values, cov_pl, size=samples)

def print_caption(name:str):
    print('###################################################################')
    print(name)
    print('###################################################################')
    print()

def print_convergence(converged:bool):
    if not converged:
        print()
        print('--> Not converging. Next iteration.')
        print()
    else:
        print()
        print('--> Converged. Stop algorithm.')
        print()

def print_status(i:int):
    # increase by one because index of samples starts with zero
    i += 1
    if i < 100:
        print('.', end='')
    elif i == 100:
        print(i)
    elif i < 200:
        print('.', end='')
    elif i == 200:
        print(i)
    elif i < 300:
        print('.', end='')
    elif i == 300:
        print(i)
    elif i < 400:
        print('.', end='')
    elif i == 400:
        print(i)
    elif i < 500:
        print('.', end='')
    elif i == 500:
        print(i)
    elif i < 600:
        print('.', end='')
    elif i == 600:
        print(i)
    elif i < 700:
        print('.', end='')
    elif i == 700:
        print(i)
    elif i < 800:
        print('.', end='')
    elif i == 800:
        print(i)
    elif i < 900:
        print('.', end='')
    elif i == 900:
        print(i)
    elif i < 1000:
        print('.', end='')
    elif i == 1000:
        print(i)


def solve_model(solver, model):
    solver.solve(model)

def get_results(model, dual=False, write=False):
    dic = {}
    for v in model.component_objects(Var, active=True):
        dic[str(v)] = getattr(model, str(v)).value
        if write:
            print('\tVariable', v, getattr(model, str(v)).value)
    if dual:
        for c in model.component_objects(pyo.Constraint, active=True):
            if str(c) == 'dual_con1' or str(c) == 'dual_con2':
                dic[str(c)] = model.dual[c[None]]
                if write:
                    print('\tDual Variable lambda', model.dual[c[None]])
    return dic

def convergence_check(objective, master_prob, results_master, results_sub,
                      samples, epsilon=0.001):
    """
    This function checks if the lower bound and upper bound are converged. It
    returns a boolean and the difference of the bounds.
    """
    upper_bound = 0
    for i, sample in enumerate(samples):
        upper_bound += objective(
            results_sub[i]['u'],
            results_sub[i]['p1'],
            results_sub[i]['pg'],
            results_sub[i]['p2']
        )
    upper_bound = upper_bound/len(samples)
    lower_bound = master_prob(
        results_master['u'],
        results_master['p1'],
        results_master['alpha']
    )
    diff = abs(upper_bound - lower_bound)
    return (not diff > epsilon, diff, upper_bound)