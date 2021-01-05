import pyomo.environ as pyo
from pyomo.core import Var, value
import numpy as np

def get_monte_carlo_samples(values:list, samples=1000, seed=12):
    """
    This function creates a monte carlo sample of size samples. For every
    element in values, a sample is drawn from a normal distribution with the
    elements value as mean and one third from the elements value as deviation.
    """
    # set seed
    np.random.seed(seed)
    # covariance matrix of pl
    cov_pl = np.diagflat(np.array(values)*1/3)

    # return random samples from normal distribution
    return np.random.multivariate_normal(values, cov_pl, size=samples)

def print_caption(name:str):
    print('###################################################################')
    print(name)
    print('###################################################################')
    print()

def print_sens_step(name:str):
    print('*******************************************************************')
    print('*******************************************************************')
    print(name)
    print('*******************************************************************')
    print('*******************************************************************')
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
    """
    This functions returns a dictionary with the results of the model. Hereby,
    the results are printed if write true. In addition, the solutions of the
    dual variables are added to the dictionary if dual true.
    """
    dic = {}
    # loop through all variables of model
    for v in model.component_objects(Var, active=True):
        varobject = getattr(model, str(v))
        dic2 = {}
        for index in varobject:
            dic2[index] = varobject[index].value
            if write:
                print('\tVariable', v, index, varobject[index].value)
        dic[str(v)] = dic2
    if dual:
        # loop through all constraints
        for c in model.component_objects(pyo.Constraint, active=True):
            # only dual variables of these constraints are important
            if str(c) == 'dual_con1' or str(c) == 'dual_con2':
                dic2 = {}
                for index in c:
                    dic2[index] = model.dual[c[index]]
                    if write:
                        print("\tLambda ", c, index, model.dual[c[index]])
                dic[str(c)] = dic2
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
    return (
        not abs(upper_bound - lower_bound) > epsilon,
        upper_bound,
        lower_bound
    )

def get_loads():
    return [
        8,8,10,10,10,16,22,24,26,32,30,28,22,18,16,16,20,24,28,34,38,30,22,12
    ]
