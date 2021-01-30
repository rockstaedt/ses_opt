import pyomo.environ as pyo
from pyomo.core import Var, value
from pyomo.opt import SolverFactory
import numpy as np
import os
from pathlib import Path
from typing import Dict

from .printing import print_status

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

def get_av_samples(values:list, sample_size=1000, seed=12):
    """
    This function creates a sample using the antithetic variates technique.
    """
    # Set seed
    np.random.seed(seed)
    # Covariance matrix of values. Variance was given.
    cov_pl = np.diagflat(np.array(values)*1/3)
    # Calculate normal distribution for half of the sample size.
    normal_dis = np.random.multivariate_normal(values, cov_pl, size=int(sample_size/2))
    # Calculate athetics of these sample
    antithetics = np.array(values) - (normal_dis - np.array(values))
    # Concatenate both samples to one.
    samples = np.concatenate((normal_dis, antithetics))
    return samples

def get_lhs_sample(sample_size = 1000,loads = LOADS,seed = 12):
    """
    This function creates a vector of samples using the Latin Hypercube technique.
    """

    sample_vectors = []
    np.random.seed(seed)
    for i in loads:
        help_array = []
        mean = i
        std = np.sqrt(i*(1/3))
        mn = mean-std*3     #intervall of 99.73 values
        mx = mean+std*3

        points = np.linspace(mn, mx, num=sample_size+1)
        intervals = np.array([points[:-1], points[1:]]).transpose()

        for j in range(sample_size):
            x = np.random.uniform(intervals[j][0],intervals[j][1],1)[0]
            help_array.append(x)

        np.random.shuffle(help_array)
        sample_vectors.append(help_array)

    sample_vectors = np.array(sample_vectors)

    return  sample_vectors.transpose()

def solve_model(solver, model):
    """
    This function solves a pyomo model with the passed solver.

    Args:
        solver: Pyomo solver factory
        model: Pyomo model
    """
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
            if 'dual_con' in str(c):
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
        not upper_bound - lower_bound > epsilon,
        upper_bound,
        lower_bound
    )

def get_loads():
    """
    This function returns the 24 load values of the problem. The load vector
    also contains a initialization value of zero.

    Returns:
        [list]: 25 load values inluding the initialization value
    """
    return [
        0,8,8,10,10,10,16,22,24,26,32,30,28,22,18,16,16,20,24,28,34,38,30,22,12
    ]

def get_path_by_task(mc_sampling:bool, av_sampling:bool, lhc_sampling:bool,
                     ev:bool, deterministic:bool, sensitivity_analysis:bool,
                     sample_size:int, multiprocessing: bool,
                     current_path:Path) -> str:
    """
    Based on the model options, this function returns the corresponding path.

    Returns:
        str: path to result folder of model
    """
    # Determine first level saving path
    if deterministic:
        if (not ev and not sensitivity_analysis):
            path = os.path.join(
                current_path.parent, '3_results', 'deterministic', 'task_1_4')
        elif (ev and not sensitivity_analysis):
            path = os.path.join(
                current_path.parent,
                '3_results',
                'deterministic',
                'task_5'
            )
        elif (ev and sensitivity_analysis):
            path = os.path.join(
                current_path.parent,
                '3_results',
                'deterministic',
                'task_5',
                'sensitivity_analysis'
            )
        else:
            path = os.path.join(
                current_path.parent, '3_results', 'deterministic', 'no_task')
    else:
        if (not mc_sampling and av_sampling and not lhc_sampling
            and multiprocessing and not sensitivity_analysis):
            path = os.path.join(
                current_path.parent,
                '3_results',
                'stochastic',
                'task_1',
                str(sample_size)
            )
        elif (not mc_sampling and not av_sampling and lhc_sampling
            and multiprocessing and not sensitivity_analysis):
            path = os.path.join(
                current_path.parent,
                '3_results',
                'stochastic',
                'task_2',
                str(sample_size)
            )
        elif (not mc_sampling and av_sampling and not lhc_sampling
            and not multiprocessing and not sensitivity_analysis):
            path = os.path.join(
                current_path.parent,
                '3_results',
                'stochastic',
                'task_4',
                'av_sampling',
                str(sample_size)
            )
        elif (not mc_sampling and not av_sampling and lhc_sampling
            and not multiprocessing and not sensitivity_analysis):
            path = os.path.join(
                current_path.parent,
                '3_results',
                'stochastic',
                'task_4',
                'lhc_sampling',
                str(sample_size)
            )
        elif (not mc_sampling and av_sampling and not lhc_sampling
                and not sensitivity_analysis):
            path = os.path.join(
                current_path.parent,
                '3_results',
                'stochastic',
                'task_5',
                'av_sampling',
                str(sample_size)
            )
        elif (not mc_sampling and av_sampling and not lhc_sampling
                and sensitivity_analysis):
            path = os.path.join(
                current_path.parent,
                '3_results',
                'stochastic',
                'task_5',
                'sensitivity_analysis',
                str(sample_size)
            )
        else:
            path = os.path.join(
                current_path.parent,
                '3_results',
                'stochastic',
                'no_task',
                str(sample_size)
            )

    return path

def solve_sample(sample:list, iterator:int, sample_size:int,
                 model:pyo.ConcreteModel, solver:SolverFactory,
                 obj_value:bool=False) -> Dict:
    """
    This function solves the model for the given sample, updating the
    corresponding constraint and printing a status to the terminal.

    Args:
        sample (list): list of load values
        iterator (int): number of sample
        sample_size (int): size of samples
        model (pyo.ConcreteModel): Pyomo model to solve
        solver (pyo.opt.SolverFactory): solver for Pyomo model
        obj_value (bool): Boolean for including objective value into result dic

    Returns:
        Dict: Dictionary of result values
    """
    print_status(iterator, sample_size)
    # Set new load sample
    set_load_values(model, sample)
    # Update constraint
    model.con_load.reconstruct()
    # Solve model
    solve_model(solver, model)
    # Get results and dual variables of master problem constraints.
    results = get_results(model, dual=True)
    # Add objective value, if specified.
    if obj_value:
        results['objective_value'] = pyo.value(model.OBJ)
    return results

def set_load_values(model:pyo.ConcreteModel, load_values:list):
    """
    This function sets a new load vector for the model variable 'load_values'.

    Args:
        model (pyo.ConcreteModel): Pyomo model
        load_values (list): list of load vectors
    """
    for i, load_value in enumerate(load_values):
        model.load_values[i] = load_value

def reset_tuple_key(dic: Dict) -> Dict:
    """
    This function resets a dictionary with a tuple key (storage_type, hour)
    to a dicationary with storage types as keys containing a dictionary of
    hours.
    """
    return_dic = {}
    for key_one, key_two in list(dic.keys()):
        if key_one in return_dic:
            return_dic[key_one][key_two] = dic[(key_one, key_two)]
        else:
            return_dic[key_one] = {}
            return_dic[key_one][key_two] = dic[(key_one, key_two)]
    return return_dic