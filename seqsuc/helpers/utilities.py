"""
This file contains several functions that are helpful while setting up and
running the model.
"""

import pyomo.environ as pyo
from pyomo.core import Var
from pyomo.opt import SolverFactory
import os
from pathlib import Path
from typing import Dict

from .printing import print_status


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
    This function returns a dictionary with the results of the model. Hereby,
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
                      samples, params):
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
            results_sub[i]['p2'],
            params
        )
    upper_bound = upper_bound/len(samples)
    lower_bound = master_prob(
        results_master['u'],
        results_master['p1'],
        results_master['alpha'],
        params
    )
    return (
        not upper_bound - lower_bound > params.epsilon,
        upper_bound,
        lower_bound
    )


def get_loads():
    """
    This function returns the 24 load values of the problem. The load vector
    also contains an initialization value of zero.

    Returns:
        [list]: 25 load values including the initialization value
    """
    return [
        0, 8, 8, 10, 10, 10,
        16, 22, 24, 26, 32, 30,
        28, 22, 18, 16, 16, 20,
        24, 28, 34, 38, 30, 22, 12
    ]


def get_path_by_task(mc_sampling: bool, av_sampling: bool, deterministic: bool,
                     sample_size: int, multiprocessing: bool,
                     current_path: Path) -> str:
    """
    Based on the model options, this function returns the corresponding path.

    Returns:
        str: path to result folder of model
    """
    # Determine first level saving path
    if deterministic:
        path = os.path.join(
            current_path.parent,
            '3_results',
            'deterministic',
            'task_1'
        )
    else:
        if multiprocessing:
            path = os.path.join(
                current_path,
                'Final_Report',
                '3_results',
                'stochastic',
                'multiprocessing'
            )
        else:
            path = os.path.join(
                current_path.parent,
                '3_results',
                'stochastic',
                'no_multiprocessing'
            )
        if not mc_sampling and av_sampling:
            path = os.path.join(
                path,
                'task_1_b',
                str(sample_size)
            )
        elif mc_sampling and not av_sampling:
            path = os.path.join(
                path,
                'task_1_c',
                str(sample_size)
            )
        else:
            path = os.path.join(
                path,
                'no_task',
                str(sample_size)
            )

    return path


def solve_sample(sample: list, iterator: int, sample_size: int,
                 model: pyo.ConcreteModel, solver: SolverFactory,
                 obj_value: bool = False, progress_info: bool = False) -> Dict:
    """
    This function solves the model for the given sample, updating the
    corresponding constraint and printing a status to the terminal.

    Args:
        sample (list): List of load values.
        iterator (int): Number of sample.
        sample_size (int): size of samples
        model (pyo.ConcreteModel): Pyomo model to solve.
        solver (pyo.opt.SolverFactory): Solver for Pyomo model.
        obj_value (bool): Include objective value in result dic.
        progress_info (bool): Print progress to terminal.

    Returns:
        Dict: Dictionary of result values.
    """
    if progress_info:
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


def set_load_values(model: pyo.ConcreteModel, load_values: list):
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
    to a dictionary with storage types as keys containing a dictionary of
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
