"""
This file contains all functions to create a two stage stochastic unit
commitment problem. It is divided into a master (first stage) and sub problem
(second stage). In addition, this file provides a function to create a test
problem which can be used to test an optimal solution.
"""

import pyomo.environ as pyo

from .parameters import Parameter


def create_master_problem(params: Parameter) -> pyo.ConcreteModel:
    """
    This function creates a master problem.
    :param params: Parameter object containing all relevant parameters.
    :return: master problem
    """

    master = pyo.ConcreteModel()

    # Save parameters as a model instance to access parameters in constraint
    # functions.
    master.params = params

    # Hour sets
    master.H = pyo.RangeSet(1, len(params.HOURS) - 1)
    master.H_all = pyo.RangeSet(0, len(params.HOURS) - 1)

    # Variables
    # Unit commitment for generator
    master.u = pyo.Var(master.H_all, within=pyo.Binary)

    # Initialization for u
    master.u[0].fix(0)

    # Electricity purchased with the forward contract
    master.p1 = pyo.Var(master.H_all, within=pyo.NonNegativeReals)

    # Initialization for p1
    master.p1[0].fix(0)

    # Value function for second stage problem
    master.alpha = pyo.Var(master.H_all)

    # Initialization for p1
    master.alpha[0].fix(0)

    def master_obj(model):
        return sum(
            params.c1 * model.u[h] + params.l1 * model.p1[h]
            + model.alpha[h] for h in model.H
        )

    master.OBJ = pyo.Objective(rule=master_obj)

    # Constraints
    # Alpha down (-500) is an arbitrary selected bound.
    def alphacon1(model, hour):
        return model.alpha[hour] >= -500

    master.alphacon1 = pyo.Constraint(master.H, rule=alphacon1)

    master.min_uptime = pyo.Constraint(master.H, rule=__min_uptime)

    master.min_downtime = pyo.Constraint(master.H, rule=__min_downtime)

    return master


def create_sub_problem(params: Parameter,
                       results_master=None) -> pyo.ConcreteModel:
    """
    This function creates a sub problem.
    :param params: Parameter object containing all relevant parameters.
    :param results_master: First stage variables. Defaults to None.
    :return: sub problem
    """

    sub = pyo.ConcreteModel()

    # Save parameters as a model instance to access parameters in constraint
    # functions.
    sub.params = params

    # Save passed results of master as a model instance to update them later.
    sub.results_master = results_master

    # Hour sets
    sub.H = pyo.RangeSet(1, len(params.HOURS) - 1)
    sub.H_all = pyo.RangeSet(0, len(params.HOURS) - 1)

    # Set of energy storage resources.
    sub.ESR = pyo.Set(initialize=params.ESRS)

    # Create a parameter for the load values.
    sub.load_values = pyo.Param(sub.H_all, mutable=True)

    # First stage variables
    # No need for declaration of variable types because that is determined by
    # corresponding variables of the master problem.
    sub.u = pyo.Var(sub.H_all)
    sub.p1 = pyo.Var(sub.H_all)

    # Second stage variables
    # Electricity produced by generator
    sub.pg = pyo.Var(sub.H_all, within=pyo.NonNegativeReals)
    # Initialization of pg
    sub.pg[0].fix(0)

    # Electricity bought from retailer
    sub.p2 = pyo.Var(sub.H_all, within=pyo.NonNegativeReals)
    # Initialization of pg
    sub.p2[0].fix(0)

    # Net injection by storage with bounds of maximum charge and discharge
    # power.
    sub.stor_net_i = pyo.Var(sub.ESR, sub.H_all, bounds=__get_net_bounds)

    # Storage level in kWh
    sub.stor_level = pyo.Var(
        sub.ESR,
        sub.H_all,
        within=pyo.NonNegativeReals,
        bounds=__get_stor_levels
    )
    # Initialization of storage level and net injections for all ESR.
    for esr_type in sub.ESR:
        if 'ev' in esr_type:
            for h in sub.H_all:
                if h < params.plug_in_hour or h > params.plug_out_hour:
                    # Fix stor level and net injection values to zero before ev
                    # is plugged in and after ev is plugged out.
                    sub.stor_level[esr_type, h].fix(0)
                    sub.stor_net_i[esr_type, h].fix(0)
            sub.stor_level[esr_type, params.plug_out_hour].fix(
                params.charge_target * params.esr_to_stor_level_max[esr_type]
            )
        else:
            sub.stor_level[esr_type, 0].fix(
                params.esr_to_stor_level_zero[esr_type]
            )

    # Objective function
    sub.OBJ = pyo.Objective(
        expr=sum(
            params.c2 * sub.pg[h] + params.l2 * sub.p2[h] for h in sub.H_all)
    )

    # Constraints
    sub.con_load = pyo.Constraint(sub.H, rule=__con_load)

    sub.con_max = pyo.Constraint(sub.H, rule=__con_max)

    # Ensure variable u is equal to the solution of the master problem.
    def dual_con1(model, hour):
        return model.u[hour] == model.results_master['u'][hour]
    sub.dual_con1 = pyo.Constraint(sub.H_all, rule=dual_con1)

    # Ensure variable p1 is equal to the solution of the master problem.
    def dual_con2(model, hour):
        return model.p1[hour] == model.results_master['p1'][hour]
    sub.dual_con2 = pyo.Constraint(sub.H_all, rule=dual_con2)

    sub.con_ramping = pyo.Constraint(sub.H, rule=__con_ramping)

    sub.stor_balance = pyo.Constraint(sub.ESR, sub.H, rule=__stor_balance)

    return sub


def create_test_problem(params: Parameter,
                        first_stage_variables: dict) -> pyo.ConcreteModel:
    """
    This function creates a problem to test first stage variables.
    :param params: Parameter object containing all relevant parameters.
    :param first_stage_variables: Dictionary of first stage variables.
    :return: test problem
    """

    model = pyo.ConcreteModel()

    # Save parameters as a model instance to access parameters in constraint
    # functions.
    model.params = params

    # Hour sets
    model.H = pyo.RangeSet(1, len(params.HOURS) - 1)
    model.H_all = pyo.RangeSet(0, len(params.HOURS) - 1)

    # Set of energy storage resources.
    model.ESR = pyo.Set(initialize=params.ESRS)

    # Create a parameter for the load values.
    model.load_values = pyo.Param(model.H_all, mutable=True)

    # Fixed model problem variables
    model.u = pyo.Var(model.H_all)
    model.p1 = pyo.Var(model.H_all)
    for h in model.H_all:
        model.u[h].fix(first_stage_variables['u'][h])
        model.p1[h].fix(first_stage_variables['p1'][h])

    # Electricity produced by generator
    model.pg = pyo.Var(model.H_all, within=pyo.NonNegativeReals)
    # Initialization for p1
    model.pg[0].fix(0)

    # Electricity bought from retailer
    model.p2 = pyo.Var(model.H_all, within=pyo.NonNegativeReals)
    # Initialization for p1
    model.p2[0].fix(0)

    # Net injection by storage with bounds of maximum charge and discharge.
    model.stor_net_i = pyo.Var(
        model.ESR,
        model.H_all,
        bounds=__get_net_bounds
    )

    # Storage level in kWh
    model.stor_level = pyo.Var(
        model.ESR,
        model.H_all,
        within=pyo.NonNegativeReals,
        bounds=__get_stor_levels
    )
    # Initialization of storage level and net injections for all ESR.
    for esr_type in model.ESR:
        if 'ev' in esr_type:
            for h in model.H_all:
                if h < params.plug_in_hour or h > params.plug_out_hour:
                    # Fix stor level and net injection values to zero before ev
                    # is plugged in and after ev is plugged out.
                    model.stor_level[esr_type, h].fix(0)
                    model.stor_net_i[esr_type, h].fix(0)
            model.stor_level[esr_type, params.plug_out_hour].fix(
                params.charge_target * params.esr_to_stor_level_max[esr_type]
            )
        else:
            model.stor_level[esr_type, 0].fix(
                params.esr_to_stor_level_zero[esr_type]
            )

    # Objective function
    model.OBJ = pyo.Objective(
        expr=sum(
            params.c1 * model.u[h] + params.l1 * model.p1[h]
            + params.c2 * model.pg[h] + params.l2 * model.p2[h] for h in model.H
        )
    )

    # Constraints
    model.con_load = pyo.Constraint(model.H, rule=__con_load)

    model.con_max = pyo.Constraint(model.H, rule=__con_max)

    model.con_ramping = pyo.Constraint(model.H, rule=__con_ramping)

    model.stor_balance = pyo.Constraint(model.ESR, model.H, rule=__stor_balance)

    return model


def __stor_balance(model, ESR, H):
    """
    Storage balance constraint.
    """
    if 'ev' not in ESR:
        return model.stor_level[ESR, H] == (
                model.stor_level[ESR, H - 1] - model.stor_net_i[ESR, H])
    else:
        if model.params.plug_in_hour < H <= model.params.plug_out_hour:
            return model.stor_level[ESR, H] == (
                    model.stor_level[ESR, H - 1] - model.stor_net_i[ESR, H]
            )
        if H == model.params.plug_in_hour:
            return model.stor_level[ESR, H] == (
                model.params.esr_to_stor_level_zero[ESR] - model.stor_net_i[
                    ESR, H
                ]
            )
        else:
            return pyo.Constraint.Skip


# Functions for pyomo constraints.

def __min_downtime(model, h):
    """
    Minimum downtime constraint
    """
    # Apply minimum downtime constraint.
    # The end value of the range function needs to be increased to
    # be included.
    vs = list(
        range(h, min([h - 1 + model.params.downtime, len(model.H_all) - 1]) + 1)
    )
    # For the last hour, the output of the range function is 0
    # because range(24,24). To include hour 24 into the list,
    # check for length and put hour 24 into V.
    if len(vs) == 0:
        vs = [h]
    # Return the sum of all hours in V to apply the constraint for
    # all hours in V.
    return sum(
        model.u[h - 1] - model.u[h] for _ in vs
    ) <= sum(1 - model.u[v] for v in vs)


def __min_uptime(model, h):
    """
    Minimum uptime constraint
    """
    # Apply minimum uptime constraint.
    # The end value of the range function needs to be increased to
    # be included.
    vs = list(
        range(h, min([h - 1 + model.params.uptime, len(model.H_all) - 1]) + 1)
    )
    # For the last hour, the output of the range function is 0
    # because range(24,24). To include hour 24 into the list,
    # check for length and put hour 24 into V.
    if len(vs) == 0:
        vs = [h]
    # Return the sum of all hours in V to apply the constraint for
    # all hours in V.
    return sum(
        model.u[h] - model.u[h - 1] for _ in vs
    ) <= sum(model.u[v] for v in vs)


def __get_net_bounds(model, esr, _):
    """
    Maximum charge and discharge power of ESRs
    """
    return -model.params.esr_to_p_w_max[esr], model.params.esr_to_p_i_max[esr]


def __get_stor_levels(model, esr, H):
    """
    Max storage level bounds for ESRs
    """
    if 'ev' not in esr:
        return 0, model.params.esr_to_stor_level_max[esr]
    else:
        if H < model.params.plug_in_hour or H > model.params.plug_out_hour:
            return 0, 0
        else:
            return (
                model.params.min_soc * model.params.esr_to_stor_level_max[esr],
                model.params.max_soc * model.params.esr_to_stor_level_max[esr]
            )


def __con_load(model, h):
    """
    Load must be covered by production, purchasing electricity or by the
    sum of net injections over all ESRs.
    """
    return (model.pg[h] + model.p1[h] + model.p2[h]
            + sum(model.stor_net_i[esr, h] for esr in model.ESR)
            ) >= model.load_values[h]


def __con_max(model, h):
    """
    Maximum capacity of generator
    """
    return model.pg[h] <= model.params.pmax * model.u[h]


def __con_ramping(model, h):
    """
    Ramping constraint of generator
    """
    return (
        -model.params.ramping_constraint,
        model.pg[h] - model.pg[h - 1],
        model.params.ramping_constraint
    )
