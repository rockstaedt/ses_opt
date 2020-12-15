import json
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np

import helper

# solver for MIP
opt = SolverFactory('gurobi')

# path for saving files
path = '../3_results/'

# **************************************************************************
# Parameters
# **************************************************************************

# fixed generator costs in $/h
c1 = 2.12*10**-5

# linear generator costs in $/kWh
c2 = 0.128

# maximum capacity of generator
pmax = 12

# electricity price forward contract in $/kWh
l1 = 0.25

# electricity price real time contract in $/kWh
l2s = [0.15, 0.2, 0.25, 0.3, 0.35]

# get load values
LOADS = helper.get_loads()

# set seed for randomness
seed = 17

# get test samples
TEST_SAMPLES = helper.get_monte_carlo_samples(LOADS, samples=500, seed=seed)

APPROACHES = ['Stochastic', 'Deterministic']

objective_values = {
    approach: {str(l2): [] for l2 in l2s} for approach in APPROACHES
}

for approach in APPROACHES:
    helper.print_caption(approach)
    for l2 in l2s:
        helper.print_sens_step(f'Solution for {l2} $/kWh')
        # init results dic
        results_dic = {}
        for i, sample in enumerate(TEST_SAMPLES):
            helper.print_status(i)
            if approach == 'Stochastic':
                # Opening JSON file
                f = open(f'../3_results/results_master_{l2}.json',)
                # returns JSON object as a dictionary
                parameter = json.load(f)
            elif approach == 'Deterministic':
                # Opening JSON file
                f = open(f'../3_results/deterministic_results_master_{l2}.json',)
                # returns JSON object as a dictionary
                parameter = json.load(f)

            # ******************************************************************
            # Model
            # ******************************************************************

            model = pyo.ConcreteModel()

            # ******************************************************************
            # Sets
            # ******************************************************************

            # hour set
            model.H = pyo.RangeSet(0,23)

            # ******************************************************************
            # Variables
            # ******************************************************************

            # fixed master problem variables
            model.u = pyo.Var(model.H)
            model.p1 = pyo.Var(model.H)

            # electricity produced by generator
            model.pg = pyo.Var(model.H, within=pyo.NonNegativeReals)

            # electrictiy bought from retailer
            model.p2 = pyo.Var(model.H, within=pyo.NonNegativeReals)

            # ******************************************************************
            # Objective function
            # ******************************************************************

            model.OBJ = pyo.Objective(
                expr=sum(
                    c1*model.u[h] + l1*model.p1[h]
                    + c2*model.pg[h] + l2*model.p2[h] for h in model.H
                )
            )

            # ******************************************************************
            # Constraints
            # ******************************************************************

            # load must be covered by production or purchasing electrictiy
            # take first random vector from samples
            pl = sample
            def con_load(model, H):
                return model.pg[H] + model.p1[H] + model.p2[H] >= pl[H]
            model.con_load = pyo.Constraint(model.H, rule=con_load)

            # maximum capacity of generator
            def con_max(model, H):
                return model.pg[H] <= pmax*model.u[H]
            model.con_max = pyo.Constraint(model.H, rule=con_max)

            # ensure variable u is equal to the solution of the master problem
            def dual_con1(model, H):
                return model.u[H] == parameter['u'][str(H)]
            model.dual_con1 = pyo.Constraint(model.H, rule=dual_con1)

            # ensure variable p1 is equal to the solution of the master problem
            def dual_con2(model, H):
                return model.p1[H] == parameter['p1'][str(H)]
            model.dual_con2 = pyo.Constraint(model.H, rule=dual_con2)

            # solve model
            helper.solve_model(opt, model)

            results_dic[i] = helper.get_results(model)

            objective_values[approach][str(l2)].append(pyo.value(model.OBJ))

        with open(
            f'{path}testing_{approach}_results_{l2}.json', 'w'
        ) as outfile:
            json.dump(results_dic, outfile)

# save objectives
with open(f'{path}testing_objectives.json', 'w') as outfile:
    json.dump(objective_values, outfile)

print(np.mean(np.array(objective_values['Stochastic']['0.3'])))
print(np.mean(np.array(objective_values['Deterministic']['0.3'])))
