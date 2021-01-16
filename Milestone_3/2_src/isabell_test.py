import concurrent.futures
import time
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from parameters import *
from model_options import *
from modeling_helper.utilities import *
from modeling_helper.printing import *


def solve_all_samples(sample, iterator, sample_size, model, solver):
    print_status(iterator, sample_size)
    # Filter for first sample because that is set in the initialization of
    # the model.
    if iterator != 0:
        # Set new load sample
        set_load_values(model, sample)
        # Update constraint
        model.con_load.reconstruct()
    # solve model
    results = solve_model(opt, model)
    return results

def set_load_values(model, load_values):
    for i, load_value in enumerate(load_values):
        model.load_values[i] = load_value

# Start time
start = time.perf_counter()

# Seet for randomness
seed = 12

# For the deterministic approach, the normal load vector with its mean values
# is used. For the stochastic approach, a monte carlo sample is created.
if deterministic:
    SAMPLES = np.array([LOADS])
    sample_size = 1
else:
    SAMPLES = get_monte_carlo_samples(
        LOADS,
        samples=sample_size,
        seed=seed
    )

# Simple model
opt = pyo.SolverFactory('gurobi')

model = pyo.ConcreteModel()

model.H = pyo.RangeSet(0, len(HOURS)-1)

model.load_values = pyo.Param(model.H, default=SAMPLES[0, :], mutable=True)

# Unit commitment for generator
model.u = pyo.Var(model.H, within=pyo.Binary)

# Electricity produced by generator
model.pg = pyo.Var(model.H, within=pyo.NonNegativeReals)

# Electricity purchased with the forward contract
model.p1 = pyo.Var(model.H, within=pyo.NonNegativeReals)

model.OBJ = pyo.Objective(
    expr=sum(c2*model.pg[h] +l1*model.p1[h] for h in model.H)
)

# Load must be covered by production or purchasing electrictiy.
pl = SAMPLES[0, :]
def con_load(model, H):
    return model.pg[H] + model.p1[H] >= model.load_values[H]
model.con_load = pyo.Constraint(model.H, rule=con_load)

# maximum capacity of generator
def con_max(model, H):
    return model.pg[H] <= pmax*model.u[H]
model.con_max = pyo.Constraint(model.H, rule=con_max)

# results = map(
#     solve_all_samples,
#     SAMPLES,
#     list(range(len(SAMPLES))),
#     [len(SAMPLES)] * len(SAMPLES),
#     [model] * len(SAMPLES),
#     [opt] * len(SAMPLES)
# )

# for i, result in enumerate(results):
#     if i < 3:
#         print(i)
#         print(result)

# results = {}
# for i, sample in enumerate(SAMPLES):
#     print_status(i, sample_size)
#     # Filter for first sample because that is set in the initialization of
#     # the model.
#     if i != 0:
#         # set new load sample
#         set_load_values(model, sample)
#         # update constraint
#         model.con_load.reconstruct()
#     # solve model
#     solve_model(opt, model)
#     results[i] = get_results(model)

# def do_something(i, seconds, solver):
#     print(f'Sleeping {seconds} second(s) and {i} iterations...')
#     print(f'Solved with {solver} ')
#     time.sleep(seconds)
#     return f'Done Sleeping...{seconds}'

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(
        solve_all_samples,
        SAMPLES,
        list(range(len(SAMPLES))),
        [len(SAMPLES)] * len(SAMPLES),
        [model] * len(SAMPLES),
        [opt] * len(SAMPLES)
    )

for i, result in enumerate(results):
    if i < 3:
        print(i)
        print(result)

    # for result in results:
    #     print(result)

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
