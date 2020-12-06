import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Var, value

# create concrete pyomo model
model = pyo.ConcreteModel()

model.x_1= pyo.Var()

model.x_2 = pyo.Var()

model.OBJ = pyo.Objective(
    expr=-24*model.x_1-28*model.x_2
)

def con1(model):
    return 6*model.x_1 + 10*model.x_2 <= 2400
model.con1 = pyo.Constraint(rule=con1)

def con2(model):
    return 8*model.x_1 + 5*model.x_2 <= 1600
model.con2 = pyo.Constraint(rule=con2)

def con3(model):
    return (0, model.x_1, 500)
model.con3 = pyo.Constraint(rule=con3)

def con4(model):
    return (0, model.x_2, 100)
model.con3 = pyo.Constraint(rule=con4)

opt = pyo.SolverFactory('gurobi')

results = opt.solve(model)

model.solutions.load_from(results)

results.write()
for v in model.component_objects(Var, active=True):
    print ("Variable", v)
    varobject = getattr(model, str(v))
    for index in varobject:
        print ("\t",index, varobject[index].value)