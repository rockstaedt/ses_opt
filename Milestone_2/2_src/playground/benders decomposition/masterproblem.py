import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Var, value

# create concrete pyomo model
model = pyo.ConcreteModel()

model.x = pyo.Var()

model.alpha = pyo.Var()

model.OBJ = pyo.Objective(
    expr=-1/4*model.x + model.alpha
)

def con1(model):
    return -19/2 + 1/2*(model.x - 16) <= model.alpha
model.con1 = pyo.Constraint(rule=con1)

def con2(model):
    return (0, model.x, 16)
model.con2 = pyo.Constraint(rule=con2)

def con3(model):
    return -25 <= model.alpha
model.con3 = pyo.Constraint(rule=con3)

def con4(model):
    return -5 - 1*(model.x - 0) <= model.alpha
model.con4 = pyo.Constraint(rule=con4)

opt = pyo.SolverFactory('gurobi')

results = opt.solve(model)

model.solutions.load_from(results)

results.write()
for v in model.component_objects(Var, active=True):
    print ("Variable", v)
    varobject = getattr(model, str(v))
    for index in varobject:
        print ("\t",index, varobject[index].value)