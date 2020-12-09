import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Var, value

# create concrete pyomo model
model = pyo.ConcreteModel()

model.y = pyo.Var(within=pyo.NonNegativeReals)

model.x = pyo.Var()

model.OBJ = pyo.Objective(
    expr=-model.y
)

def con1(model):
    return model.y - model.x <= 5
model.con1 = pyo.Constraint(rule=con1)

def con2(model):
    return model.y - 0.5*model.x <= 15/2
model.con2 = pyo.Constraint(rule=con2)

def con3(model):
    return model.y + 0.5*model.x <= 35/2
model.con3 = pyo.Constraint(rule=con3)

def con4(model):
    return -model.y + model.x <= 10
model.con4 = pyo.Constraint(rule=con4)

def con5(model):
    return model.x == 0
model.con5 = pyo.Constraint(rule=con5)

model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

opt = pyo.SolverFactory('gurobi')

results = opt.solve(model)

#model.solutions.load_from(results)

results.write()
for v in model.component_objects(Var, active=True):
    print ("Variable", v)
    varobject = getattr(model, str(v))
    for index in varobject:
        print ("\t",index, varobject[index].value)

# display all duals
print ("Duals")
for c in model.component_objects(pyo.Constraint, active=True):
    print ("   Constraint",c)
    for index in c:
        print ("      ", index, model.dual[c[index]])