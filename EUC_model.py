import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Var
import numpy as np

###############################################################################
### Model Options
###############################################################################

static_lambda = True
#test commit
###############################################################################
### Parameters
###############################################################################


# retail electricity price in €/kWh
if static_lambda:
    lamda = 21.97/100

# load values for 24 hours
pl = [8,8,10,10,10,16,22,24,26,32,30,28,22,18,16,16,20,24,28,34,38,30,22,12]

# fuel cost parameters
c2 = np.array([1.2,1.12])*0.001
c1 = np.array([0.128,0.532])
c  = np.array([2.12,12.8])*0.0001

price_generators_kwh = [c2[g]*1**2 + c1[g]*1 + c[g] for g in range(0,2)]

# min max power values
pmin = np.array([0,0])
pmax = np.array([20,40])


###############################################################################
### Model
###############################################################################


# create concrete pyomo model
model = pyo.ConcreteModel()

#------------------------------------------------------------------------------
# Sets
#------------------------------------------------------------------------------

# hourly set for 24 hours
# (not from 1 to 24 because we use a python list for the load values)
model.H = pyo.RangeSet(0,23)

# generator set for the two generators
model.G = pyo.RangeSet(0,1)

#------------------------------------------------------------------------------
# Variables
#------------------------------------------------------------------------------

# net power needed from external network per hour
model.pn = pyo.Var(model.H)

# power generation of each generator per hour, non negativity constraint
model.pg = pyo.Var(model.H, model.G,within=pyo.NonNegativeReals)

# binary unit commitment variable for each generator and hour
model.u = pyo.Var(model.H, model.G, within=pyo.Binary)

# helper variable to prevent quadratic solver problem
model.y = pyo.Var(model.H, model.G, within=pyo.NonNegativeReals)

#------------------------------------------------------------------------------
# Objective Function
#------------------------------------------------------------------------------

# first part is net power cost with distribution company,
# second part is fuel costs
model.OBJ = pyo.Objective(
    expr=sum(lamda*model.pn[h] for h in model.H)
        + sum(
            (c2[g]*model.y[h,g] + c1[g]*model.pg[h,g] + c[g])*model.u[h,g]
            for h in model.H for g in model.G
        )
)

#------------------------------------------------------------------------------
# Constraints
#------------------------------------------------------------------------------

# load for each hour
def loadc(model, H):
    return sum(model.pg[H,g] for g in model.G) + model.pn[H] == pl[H]
model.loadc = pyo.Constraint(model.H, rule=loadc)

# minimum generation for each used generator and hour
def minc(model, H, G):
    return model.u[H,G]*pmin[G] <= model.pg[H,G]
model.minc = pyo.Constraint(model.H, model.G, rule=minc)
model.minc.pprint()

# maximum generation for each used generator and hour
def maxc(model, H, G):
    return model.u[H,G]*pmax[G] >= model.pg[H,G]
model.maxc = pyo.Constraint(model.H,model.G, rule=maxc)

# constraint for helper variable
def hvc(model, H, G):
    return model.y[H,G] == model.pg[H,G]**2
model.hvc = pyo.Constraint(model.H, model.G, rule=hvc)

#------------------------------------------------------------------------------
# Results
#------------------------------------------------------------------------------

opt = pyo.SolverFactory('gurobi')
opt.options['NonConvex'] = 2

results = opt.solve(model, tee=True)
results.write()

model.solutions.load_from(results)

for v in model.component_objects(Var, active=True):
    print ("Variable", v)
    varobject = getattr(model, str(v))
    for index in varobject:
        print ("\t",index, varobject[index].value)