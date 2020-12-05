from pyomo.core import *
import numpy as np

# Model
model = AbstractModel()

#Sets
model.H = Set()
model.G = Set()

#Parameters
model.c1 = Param(model.G,within=PositiveReals)
model.lmda1 = Param(within=PositiveReals)

model.c2 = Param(model.G,within=PositiveReals)
model.lmda2 = Param(within=PositiveReals)
model.pmax = Param(model.G, within=PositiveReals)

model.pl = Param(model.H,within =NonNegativeReals )

#Vars
model.u = Var(model.H, model.G, within=Binary)               #binary up/down indicator
model.p1 = Var(model.H, model.G, within=NonNegativeReals)    #forward electricity

model.p2 = Var(model.H, model.G, within=NonNegativeReals)    #real time electricity
model.pg = Var(model.H, model.G, within=NonNegativeReals)    #produced electricity

#constraints- all for the second stage, not sure if i have to connect them to second stage

# minimum generation for each used generator and hour
def minc(model, H, G):
        return 0 <= model.pg[H,G]
model.minc = Constraint(model.H, model.G, rule=minc)

# maximum generation for each used generator and hour
def maxc(model, H, G):
    return model.u[H,G]*pmax[G] >= model.pg[H,G]
model.maxc = Constraint(model.H,model.G, rule=maxc)

# load for each hour
def loadc(model, H):
    return sum(model.pg[H,g] for g in model.G) + model.p1[H] +model.p2[H] == pl[H]
model.loadc = Constraint(model.H, rule=loadc)


# Stage-specific cost computations

def ComputeFirstStageCost_rule(model,H,G):
    return summation(model.c1[G]*model.u[H,G], model.lmda1*model.p1[H])

model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

def ComputeSecondStageCost_rule(model):
    return summation(model.c2[G]*model.pg[H,G], model.lmda2*model.p2[H])

model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)