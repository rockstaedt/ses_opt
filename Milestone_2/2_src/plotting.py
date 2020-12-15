import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.lines import Line2D
import json

###############################################################################
### Matplotlib Settings
###############################################################################

settings = {
    'text.usetex': True,
    'font.weight' : 'normal',
    'font.size'   : 14
}
plt.rcParams.update(**settings)

# resolution for plots
dpi = 300

# path to csv files
path = '../3_results/'

# saving path
saving_path = '../4_plots/'

###############################################################################
### Plots
###############################################################################

# Opening JSON file
f = open(f'{path}results_master_0.3.json',)
g = open(f'{path}testing_Stochastic_results_0.3.json')

# returns JSON object as
# a dictionary
master_data = json.load(f)
sub_data = json.load(g)

DIK = master_data['p1']

hours = []
forwards = []

#My attempt:
for key, value in DIK.items():
    aKey = key
    aValue = value
    hours.append(aKey)
    forwards.append(aValue)
    #dictList.append(temp)
    #aKey = ""
    #aValue = ""

# One Plot

plt.figure(figsize =(12,7))
plt.plot(hours, forwards, label = "Forward Electricity Contracted",linewidth=4)

plt.legend(fontsize = 14)
#plt.yscale("log")
plt.xlabel("hours", fontsize=18)
plt.ylabel("kW", fontsize=18)
plt.grid(axis='both')
plt.title("Stochastic Model - Forward Electricity Purchased", fontsize = 18)
plt.show()

list(sub_data['0']['pg'].values())

#Pg generated plot

plt.figure(figsize =(12,7))
pg_array = []
for i in range(500):
    plt.plot(hours,list(sub_data[str(i)]['pg'].values()),linewidth=0.5,color = 'mediumblue',alpha = 0.2)
    pg_array.append(list(sub_data[str(i)]['pg'].values()))

pg_array = np.array(pg_array)
pg_array = np.mean(pg_array,axis=0)
plt.plot(hours,pg_array,linewidth = 3,color='forestgreen')

plt.xlabel("Hours", fontsize=18)
plt.ylabel("kW", fontsize=18)

plt.title("Stochastic Model - Power Generated", fontsize = 18)

plt.savefig(
    f'{saving_path}stochastic_model_pg.png',
    dpi=dpi,
    bbox_inches='tight'
)

plt.show

#P2 plot

plt.figure(figsize =(12,7))
p2_array = []
for i in range(500):
    plt.plot(hours,list(sub_data[str(i)]['p2'].values()),linewidth=0.5,color = 'mediumblue',alpha = 0.2)
    p2_array.append(list(sub_data[str(i)]['p2'].values()))

p2_array = np.array(p2_array)
p2_array = np.mean(p2_array,axis=0)
plt.plot(hours,p2_array,linewidth = 3,color='forestgreen')

plt.xlabel("Hours", fontsize=18)
plt.ylabel("kW", fontsize=18)

plt.title("Stochastic Model - Realtime Power Purchase", fontsize = 18)

plt.savefig(
    f'{saving_path}stochastic_model_p2.png',
    dpi=dpi,
    bbox_inches='tight'
)

plt.show

# Opening JSON file
ff = open(f'{path}deterministic_results_master_0.3.json',)
gg = open(f'{path}deterministic_results_sub_0.3.json')

# returns JSON object as
# a dictionary
master_data_D = json.load(ff)
sub_data_D = json.load(gg)

#Deterministic Cost
up_cost_D = 2.12*10**(-5)*24
fw_cost_D = sum(list(sub_data_D['p1'].values()))*0.25
re_cost_D = sum(list(sub_data_D['p2'].values()))*0.3
pg_cost_D = sum(list(sub_data_D['pg'].values()))*0.128

total_D = up_cost_D+fw_cost_D+re_cost_D+pg_cost_D

#Stochastic Cost

up_cost_S = 2.12*10**(-5)*24
fw_cost_S = sum(list(master_data['p1'].values()))*0.25
re_cost_S = sum(p2_array)*0.3
pg_cost_S = sum(pg_array)*0.128

total_S = up_cost_S+fw_cost_S+re_cost_S+pg_cost_S
total_S

ind = ['Deterministic','Stochastic']

upc = (up_cost_D,up_cost_S)
fwc = (fw_cost_D,fw_cost_S)
rec = (re_cost_D,re_cost_S)
pgc = (pg_cost_D,pg_cost_S)
width = 0.25

plt.figure(figsize =(12,7))

#plt.bar(ind, upc, width,color='blue',edgecolor='white')

p1 = plt.bar(ind, rec, width,color='limegreen',edgecolor='white',alpha=0.8)
p2 = plt.bar(ind, pgc, width, bottom=rec,color='darkcyan',edgecolor='white',alpha=0.8)
p3 = plt.bar(ind, fwc, width, bottom=pgc,color='mediumblue',edgecolor='white')

plt.ylabel('\$', fontsize=18)
plt.title('Total Cost', fontsize=18)
#plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0],p3[0]), ('Realtime Power Costs', 'Power Generation Costs','Forward Power Costs'), fontsize=18)

plt.savefig(
    f'{saving_path}total_costs.png',
    dpi=dpi,
    bbox_inches='tight'
)

plt.show()


width = 0.35
plt.figure(figsize =(12,9))
iterator = 0
for i in ['0.15','0.2','0.25','0.3','0.35']:
    ind = [i]
    i_float = float(i)
    ggg = open(f'{path}testing_Stochastic_results_'+i+'.json')

    # returns JSON object as
    # a dictionary
    sub_data_i = json.load(ggg)


    for sample, var in sub_data_i.items():
         pg = 0
         p1 = 0
         p2 = 0
         for hour, value in sub_data_i[sample][var].items():
              pg += sub_data_i[sample][var][hour]


    p2_array = []
    pg_array = []
    p1_array = []

    pg_array = np.array(pg_array)
    pg_array = np.mean(pg_array,axis=0)

    p2_array = np.array(p2_array)
    p2_array = np.mean(p2_array,axis=0)

    p1_array = np.array(p1_array)
    p1_array = np.mean(p1_array,axis=0)


    fw_cost_i = sum(p1_array)*0.25
    re_cost_i = sum(p2_array)*i_float
    pg_cost_i = sum(pg_array)*0.128

    #,edgecolor='white'

    p1 = plt.bar(ind,re_cost_i, width,color='limegreen',alpha=0.8)
    p2 = plt.bar(ind,pg_cost_i, width, bottom=re_cost_i,color='darkcyan',alpha=0.8)
    p3 = plt.bar(ind,fw_cost_i, width, bottom=pg_cost_i,color='mediumblue')

    plt.legend((p1[0], p2[0],p3[0]), ('Realtime Power Costs', 'Power Generation Costs','Forward Power Costs'), fontsize=13)
    iterator +=1

plt.ylabel('\$', fontsize=18)
plt.xlabel('Realtime Power Price', fontsize=18)
plt.title('Sensitivity Analysis - Total Costs', fontsize=18)

plt.savefig(
    f'{saving_path}sensitivity_anaylsis.png',
    dpi=dpi,
    bbox_inches='tight'
)

plt.show

i = '0.3'
i_float = float(i)
ffff = open(f'{path}results_master_'+i+'.json',)
gggg = open(f'{path}results_sub_'+i+'.json')

# returns JSON object as
# a dictionary
master_data_i = json.load(ffff)
sub_data_i = json.load(gggg)

p2_array = []
pg_array = []
for i in range(1000):
    p2_array.append(list(sub_data_i[str(i)]['p2'].values()))
    pg_array.append(list(sub_data_i[str(i)]['pg'].values()))

pg_array = np.array(pg_array)
pg_array = np.mean(pg_array,axis=0)

p2_array = np.array(p2_array)
p2_array = np.mean(p2_array,axis=0)


fw_cost_i = sum(list(master_data_i['p1'].values()))*0.25
re_cost_i = sum(p2_array)*i_float
pg_cost_i = sum(pg_array)*0.128

objective_values = np.loadtxt(
     f'{path}sensitivity analysis/objective_values_sensitivity_0.3.csv',
     delimiter=','
)

bounds_differences = np.loadtxt(
     f'{path}sensitivity analysis/bounds_differences_sensitivity_0.3.csv',
     delimiter=','
)

# One Plot
iterations = range(1, len(objective_values)+1)
plt.figure(figsize =(12,7))
plt.plot(iterations, bounds_differences, label = "Bounds Difference",linewidth=4)
plt.plot(iterations, objective_values, label = "Objective Value",linewidth=4)
plt.legend(fontsize = 14)
plt.yscale("log")
plt.xlabel("Iterations", fontsize=18)
plt.ylabel("\$", fontsize=18)
plt.grid(axis='both')
plt.title("Performance - Objective Value and Bound Difference", fontsize = 18)

plt.savefig(
    f'{saving_path}performance_stochastic.png',
    dpi=dpi,
    bbox_inches='tight'
)

plt.show()


# Opening JSON file
f = open('../3_results/testing_objectives.json',)

# returns JSON object as
# a dictionary
data = json.load(f)

for key, value in data['Stochastic'].items():
     print(key)
     mean = np.mean(np.array(value))
     print(mean)


p1 = plt.bar('0.15', 69.57, width, color='forestgreen',alpha=0.8)
p1 = plt.bar('0.20', 81.13, width, color='forestgreen',alpha=0.8)
p1 = plt.bar('0.25', 92.69, width, color='forestgreen',alpha=0.8)
p1 = plt.bar('0.30', 95.74, width, color='forestgreen',alpha=0.8)
p1 = plt.bar('0.35', 97.3, width, color='forestgreen',alpha=0.8)

plt.ylabel('\$', fontsize=18)
plt.xlabel('Realtime Power Price', fontsize=18)
plt.title('Sensitivity Analysis - Total Costs', fontsize=18)

plt.savefig(
    f'{saving_path}sensitivity_anaylsis.png',
    dpi=dpi,
    bbox_inches='tight'
)

## ANSWER QUESTION FROM PRESENTATION REGARDING HIGHER REAL TIME COSTS IN
## STOCHASTIC MODEL

f = open('../3_results/testing_Deterministic_results_0.3.json',)

# returns JSON object as
# a dictionary
data = json.load(f)

pg_list = []
p1_list = []
p2_list = []

for sample, dic_var in data.items():
    pg = 0
    p1 = 0
    p2 = 0
    for var, dic_values in dic_var.items():
        if var == 'pg':
            for hour, value in dic_values.items():
                pg += float(value)
        if var == 'p1':
            for hour, value in dic_values.items():
                p1 += float(value)
        if var == 'p2':
            for hour, value in dic_values.items():
                p2 += float(value)
    pg_list.append(pg)
    p1_list.append(p1)
    p2_list.append(p2)

print('DETERMINISTIC')

mean_p1_deter = np.mean(np.array(p1_list))
print(mean_p1_deter)

mean_pg_deter = np.mean(np.array(pg_list))
print(mean_pg_deter)

mean_p2_deter = np.mean(np.array(p2_list))
print(mean_p2_deter)

print('##########################')

f = open('../3_results/testing_Stochastic_results_0.3.json',)

# returns JSON object as
# a dictionary
data = json.load(f)

pg_list = []
p1_list = []
p2_list = []

for sample, dic_var in data.items():
    pg = 0
    p1 = 0
    p2 = 0
    for var, dic_values in dic_var.items():
        if var == 'pg':
            for hour, value in dic_values.items():
                pg += float(value)
        if var == 'p1':
            for hour, value in dic_values.items():
                p1 += float(value)
        if var == 'p2':
            for hour, value in dic_values.items():
                p2 += float(value)
    pg_list.append(pg)
    p1_list.append(p1)
    p2_list.append(p2)

print('STOCHASTIC')

mean_p1_stoch = np.mean(np.array(p1_list))
print(mean_p1_stoch)

mean_pg_stoch = np.mean(np.array(pg_list))
print(mean_pg_stoch)

mean_p2_stoch = np.mean(np.array(p2_list))
print(mean_p2_stoch)