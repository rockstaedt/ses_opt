###############################################################################
### Paremeters
###############################################################################

from modeling_helper.utilities import *
from model_options import sensitivity_analysis, ev

# Size of monte carlo sample
sample_size = 10

# Size of test sample size
test_size = 5

# Fixed generator costs in $/h
c1 = 2.12*10**-5

# Linear generator costs in $/kWh
c2 = 0.128

# Maximum capacity of generator
pmax = 12

# Minimum uptime of generator in hours
uptime = 3

# Minimum downtime of generator in hours
downtime = 4

# Ramping constraint of generator in kW
ramping_constraint = 5

# Electricity price forward contract in $/kWh
l1 = 0.25

# Electricity price real time contract in $/kWh
l2 = 0.3

# Storage types
if ev:
    esr_types = ['battery', 'ev']
else:
    esr_types = ['battery']

# Maximum charging power of storage in kW
esr_to_p_w_max = {
    'battery': 10,
    'ev': 11
}

# Maxium discharging power of storage in kW
esr_to_p_i_max = {
    'battery': 10,
    'ev': 11
}

# Maximum storage level in kWh
esr_to_stor_level_max = {
    'battery': 5,
    'ev': 38
}

# Initial storage level
esr_to_stor_level_zero = {
    'battery': 0,
    'ev': 0.2*esr_to_stor_level_max['ev']
}

# Charge target of storages
# This parameter is part of a sensitivity analysis from 20% to 100% in steps
# of 10%. Therefore the parameter is defined as a list in case of a sensitivity
# analysis. If no sensitivity analysis is performed, the intial value of 80%
# is used.
if sensitivity_analysis:
    charge_targets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
else:
    charge_targets = [0.8]

# Load values in kW
LOADS = get_loads()

# Hours
HOURS = list(range(0, len(LOADS)))

# Arbitrary value for convergence check
epsilon = 0.0001
