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
esr_types = ['battery', 'ev_1', 'ev_2', 'ev_3']

# Maximum charging power of storage in kW
esr_to_p_w_max = {
    esr_type: 11 if 'ev' in esr_type else 10 for esr_type in esr_types
}

# Maxium discharging power of storage in kW
esr_to_p_i_max = {
    esr_type: 11 if 'ev' in esr_type else 10 for esr_type in esr_types
}

# Maximum storage level in kWh
esr_to_stor_level_max = {
    esr_type: 38 if 'ev' in esr_type else 5 for esr_type in esr_types
}

# Initial storage level in kWh
esr_to_stor_level_zero = {
    esr_type: 0.3*esr_to_stor_level_max[esr_type] if (
        'ev' in esr_type ) else 0 for esr_type in esr_types
}

# Charge target of ev in %
charge_target = 0.6

# Hour when EVs are plugged in
plug_in_hour = 8

# Hour when EVs are plugged out
plug_out_hour = 17

# Value for minimum state of charge in % of maximum capacity
min_soc = 0.2

# Value for maximum state of charge in % of maximum capacity
max_soc = 0.8

# Load values in kW
LOADS = get_loads()

# Hours
HOURS = list(range(0, len(LOADS)))

# Arbitrary value for convergence check
epsilon = 0.0001
