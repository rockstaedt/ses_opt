from modeling_helper.utilities import *

# Seet for randomness
seed = 12

# Size of monte carlo sample
sample_size = 10

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

# Maximum charging power of storage in kW
p_w_max = 10

# Maxium discharging power of storage in kW
p_i_max = 10

# Load values in kW
LOADS = get_loads()

# Hours
HOURS = list(range(0, len(LOADS)))

# Arbitrary value for convergence check
epsilon = 0.0001