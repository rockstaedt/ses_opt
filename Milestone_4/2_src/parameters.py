###############################################################################
### Paremeters
###############################################################################

from modeling_helper.utilities import *
from model_options import deterministic, sensitivity_analysis

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

# Maximum storage level in kWh
# This parameter is part of a sensitivity analysis from 0 kWh to 20 kWh in steps
# of 4. Therefore the parameter is defined as a list in case of a sensitivity
# analysis. If no sensitivity analysis is performed, the intial value of 4 kWh
# is used.
if sensitivity_analysis:
    stor_levels_max = [0, 4, 8, 12, 16, 20]
else:
    stor_levels_max = [4]