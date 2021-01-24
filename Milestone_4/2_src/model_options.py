###############################################################################
### Model Options
###############################################################################

# This option enables a deterministic approach. If false then the stochastic
# approch is performed.
deterministic = True

# This option enables sensitivity analysis regarding the storage capacity from
# 0 to 20 kWh in steps of 4 kWh
sensitivity_analysis = False

# This option enables the output of result files, saved into '3_results'.
output = True

# This option enables the implementation of uptime and downtime constraint of
# the generator
up_down_time = True

# This option enables the implementation of the ramping constraint of the
# generator
ramping = True

# This option enables the implementation of a energy storage resource.
esr = False