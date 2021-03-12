"""
In this file the object 'Parameter' is defined which contains all relevant
parameters to set up and run the model as well as the sequential sampling
method.
"""


from .helpers import get_loads


class Parameter:
    """
    This object contains all relevant parameters for the model.
    """

    def __init__(self):

        # Size of monte carlo sample
        self.sample_size = 10

        # Size of test sample size
        self.test_size = 5

        # Fixed generator costs in $/h
        self.c1 = 2.12*10**-5

        # Linear generator costs in $/kWh
        self.c2 = 0.128

        # Maximum capacity of generator
        self.pmax = 12

        # Minimum uptime of generator in hours
        self.uptime = 3

        # Minimum downtime of generator in hours
        self.downtime = 4

        # Ramping constraint of generator in kW
        self.ramping_constraint = 5

        # Electricity price forward contract in $/kWh
        self.l1 = 0.25

        # Electricity price real time contract in $/kWh
        self.l2 = 0.3

        # Storage types
        self.ESRS = ['battery']

        # Maximum charging power of storage in kW
        self.esr_to_p_w_max = {
            esr_type: 11 if 'ev' in esr_type else 10 for esr_type in self.ESRS
        }

        # Maximum discharging power of storage in kW
        self.esr_to_p_i_max = {
            esr_type: 11 if 'ev' in esr_type else 10 for esr_type in self.ESRS
        }

        # Maximum storage level in kWh
        self.esr_to_stor_level_max = {
            esr_type: 38 if 'ev' in esr_type else 5 for esr_type in self.ESRS
        }

        # Initial storage level in kWh
        self.esr_to_stor_level_zero = {
            esr_type: 0.3*self.esr_to_stor_level_max[esr_type] if (
                'ev' in esr_type) else 0 for esr_type in self.ESRS
        }

        # Charge target of ev in %
        self.charge_target = 0.6

        # Hour when EVs are plugged in
        self.plug_in_hour = 8

        # Hour when EVs are plugged out
        self.plug_out_hour = 17

        # Value for minimum state of charge in % of maximum capacity
        self.min_soc = 0.2

        # Value for maximum state of charge in % of maximum capacity
        self.max_soc = 0.8

        # Load values in kW
        self.LOADS = get_loads()

        # Hours
        self.HOURS = list(range(0, len(self.LOADS)))

        # Arbitrary value for convergence check
        self.epsilon = 0.0001

        # Parameter for sequential sampling
        self.seq_h = 0.008
        self.seq_epsilon = 0.22
        self.seq_p = 2.7 * 10 ** -2
        self.seq_c_p = 9.7667
