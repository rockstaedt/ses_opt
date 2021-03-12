"""
This file contains the 'SequentialSampling' object. This object is used to run
a sequential sampling method. Hereby, it is possible to define the sampling
method like crude monte carlo (MC), antithetic variates (AV) or latin
hypercube sampling (LHS). Furthermore, it is possible to select the single
replication procedure (SRP), or the averaged two-replication procedure (A2RP)
to estimate the gap and the sample variance.
"""

import math
import numpy as np
from random import randint

from .l_shape import LShapeMethod
from .parameters import Parameter
from .helpers import print_title, print_caption


class SequentialSampling:
    """
    This object is used to run a sequential sampling method.
    """
    def __init__(self, params: Parameter, sampling_method: str,
                 multiprocessing: bool, estimator_method: str = 'SRP'):
        """
        Init SequentialSampling object.

        :param params: Parameter object containing all relevant parameters.
        :param sampling_method: Method to draw samples. Possible options:
        crude monte carlo ('MC'), antithetic variates ('AV'), latin hypercube
        sampling ('LHS').
        :param multiprocessing: Enables multiprocessing while solving the
        problem.
        :param estimator_method: Select the single replication procedure (SRP),
        or the averaged two-replication procedure (A2RP) to estimate the gap
        and the sample variance.
        """
        self.params = params
        self.sampling_method = sampling_method
        self.multiprocessing = multiprocessing
        self.estimator_method = estimator_method

        # Init variables
        # Gap estimator
        self.G = 10
        # Sample variance
        self.SV = 10
        # Iteration
        self.k = 0
        # Final iteration
        self.T = 0
        # Confidence intervall
        self.CI = 0
        # High quality solution results
        self.x_star = None
        self.z_star = None

        # Seed for randomness
        self.seed_m = 12
        self.seed_n = 5639

        # Sample size candidate solution
        self.m = 20
        # Sample size optimal solution
        self.n = None
        # List of all sample sizes of the sequential sampling method
        self.ns = []

    def run_seq_sampling(self):
        """
        This function runs a sequential sampling method.
        """

        # Init variables
        x_cans = None
        z_cans = None

        while self.G > self.__get_border():

            self.k += 1

            self.n = self.__get_n_k()

            self.m = self.__get_m_k()

            print_title(f'Iteration k={self.k}')

            print(f'--> Calculate candidate solution for m = {self.m}\n')

            saa = LShapeMethod(
                self.params,
                sample_size=self.m,
                seed=self.seed_m,
                sampling_method='MC',
                multiprocessing=self.multiprocessing,
                progress_info=False
            )

            _, x_cans = saa.get_solution()

            print(f'\n--> Calculate optimal solution for n = {self.n}\n')

            # Init variable in case it is not set by the following if structure.
            seq = None

            if self.estimator_method == 'SRP':
                if self.sampling_method == 'MC' or self.sampling_method == 'AV':
                    seq = LShapeMethod(
                        self.params,
                        sample_size=self.n,
                        seed=self.seed_n,
                        sampling_method=self.sampling_method,
                        multiprocessing=self.multiprocessing,
                        progress_info=False
                    )
                elif self.sampling_method == 'LHS':
                    # A completely new sample is drawn for each iteration
                    # in the case of LHS. That is why, a random integer
                    # number in the range from 200 to 300 is drawn and passed
                    # to the LHS method.
                    seq = LShapeMethod(
                        self.params,
                        sample_size=self.n,
                        seed=randint(200, 300),
                        sampling_method=self.sampling_method,
                        multiprocessing=self.multiprocessing,
                        progress_info=False
                    )

                z_opt, _ = seq.get_solution()

                seq.run_test(x_cans)

                z_cans = seq.test_objective_values

                print(f'\n--> Calculate Gap\n')

                self.G, self.SV = self.__calculate_estimators(
                    self.sampling_method,
                    self.n,
                    z_cans,
                    z_opt
                )

            elif self.estimator_method == 'A2RP':
                # For A2RP two estimators are calculated and then averaged.
                gs = []
                svs = []
                for i in range(2):
                    print(f'--> Part {i + 1}')
                    if (self.sampling_method == 'MC'
                            or self.sampling_method == 'AV'):
                        seq = LShapeMethod(
                            self.params,
                            sample_size=int(self.n/2),
                            seed=self.seed_n,
                            sampling_method=self.sampling_method,
                            multiprocessing=self.multiprocessing,
                            progress_info=False
                        )
                    elif self.sampling_method == 'LHS':
                        seq = LShapeMethod(
                            self.params,
                            sample_size=int(self.n/2),
                            seed=randint(200, 300),
                            sampling_method=self.sampling_method,
                            multiprocessing=self.multiprocessing,
                            progress_info=False
                        )

                    z_opt, _ = seq.get_solution()

                    seq.run_test(x_cans)

                    z_cans = seq.test_objective_values

                    g, sv = self.__calculate_estimators(
                        self.sampling_method,
                        int(self.n/2),
                        z_cans,
                        z_opt
                    )

                    gs.append(g)
                    svs.append(sv)

                print(f'\n--> Calculate Gap\n')

                self.G = np.sum(gs) * 0.5
                self.SV = np.sum(svs) * 0.5

            print(f'--> Results:')

            print(f'\tGAP = {self.G}')

            print(f'\tSV = {self.SV}')

            print(f'\tBorder = {self.__get_border()}\n')

        self.T = self.k
        self.x_star = x_cans
        self.z_star = np.mean(z_cans)
        # 'k' and 'T' are the same. That is why function '__a_k' can be used
        # in the calculation of the confidence intervall.
        self.CI = (
            self.params.seq_epsilon*math.sqrt(self.SV)
            + self.__a_k() + self.params.epsilon
        )

        print_caption('End.')

        print(
            f'Results:\n\tG={self.G}\n\tT={self.T}\n\tCI={self.CI}'
            f'\n\tz_star={self.z_star}'
        )

    def __get_n_k(self):
        """
        This function calculates the sample size in iteration k.
        :return: sample size for optimal solution in iteration k
        """
        n_k = math.pow(1 / (self.params.seq_h - self.params.seq_epsilon), 2) * (
                self.params.seq_c_p + 2 * self.params.seq_p
                * math.pow(self.__a_k(), -2) * math.pow(np.log(self.k), 2)
        )
        # Cut decimal points and increase by one to obtain an integer.
        n_k = math.trunc(n_k) + 1
        if n_k % 4 != 0:
            # To get a sample size which divisible by four, modulo is
            # subtracted and 4 added.
            n_k = n_k - n_k % 4 + 4

        if self.k != 1:
            # Increase sample size until it is bigger than the previous.
            # This is only needed for k > 1 iterations.
            while n_k <= self.ns[-1]:
                n_k += 4
        self.ns.append(n_k)
        return n_k

    def __get_m_k(self):
        """
        This function defines how the sample size to calculate the candidate
        solution is increased in iteration k.
        :return: sample size for candidate solution in iteration k
        """
        if self.k == 1:
            return self.m
        else:
            return self.m + 200

    def __get_border(self):
        """
        This function calculates the termination condition.
        :return: value of termination condition
        """
        return self.params.seq_h*math.sqrt(self.SV)+self.params.epsilon

    def __a_k(self): return 1/self.k

    @staticmethod
    def __calculate_estimators(sampling_method: str, n: int, z_cans: list,
                               z_opt: float):
        """
        This function calculates the gap and sample variance estimator based
        on the sampling method.
        :param sampling_method: Method for sampling.
        :param n: Sample size.
        :param z_cans: List of objective values of the candidate solution.
        :param z_opt: Optimal solution.
        :return: gap estimator, sample variance.
        """
        if sampling_method == 'LHS' or sampling_method == 'MC':
            # GAP estimator
            g = 1 / n * np.sum(
                np.array(z_cans) - np.array([z_opt] * n)
            )
            # SV estimator
            sv = 1 / (n - 1) * np.sum(
                np.power(
                    (np.array(z_cans) - np.array([z_opt] * n)) - g,
                    2
                )
            )
        else:
            # Estimator equation are different for AV sampling.
            mid = int(n / 2)
            left = np.array(z_cans[:mid])
            right = np.array(z_cans[mid:])

            z_cans_new = list(0.5 * (left + right))

            g = 1 / (n / 2) * np.sum(
                np.array(z_cans_new) - np.array([z_opt] * mid)
            )
            # SV estimator
            sv = 1 / ((n / 2) - 1) * np.sum(
                np.power(
                    (np.array(z_cans_new) - np.array([z_opt] * mid)) - g,
                    2
                )
            )

        return g, sv
