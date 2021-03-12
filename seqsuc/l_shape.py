"""
This file contains the object 'LShapeMethod' which is used to solve a two stage
stochastic unit commitment problem.
"""

import pathos.pools as pp

from .helpers import *
from .uc_model import *


class LShapeMethod:
    """
    This object implements an L-shape decomposition algorithm to solve a two
    stage stochastic unit commitment problem.
    """

    def __init__(self, params: Parameter, sample_size: int, seed: int,
                 sampling_method: str, multiprocessing: bool,
                 output: bool = False, progress_info: bool = True):
        """
        Initialize a LShapeMethod object.

        :param params: Parameter object containing all relevant parameters.
        :param sample_size: Sample size of load values.
        :param seed: Seed for randomness.
        :param sampling_method: Method to draw samples. Possible options:
        crude monte carlo ('MC'), antithetic variates ('AV'), latin hypercube
        sampling ('LHS').
        :param multiprocessing: Enables multiprocessing while solving the
        problem.
        :param output: Enables the output of result files.
        :param progress_info: Enables print statements about the progress in
        the terminal.
        """
        self.params = params
        self.sample_size = sample_size
        self.seed = seed
        self.sampling_method = sampling_method
        self.multiprocessing = multiprocessing
        self.output = output
        self.progress_info = progress_info

        # Pyomo solver
        self.opt = pyo.SolverFactory('gurobi')

        self.SAMPLES = self.__get_samples()
        # Iteration counter
        self.iteration: int = 0
        self.master: pyo.ConcreteModel = pyo.ConcreteModel()
        self.sub: pyo.ConcreteModel = pyo.ConcreteModel()
        # List of the objective values
        self.objective_values = []
        # List of lower bound values
        self.lower_bounds = []
        self.results_master = None
        self.results_sub = {}
        self.solved = False

        # Testing
        self.test_sample_size = None
        self.TEST_SAMPLES = None
        self.test_seed = None
        self.test: pyo.ConcreteModel = pyo.ConcreteModel()
        self.test_results = {}
        self.test_objective_values = []
        self.tested = False

    def solve_model(self):
        """
        This function solves the L-shape method.
        """
        if self.progress_info:
            print_caption('Initialization')
            print('Create master problem...')

        self.master = create_master_problem(self.params)

        if self.progress_info:
            print('Solve master problem...')

        solve_model(self.opt, self.master)
        self.results_master = get_results(self.master)

        if self.progress_info:
            print('Create sub problem...')
        self.sub = create_sub_problem(self.params, self.results_master)

        # Enable calculation of dual variables in pyomo.
        self.sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        if self.progress_info:
            print(f'Solve sub problem for samples size = {self.sample_size}')

        # Create process pool for multiprocessing. Numbers of worker are set
        # automatically.
        pool = pp.ProcessPool()

        if self.multiprocessing:
            # Use a mapping function to solve samples in parallel.
            results = pool.map(
                solve_sample,
                self.SAMPLES,
                list(range(len(self.SAMPLES))),
                [len(self.SAMPLES)] * len(self.SAMPLES),
                [self.sub] * len(self.SAMPLES),
                [self.opt] * len(self.SAMPLES),
                [False] * len(self.SAMPLES),
                [self.progress_info] * len(self.SAMPLES)
            )
        else:
            # Solve model per iteration.
            results = map(
                solve_sample,
                self.SAMPLES,
                list(range(len(self.SAMPLES))),
                [len(self.SAMPLES)] * len(self.SAMPLES),
                [self.sub] * len(self.SAMPLES),
                [self.opt] * len(self.SAMPLES),
                [False] * len(self.SAMPLES),
                [self.progress_info] * len(self.SAMPLES)
            )

        # Results are stored in a map object and have to be unpacked into a
        # dict.
        for i, result in enumerate(results):
            self.results_sub[i] = result

        # Check if upper and lower bound are converging.
        converged, upper_bound, lower_bound = convergence_check(
            self.__objective,
            self.__master_prob,
            self.results_master,
            self.results_sub,
            samples=self.SAMPLES,
            params=self.params
        )

        if self.progress_info:
            print_convergence(converged)

        self.objective_values.append(upper_bound)

        self.lower_bounds.append(lower_bound)

        # Optimize until upper and lower bound are converging
        while not converged:
            self.iteration += 1

            if not self.progress_info:
                print(f'\tIteration {self.iteration}...')

            if self.progress_info:
                print_caption(f'Iteration {self.iteration}')

            def cut(master, h):
                return (
                        sum(
                            self.params.c2 * self.results_sub[j]['pg'][h]
                            + self.params.l2 * self.results_sub[j]['p2'][h]
                            + self.results_sub[j]['dual_con1'][h] * (
                                    self.master.u[h]
                                    - self.results_master['u'][h]
                            )
                            + self.results_sub[j]['dual_con2'][h] * (
                                    self.master.p1[h]
                                    - self.results_master['p1'][h]
                            ) for j, sample in enumerate(self.SAMPLES)
                        ) / self.sample_size
                        <= master.alpha[h]
                )

            setattr(
                self.master, f'cut_{self.iteration}',
                pyo.Constraint(self.master.H, rule=cut)
            )

            if self.progress_info:
                print(f'Added cut_{self.iteration}')
                print('Solving master problem...')

            solve_model(self.opt, self.master)
            self.results_master = get_results(self.master)

            # Update dual constraint in sub problem
            self.sub.results_master = self.results_master
            self.sub.dual_con1.reconstruct()
            self.sub.dual_con2.reconstruct()

            if self.progress_info:
                print(
                    f'Solving sub problem for samples size = {self.sample_size}'
                )

            if self.multiprocessing:
                # Use a mapping function to solve samples in parallel.
                results = pool.map(
                    solve_sample,
                    self.SAMPLES,
                    list(range(len(self.SAMPLES))),
                    [len(self.SAMPLES)] * len(self.SAMPLES),
                    [self.sub] * len(self.SAMPLES),
                    [self.opt] * len(self.SAMPLES),
                    [False] * len(self.SAMPLES),
                    [self.progress_info] * len(self.SAMPLES)
                )
            else:
                # Solve model per iteration.
                results = map(
                    solve_sample,
                    self.SAMPLES,
                    list(range(len(self.SAMPLES))),
                    [len(self.SAMPLES)] * len(self.SAMPLES),
                    [self.sub] * len(self.SAMPLES),
                    [self.opt] * len(self.SAMPLES),
                    [False] * len(self.SAMPLES),
                    [self.progress_info] * len(self.SAMPLES)
                )

            # Results are stored in a map object and have to be unpacked
            # into a dict.
            for i, result in enumerate(results):
                self.results_sub[i] = result

            # Check if upper and lower bound are converging.
            converged, upper_bound, lower_bound = convergence_check(
                self.__objective,
                self.__master_prob,
                self.results_master,
                self.results_sub,
                samples=self.SAMPLES,
                params=self.params
            )

            if self.progress_info:
                print_convergence(converged)

            self.objective_values.append(upper_bound)

            self.lower_bounds.append(lower_bound)

        self.solved = True

        if self.progress_info:
            print_caption('End')

    def get_solution(self):
        """
        This function checks if the model was already solved. If so, the
        objective value and the first stage variables are returned.
        :return: Objective value, first stage variables
        """
        if not self.solved:
            self.solve_model()
        return self.objective_values[-1], self.results_master

    def run_test(self, first_stage_variables, new_samples: bool = False,
                 test_sample_size: int = None, test_seed: int = None):
        """
        This function tests the passed first stage variables on a new sample
        set if 'new_samples' are enabled.

        :param first_stage_variables: First stage variables to test.
        :param new_samples: Enables the drawing of new samples. Defaults to
        False.
        :param test_sample_size: Size of the test samples.
        :param test_seed: Seed for the randomness of the test sample.
        """
        self.test_seed = test_seed
        self.test_sample_size = test_sample_size

        if self.tested:
            # Reset test results.
            self.test_objective_values = []
            self.test_results = {}

        if self.progress_info:
            print_caption('Test Initialization')

        if new_samples and test_sample_size and test_seed:
            self.TEST_SAMPLES = get_monte_carlo_samples(
                self.params.LOADS,
                sample_size=self.test_sample_size,
                seed=self.test_seed
            )
        else:
            self.TEST_SAMPLES = get_monte_carlo_samples(
                self.params.LOADS,
                sample_size=self.sample_size,
                seed=self.seed
            )

        if self.progress_info:
            print('Create test problem...')
        self.test = create_test_problem(
            self.params,
            first_stage_variables
        )

        if self.progress_info:
            print('Solve test problem...')

        if self.multiprocessing:
            # Create process pool. Numbers of worker are set automatically.
            pool = pp.ProcessPool()
            # Use a mapping function to solve samples in parallel.
            results = pool.map(
                solve_sample,
                self.TEST_SAMPLES,
                list(range(len(self.TEST_SAMPLES))),
                [len(self.TEST_SAMPLES)] * len(self.TEST_SAMPLES),
                [self.test] * len(self.TEST_SAMPLES),
                [self.opt] * len(self.TEST_SAMPLES),
                [True] * len(self.TEST_SAMPLES),
                [self.progress_info] * len(self.TEST_SAMPLES)
            )
        else:
            # Solve model per iteration.
            results = map(
                    solve_sample,
                    self.TEST_SAMPLES,
                    list(range(len(self.TEST_SAMPLES))),
                    [len(self.TEST_SAMPLES)] * len(self.TEST_SAMPLES),
                    [self.test] * len(self.TEST_SAMPLES),
                    [self.opt] * len(self.TEST_SAMPLES),
                    [True] * len(self.TEST_SAMPLES),
                    [self.progress_info] * len(self.TEST_SAMPLES)
                )

        # Results are stored in a map object and have to be unpacked
        # into a dict.
        for i, result in enumerate(results):
            self.test_results[i] = result
            self.test_objective_values.append(
                self.test_results[i]['objective_value']
            )

        self.tested = True
        if self.progress_info:
            print_caption('End')

    def __get_samples(self):
        """
        This function returns the drawn samples according to the passed sampling
        method.
        :return: samples
        """
        if self.sampling_method == 'MC':
            return get_monte_carlo_samples(
                self.params.LOADS,
                sample_size=self.sample_size,
                seed=self.seed
            )
        elif self.sampling_method == 'AV':
            return get_av_samples(
                self.params.LOADS,
                sample_size=self.sample_size,
                seed=self.seed
            )
        elif self.sampling_method == 'LHS':
            return get_lhs_samples(
                self.params.LOADS,
                sample_size=self.sample_size,
                seed=self.seed
            )

    @staticmethod
    def __objective(u, p1, pg, p2, params):
        """
        This function calculates the objective value for all hours in hours.
        This is also the upper bound of the decomposition.
        """
        return sum(
            params.c1*u[h] + params.l1*p1[h] + params.c2*pg[h]
            + params.l2*p2[h] for h in params.HOURS
        )

    @staticmethod
    def __master_prob(u, p1, alpha, params):
        """
        This function calculates the lower bound of the decomposition.
        """
        return sum(
            params.c1*u[h] + params.l1*p1[h]
            + alpha[h] for h in params.HOURS
        )
