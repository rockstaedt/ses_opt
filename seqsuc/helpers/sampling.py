"""
This file contains all sampling functions that were used in the project.
"""

import numpy as np
from scipy.stats import norm


def get_monte_carlo_samples(values: list, sample_size=1000, seed=12):
    """
    This function creates a monte carlo sample of size samples. For every
    element in values, a sample is drawn from a normal distribution with the
    elements value as mean and one third from the elements value as deviation.
    """
    # Set seed
    np.random.seed(seed)
    # Covariance matrix of pl
    cov_pl = np.diagflat(np.array(values)*1/3)

    return np.random.multivariate_normal(values, cov_pl, size=sample_size)


def get_av_samples(values: list, sample_size=1000, seed=12):
    """
    This function creates a sample using the antithetic variates (AV) technique.
    """
    # Set seed
    np.random.seed(seed)
    # Covariance matrix of values. Variance was given.
    cov_pl = np.diagflat(np.array(values)*1/3)
    # Calculate normal distribution for half of the sample size.
    normal_dis = np.random.multivariate_normal(
        values,
        cov_pl,
        size=int(sample_size/2)
    )
    # Calculate antithetic of these sample
    antithetic = np.array(values) - (normal_dis - np.array(values))
    # Concatenate both samples to one.
    samples = np.concatenate((normal_dis, antithetic))
    return samples


def get_lhs_samples(values: list, sample_size=1000, seed=12):
    """
    This function creates a vector of samples using the Latin Hypercube
    technique.
    """

    sample_vector = []

    np.random.seed(seed)
    for v in values:

        perc_arr = []
        help_array = []

        if v == 0:
            help_array = np.zeros(sample_size)
        else:
            value = v

            i = 0
            perc = 1/sample_size
            while i < 1:
                perc_arr.append(i)
                i += perc

            if len(perc_arr) != sample_size+1:
                perc_arr.append(0.999999999999999)

            perc_arr[0] += 0.0000000000000001

            for j in range(len(perc_arr)-1):
                x = np.random.uniform(
                    norm.ppf(
                        perc_arr[j],
                        loc=value,
                        scale=np.sqrt(value*(1/3))
                    ),
                    norm.ppf(
                        perc_arr[j+1],
                        loc=value,
                        scale=np.sqrt(value*(1/3))
                    )
                )
                help_array.append(x)

        np.random.shuffle(help_array)
        sample_vector.append(np.array(help_array))

    sample_vector = np.array(sample_vector)
    sample_vector = sample_vector.transpose()

    return sample_vector
