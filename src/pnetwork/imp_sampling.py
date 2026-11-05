"""
Uncertainty sampling for subcatchments' % impervious area

Main Functions:
- imperv_sampling_sampling(modes, num_sample, num_resample, concentration=250, file_name='', folder='samples/', lhs_even_prob=True)
-
Note:
ChatGPT helps with function documentation using the prompt: 'document the function: {function}'

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import numpy as np

def beta_fixed_beta(_beta, mode):
    """
    Calculate the alpha parameter of a Beta distribution given the beta parameter and the mode.

    Parameters
    ----------
    _beta : float
        The beta parameter of the Beta distribution (must be > 2).
    mode : float
        The mode of the Beta distribution (must be between 0 and 1).

    Returns
    -------
    float
        The calculated alpha parameter.
    """
    return (mode * (_beta - 2) + 1) / (1 - mode)


def beta_fixed_concentration(concentration, mode):
    """
    Calculate the alpha and beta parameters of a Beta distribution given its total concentration and mode.

    Parameters
    ----------
    concentration : float
        The sum of alpha and beta (i.e., alpha + beta). Must be > 2.
    mode : float
        The mode of the Beta distribution (must be between 0 and 1).

    Returns
    -------
    alpha : float
        The alpha parameter of the Beta distribution.
    beta : float
        The beta parameter of the Beta distribution.
    """
    alpha = mode * (concentration - 2) + 1
    beta = (1 - mode) * (concentration - 2) + 1
    return alpha, beta


from scipy.stats import beta
from pnetwork.ag_sampling import even_prob_lhs

def beta_sampling(_alpha, _beta, sample_size, lhs=True, seed=0):
    """
    Sample values from a Beta distribution and convert them to percentages.

    Parameters
    ----------
    _alpha : float
        The alpha parameter of the Beta distribution.
    _beta : float
        The beta parameter of the Beta distribution.
    sample_size : int
        Number of samples to draw.
    lhs : bool, optional
        If True, use Latin Hypercube Sampling (LHS) for better sample coverage.
        If False, use random sampling. Default is True.
    seed : int, optional
        Random seed for reproducibility (default is 0).

    Returns
    -------
    ndarray
        An array of sampled values (in percentage, i.e., scaled to [0, 100]).
    """
    if lhs:
        x = beta.ppf(even_prob_lhs(sample_size, seed=seed), _alpha, _beta) * 100
    else:
        x = beta.rvs(_alpha, _beta, size=sample_size, random_state=seed) * 100
    return x

def beta_sampling_lhs(_alpha, _beta, even_prob):
    """
    Sample values from a Beta distribution using Latin Hypercube Sampling (LHS)
    and convert them to percentages.

    Parameters
    ----------
    _alpha : float
        The alpha parameter of the Beta distribution.
    _beta : float
        The beta parameter of the Beta distribution.
    even_prob : ndarray
        An array of evenly spaced probabilities for LHS.

    Returns
    -------
    ndarray
        An array of sampled values (in percentage, i.e., scaled to [0, 100]).
    """
    x = beta.ppf(even_prob, _alpha, _beta) * 100
    return x



def imperv_sampling_per_subcatchment(mode, concentration, num_sample, num_resample, seeds=1):
    """
    Samples percentage of impervious area for a single subcatchment based on the given mode
    and concentration using a concave Beta distribution, and performs repeated resampling.

    Parameters
    ----------
    mode : float
        Mode of the percentage impervious area (range: 0 to 100).
    concentration : float
        Concentration parameter (alpha + beta) of the Beta distribution. Must be > 2.
    num_sample : int
        Number of samples to draw in each resample.
    num_resample : int
        Number of independent resamples to generate.
    seeds : int, optional
        Number of random seeds to use (default is 1).

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_resample, num_sample) containing resampled
        percentage impervious area values for the subcatchment.
    """
    resample = np.zeros((seeds,num_resample, num_sample))

    for seed in range(seeds):
        for i in range(num_resample):
            para_alpha, para_beta = beta_fixed_concentration(concentration, mode / 100)
            resample[seed, i, :] = beta_sampling(para_alpha, para_beta, sample_size=num_sample, lhs=True, seed=seed)

    return resample


def imperv_sampling_per_subcatchment_lhs(mode, concentration, num_sample, even_prob, num_resample):
    """
    Samples percentage of impervious area for a single subcatchment based on the given mode
    and concentration using a concave Beta distribution, and performs repeated resampling.

    Parameters
    ----------
    mode : float
        Mode of the percentage impervious area (range: 0 to 100).
    concentration : float
        Concentration parameter (alpha + beta) of the Beta distribution. Must be > 2.
    num_sample : int
        Number of samples to draw in each resample.
    even_prob : ndarray
        An array of evenly spaced probabilities for LHS.
    num_resample : int
        Number of independent resamples to generate.

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_resample, num_sample) containing resampled
        percentage impervious area values for the subcatchment.
    """
    resample = np.zeros((num_resample, num_sample))

    for i in range(num_resample):
        para_alpha, para_beta = beta_fixed_concentration(concentration, mode / 100)
        resample[i, :] = beta_sampling_lhs(para_alpha, para_beta, even_prob=even_prob)

    return resample

import os
import random

def imperv_sampling_sampling(modes, num_sample, num_resample, concentration=250, 
                             file_name='', folder='samples/', lhs_even_prob=True, seeds=1):
    """
    Samples percentage of impervious area for multiple subcatchments based on their respective modes,
    using a concave Beta distribution. Stores the samples in a file and returns the shuffled result.

    Parameters
    ----------
    modes : list of float
        A list of mode values (0–100) for percentage impervious area, one per subcatchment.
    num_sample : int
        Number of samples to draw in each resample.
    num_resample : int
        Number of independent resamples to generate per subcatchment.
    concentration : float, optional
        Concentration parameter (alpha + beta) of the Beta distribution (default is 250).
    file_name : str, optional
        Optional file prefix used when saving the samples (default is an empty string).
    folder : str, optional
        Directory path to store the sample file (default is 'samples/').
    lhs_even_prob : bool, optional
        If True, use the function with "_lhs" Latin Hypercube Sampling (LHS) for better sample coverage.
        If False, still do LHS but sampling different per subcatchment. Default is True.
    seeds : int, optional
        Number of random seeds to use (default is 1).

    Returns
    -------
    np.ndarray
        A 3D array of shape (num_subcatchments, num_resample, num_sample),
        containing the shuffled resampled percentage impervious area values.
    """
    os.makedirs(folder, exist_ok=True)
    num_sub = len(modes)

    imperv_samples = np.zeros((num_sub, seeds, num_resample, num_sample))

    if lhs_even_prob:
        for seed in range(seeds):
            even_prob = even_prob_lhs(num_sample, seed=seed)
            for i in range(num_sub):
                imperv_samples[i, seed, :, :] = imperv_sampling_per_subcatchment_lhs(
                    modes[i], concentration, num_sample, even_prob, num_resample
                )
    else:
        for i in range(num_sub):
            imperv_samples[i, :, :, :] = imperv_sampling_per_subcatchment(
                modes[i], concentration, num_sample, num_resample, seeds=seeds
            )

    shuffled_ind = random.sample(range(num_sample), num_sample)

    np.save(os.path.join(folder, file_name + 'imperv.npy'), imperv_samples[:, :, :, shuffled_ind])

    return imperv_samples[:, :, :, shuffled_ind]
