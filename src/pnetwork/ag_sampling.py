"""
Uncertainty sampling for pipe aging (Manning's roughness coefficient)

Main Functions:
- pipe_aging_sampling(models, num_sample, num_resample, folder='samples/')
-
Note:
ChatGPT helps with function documentation using the prompt: 'document the function: {function}'

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import numpy as np
from scipy.stats import beta

def statistical_n(sample_size, lhs=True, seed=0):
    """
    Samples Manning’s roughness coefficient (n) values based on a fitted Beta distribution.

    Parameters:
    -----------
    sample_size : int
        Number of samples to generate.
    lhs : bool, optional
        If True, use Latin Hypercube Sampling (LHS) to ensure stratified sampling across the distribution.
        If False, draw random samples from the fitted Beta distribution (default is True).
    seed : int, optional
        Random seed for reproducibility (default is 0).

    Returns:
    --------
    np.ndarray
        An array of sampled Manning's n values, shuffled to remove any ordering.

    Notes:
    ------
    - The Beta distribution used here is fitted to represent statistical variation in Manning’s roughness
      coefficient among different materials, based on Coelho et al. (2022).
    - The fitted Beta distribution has parameters: alpha = 1.0310, beta = 2.4812.
    - The values are rescaled to match the empirical range:
        * Minimum: 0.0118
        * Median:  0.0235
        * Maximum: 0.0583
    - Latin Hypercube Sampling ensures better coverage of the distribution space, especially for small sample sizes.
    """

    # Empirical min, median, and max values for Manning’s n
    min_val = 0.0118
    median_val = 0.0235
    max_val = 0.0583

    # Draw samples from the fitted Beta distribution (in [0, 1])
    if lhs:
        x = beta.ppf(even_prob_lhs(sample_size, seed=seed), 1.0310, 2.4812)
    else:
        x = beta.rvs(1.0310, 2.4812, size=sample_size, random_state=seed)

    # Rescale to empirical range
    x_rescaled = min_val + x * (max_val - min_val)

    # Shuffle samples to randomize order
    np.random.shuffle(x_rescaled)


    """ Code for finding the best fit alpha and beta
    
    # Rescale the values to [0, 1] range
    def rescale(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)
    
    median_rescaled = rescale(median_val, min_val, max_val)
    
    # Objective function to minimize: difference between the desired and actual percentiles
    def objective(params):
        alpha, beta_param = params
        # Calculate the CDF values at the 50th percentile (median)
        median_cdf = beta.cdf(median_rescaled, alpha, beta_param)
        # We want the 50th percentile to match the rescaled median value (which should be 0.5)
        return (median_cdf - 0.5) ** 2
    
    # Initial guess for alpha and beta
    initial_guess = [2, 2]
    
    # Use scipy's minimize to find the best-fit alpha and beta
    result = minimize(objective, initial_guess, bounds=((0.1, None), (0.1, None)))
    alpha_opt, beta_opt = result.x ## optimized alpha: 1.0310, optimized beta: 2.4812
    
    """
    return x_rescaled

import numpy as np
from scipy.stats import uniform

def udd_n(sample_size, lhs=True, seed=0):
    """
    Samples Manning’s roughness coefficient (n) from a Uniform Distribution defined by Chow (1959), 
    also suggested in many codes including Urban Drainage Design ((USDT Federal Highway Administration, 2024).

    Parameters:
    -----------
    sample_size : int
        Number of samples to generate.
    lhs : bool, optional
        If True, uses Latin Hypercube Sampling (LHS) to improve distributional coverage. 
        If False, uses simple random sampling from a uniform distribution. Default is True.

    Returns:
    --------
    np.ndarray
        An array of sampled Manning's n values in the range [0.013, 0.017].

    Reference:
    ----------
    Chow, V.T. (1959). *Open-Channel Hydraulics*. McGraw-Hill.
    """
    if lhs:
        samples = udd_sampling_lhs(a=0.013, b=0.017, sample_size=sample_size, seed=seed)
    else:
        rng = np.random.RandomState(seed)
        samples = rng.uniform(0.013, 0.017, sample_size)
    return samples


def even_prob_lhs(sample_size, seed=0):
    """
    Generates stratified probabilities for Latin Hypercube Sampling (LHS).

    Parameters:
    -----------
    sample_size : int
        Number of probabilities to generate.
    seed : int, optional
        Random seed for reproducibility (default is 0).

    Returns:
    --------
    np.ndarray
        An array of `sample_size` uniformly spaced probabilities in the range (0, 1),
        stratified into equal-probability intervals.
    
    Notes:
    ------
    Each probability is drawn from a unique interval of width 1/sample_size, then randomly offset within it.
    """
    rng = np.random.RandomState(seed)
    uniform_random = rng.uniform(0, 1, sample_size)
    for i in range(sample_size):
        uniform_random[i] = (1 / sample_size) * (i + uniform_random[i])
    return uniform_random


def udd_sampling_lhs(a, b, sample_size, seed=0):
    """
    Samples from a Uniform(a, b) distribution using Latin Hypercube Sampling (LHS).

    Parameters:
    -----------
    a : float
        Lower bound of the uniform distribution.
    b : float
        Upper bound of the uniform distribution.
    sample_size : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility (default is 0).

    Returns:
    --------
    np.ndarray
        An array of LHS-based samples from the Uniform(a, b) distribution, shuffled for randomness.

    Notes:
    ------
    - Uses `even_prob_lhs` to ensure samples cover the full range evenly.
    - Applies `uniform.ppf` (percent point function) to map probabilities to values.
    - Final samples are shuffled to remove ordering bias.
    """

    samples = uniform.ppf(even_prob_lhs(sample_size, seed=seed), loc=a, scale=b - a)
    np.random.shuffle(samples)
    return samples

import os
import pandas as pd

def pipe_aging_sampling_per_model(model, num_sample, num_resample, folder='samples/', seeds=1):
    """
    Generates pipe aging samples for a given roughness aging model and stores resampled datasets.

    Parameters:
    -----------
    model : str
        The name of the roughness aging model to use. Must be either `'udd'` or `'stat'`:
            - `'udd'`: Uniform distribution derived from Chow (1959).
            - `'stat'`: Statistically derived Beta distribution from Coelho et al. (2022).
    num_sample : int
        The number of samples to generate for each resample.
    num_resample : int
        The number of times to resample (i.e., how many replicated datasets to generate).
    folder : str, optional
        Directory where the output CSV file will be saved (default is `'samples/'`).
    seeds : int, optional
        Number of random seeds to use (default is 1).

    Returns:
    --------
    pd.DataFrame or None
        A DataFrame containing all sampled roughness values and metadata for the given model.
        Returns None if the model name is not recognized.

    Notes:
    ------
    - The function calls either `udd_n` or `statistical_n` based on the model name.
    - All resamples are concatenated into a single DataFrame with metadata columns:
        `'samples_id'`, `'roughness'`, `'aging_model'`, `'resample_id'`.
    - The final combined dataset is saved as a CSV file named `aging_{model}.csv` in the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    resample_df = []

    for seed in range(seeds):
        for i in range(num_resample):
            # Generate samples based on the specified aging model
            if model == 'udd':
                samples = udd_n(num_sample, lhs=True, seed=seed)
            elif model == 'stat':
                samples = statistical_n(num_sample, lhs=True, seed=seed)
            else:
                # print('Wrong model type')
                return None

            df = pd.DataFrame({
                'samples_id': list(range(num_sample)), 
                'roughness': samples, 
                'aging_model': [model] * num_sample, 
                'resample_id': [i] * num_sample,
                'seed': [seed] * num_sample
            })

            resample_df.append(df)

    # Combine all resampled data into a single DataFrame
    model_df = pd.concat(resample_df, ignore_index=True)

    # Save to CSV
    model_df.to_csv(os.path.join(folder, f"aging_{model}.csv"), index=False)

    return model_df


def pipe_aging_sampling(models, num_sample, num_resample, folder='samples/', seeds=1):
    """
    Generates and stores pipe aging samples for multiple roughness aging models.

    Parameters:
    -----------
    models : list of str
        A list of roughness aging models to sample from (e.g., ['udd', 'stat']).
    num_sample : int
        The number of samples per resample.
    num_resample : int
        The number of resampling iterations to perform for each model.
    folder : str, optional
        Directory where all result CSV files will be saved (default is `'samples/'`).
    seeds : int, optional
        Number of random seeds to use (default is 1).

    Returns:
    --------
    dict
        A dictionary mapping each model name to its corresponding DataFrame of resampled data.

    Notes:
    ------
    - Calls `pipe_aging_sampling_per_model()` for each model in the input list.
    - Each model's sampled data is saved as `{model}.csv` in the specified folder.
    - The result is useful for analyzing uncertainty in hydraulic roughness due to aging.
    """
    os.makedirs(folder, exist_ok=True)

    pre_df = {}

    for model in models:
        df = pipe_aging_sampling_per_model(model=model, 
                                           num_sample=num_sample, 
                                           num_resample=num_resample, 
                                           folder=folder, seeds=seeds)
        pre_df[model] = df

    return pre_df
