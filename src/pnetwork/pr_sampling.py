"""
Uncertainty sampling for precipitation

Main Functions:
- precipitation_sampling(num_sample, num_resample, return_period, 
                           climate_model, year_name, folder='samples/pr/')
-
Note:
ChatGPT helps with function documentation using the prompt: 'document the function: {function}'

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import numpy as np
import pandas as pd
import pyreadr
import scipy.stats as stats
from scipy.interpolate import interp1d
import os
import pnetwork.rtd as rtd


def myreturnlevel(t, mu, sigma, xi):
    """
    Computes the rainfall intensity for a given return period using the Generalized Extreme Value (GEV) distribution.

    Parameters:
    -----------
    t : float
        Return period (in years).
    mu : float
        Location parameter of the GEV distribution.
    sigma : float
        Scale parameter of the GEV distribution.
    xi : float
        Shape parameter of the GEV distribution.

    Returns:
    --------
    float
        Rainfall intensity in inches (converted from mm/hr).

    Notes:
    ------
    - The result is first computed in mm/hr and then converted to inches (1 inch = 25.4 mm).
    """
    x = stats.genextreme.ppf(1 - (1 / t), -xi, loc=mu, scale=sigma)  # Rainfall intensity in mm/hr
    return x / 25.4  # Convert to inches


def inverse_transform(inversed_cdf_function, sample_size, seed=0):
    """
    Generates random samples using the inverse transform sampling method.

    Parameters:
    -----------
    inversed_cdf_function : callable
        A function representing the inverse cumulative distribution function (inverse CDF).
        Typically, this is an interpolated piecewise linear function (e.g., `scipy.interpolate.interp1d`).
    sample_size : int
        The number of samples to generate.
    seed : int, optional
        Random seed for reproducibility (default is 0).

    Returns:
    --------
    np.ndarray
        An array of samples drawn from the empirical distribution.
    """
    np.random.seed(seed)
    uniform_random = np.random.uniform(0, 1, sample_size)
    samples = inversed_cdf_function(uniform_random)
    return samples


def inverse_transform_lhs(inversed_cdf_function, sample_size, seed=0):
    """
    Generates random samples using the inverse transform sampling method with Latin Hypercube Sampling (LHS).

    Parameters:
    -----------
    inversed_cdf_function : callable
        A function representing the inverse cumulative distribution function (inverse CDF).
        Typically, this is an interpolated piecewise linear function (e.g., `scipy.interpolate.interp1d`).
    sample_size : int
        The number of samples to generate.

    Returns:
    --------
    np.ndarray
        An array of samples drawn from the empirical distribution, ensuring stratification via LHS.

    Notes:
    ------
    - Latin Hypercube Sampling (LHS) ensures that samples are drawn more evenly across the distribution.
    - Instead of purely random uniform values, LHS stratifies the sampling space into equal intervals.
    - The final samples are shuffled to remove any ordering bias.
    """
    # Generate uniform random values
    np.random.seed(seed)
    uniform_random = np.random.uniform(0, 1, sample_size)

    # Apply Latin Hypercube Sampling (LHS) transformation
    for i in range(sample_size):
        uniform_random[i] = (1 / sample_size) * (i + uniform_random[i])

    # Map to the inverse CDF to obtain samples
    samples = inversed_cdf_function(uniform_random)

    # Shuffle samples to remove any ordering bias
    np.random.shuffle(samples)

    return samples


def read_rainfall_data_inverse(model_name, year_name, sample_size, 
                               return_period=10, source_file='precipitation/projection/results/', lhs=True, seed=0):
    """
    Reads projection results from an RData file for a given climate model and year, 
    then samples from the computed rainfall intensity distribution.

    Parameters:
    -----------
    model_name : str
        The name of the climate model.
    year_name : int or str
        The year of the projection.
    sample_size : int
        The number of samples to generate.
    return_period : int, optional
        The return period in years (default is 10 years).
    source_file : str, optional
        The base directory where projection results are stored (default is 'precipitation/projection/results/').
    lhs : bool, optional
        If True, uses Latin Hypercube Sampling (LHS) for sampling. If False, uses inverse transform sampling (default is True).

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing sampled rainfall intensities (in/hr) along with metadata.

    Notes:
    ------
    - The function reads `.RData` files containing non-stationary GEV parameters (`xi_chain`, `muproj_chain`, `sigma_chain`).
    - Computes rainfall intensity using the `myreturnlevel` function.
    - Uses either Latin Hypercube Sampling (`inverse_transform_lhs`) or regular inverse transform sampling (`inverse_transform`).
    - Returns a DataFrame with the sampled intensities, associated metadata (year, return period, and model name).
    """
    # Construct the file path
    file_name = f"{source_file}{model_name}/nonstat_param{year_name}.RData"

    # Read the RData file
    rainfall_data = pyreadr.read_r(file_name)
    
    # Extract GEV parameters
    xi_df = rainfall_data['xi_chain']
    mu_df = rainfall_data['muproj_chain']
    sigma_df = rainfall_data['sigma_chain']

    # Compute rainfall intensity using the return level function
    rainfall_df = myreturnlevel(return_period, mu_df, sigma_df, xi_df)

    # Perform sampling using the chosen method
    if lhs:
        rainfall_samples = inverse_transform_lhs(inverse_cdf(rainfall_df), sample_size, seed=seed)
    else:
        rainfall_samples = inverse_transform(inverse_cdf(rainfall_df), sample_size, seed=seed)

    # Create the result DataFrame
    result_df = pd.DataFrame({'intensity_in_hr': rainfall_samples})
    result_df['year'] = year_name
    result_df['return_period'] = return_period
    result_df['model_name'] = model_name
    result_df['sample_id'] = list(range(sample_size))

    return result_df


def inverse_cdf(value_list):
    """
    Computes the inverse cumulative distribution function (inverse CDF) 
    from a given list of values, approximating the probability distribution function (PDF).

    Parameters:
    -----------
    value_list : array-like
        A list or array of values representing an empirical distribution.
        These values are assumed to approximate the probability distribution.

    Returns:
    --------
    callable
        A piecewise linear function (`interp1d`) that serves as the inverse CDF.

    Notes:
    ------
    - The function first sorts the input values to create an empirical cumulative distribution function (CDF).
    - It constructs the inverse CDF by interpolating between sorted values.
    - `bounds_error=False` ensures that values outside the input range return the min/max values.
    """
    # Sort the values to construct the empirical CDF
    sorted_values = np.sort(value_list).flatten()

    # Compute empirical CDF values
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    # Create an inverse CDF function using piecewise linear interpolation
    inverse_cdf_func = interp1d(cdf, sorted_values, bounds_error=False, 
                                 fill_value=(sorted_values[0], sorted_values[-1]))

    return inverse_cdf_func





def precipitation_sampling_per_model(model_name, year_name, num_sample, 
                                     num_resample, return_period, folder='samples/pr/', source_file='precipitation/projection/results/', seeds=1):
    """
    Samples precipitation data for a given climate model and year, and stores resampled datasets.

    Parameters:
    -----------
    model_name : str
        Name of the climate model.
    year_name : int or str
        The projection year.
    num_sample : int
        The number of samples per resample.
    num_resample : int
        The number of times to resample.
    return_period : int
        The return period in years for rainfall intensity estimation.
    folder : str, optional
        The base directory for storing sampled precipitation data (default is 'samples/pr/').

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing all resampled precipitation data for the given model.

    Notes:
    ------
    - The function generates multiple resamples of precipitation data.
    - Each resample is saved in its own subfolder within the model's directory.
    - The function writes the sampled data to `.dat` files and a CSV summary.
    """
    # Construct the model's folder path
    model_folder = os.path.join(folder, model_name)
    os.makedirs(model_folder, exist_ok=True)

    resample_df = []
    for seed in range(seeds):
        for i in range(num_resample):
            # Read and sample rainfall data
            df = read_rainfall_data_inverse(model_name, year_name, num_sample, 
                                            return_period=return_period, 
                                            source_file=source_file, lhs=True, seed=seed)
            df['seed'] = seed
            df['resample_id'] = i
            resample_df.append(df)

            # Create a folder for each resample
            resample_folder = os.path.join(model_folder, f"resample{i}_seed_{seed}")
            os.makedirs(resample_folder, exist_ok=True)

            # Write the sampled data to .dat files
            rtd.write_dat_file_given_name(rainfall_data=df, name_column='sample_id', 
                                        folder_name=resample_folder, start_time='01/01/2025 00:00')

    # Combine all resampled data into a single DataFrame
    model_df = pd.concat(resample_df, ignore_index=True)
    
    # Save the combined resampled data as a CSV file
    model_df.to_csv(os.path.join(folder, f"{model_name}.csv"), index=False)

    return model_df


def precipitation_sampling(num_sample, num_resample, return_period, 
                           climate_model, year_name, folder='samples/pr/', source_file='precipitation/projection/results/', seeds=1):
    """
    Samples precipitation data for multiple climate models and stores the results.

    Parameters:
    -----------
    num_sample : int
        The number of samples per resample.
    num_resample : int
        The number of times to resample.
    return_period : int
        The return period in years for rainfall intensity estimation.
    climate_model : list of str
        A list of climate models to process.
    year_name : int or str
        The projection year.
    folder : str, optional
        The base directory for storing sampled precipitation data (default is 'samples/pr/').

    Returns:
    --------
    dict
        A dictionary where each key is a climate model name, and the value is a DataFrame 
        containing the resampled precipitation data for that model.

    Notes:
    ------
    - Iterates over a list of climate models and calls `precipitation_sampling_per_model` for each.
    - Ensures that the output folder exists before starting the sampling process.
    - Returns a dictionary of DataFrames for each climate model.
    """
    os.makedirs(folder, exist_ok=True)

    pre_df = {}

    for model in climate_model:
        df = precipitation_sampling_per_model(model_name=model, year_name=year_name, 
                                              num_sample=num_sample, num_resample=num_resample, 
                                              return_period=return_period, folder=folder, source_file=source_file, seeds=seeds)
        pre_df[model] = df

    return pre_df  