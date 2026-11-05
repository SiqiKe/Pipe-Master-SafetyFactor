"""
Safety Factor methods for pipe network design under uncertainty.

Main Functions:
- safety_factors: Generate system-wide safety factor designs
- safety_factors_functional: Generate functional safety factor designs based on pipe characteristics
- sf_result_uncertainty: Evaluate safety factor designs under uncertainty using grid search
- sf_functional_result_uncertainty_ssf: Evaluate functional safety factor designs under uncertainty with system-wide results using grid search

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import numpy as np
import csv
import os
import itertools

from .ga import get_diameter_all_per_pipe
from .evaluation import (overflow_swmm, system_cost, design_evaluation)
import pnetwork.utils as utils
from .udd import get_diameter

def safety_factors(num_pipe, baseline_design, size_choice, sf_range=None, num_points=None):
    """
    Generate system-wide safety factor designs by scaling baseline design. By default, it considers all unique multipliers derived from the available pipe sizes. If sf_range and num_points are provided, it will sample safety factors within the specified range.
    
    Args:
        num_pipe (int): Number of pipes in the network.
        baseline_design (list): Baseline pipe diameters.
        size_choice (list): Available pipe sizes for each pipe.
        sf_range (list, optional): Range of safety factors to consider. Defaults to None.
        num_points (int, optional): Number of points to sample within the safety factor range. Defaults to None.
    
    Returns:
        dict: Dictionary mapping safety factors to corresponding designs.
    """
    
    
    designs = {}
    tracker = []
    
    if sf_range is not None and num_points is not None:
        sampled_multipliers = np.linspace(sf_range[0], sf_range[1], num_points)
        for multiplier in sampled_multipliers:
            scaled_design = list(get_diameter_all_per_pipe(size_choice, np.array(baseline_design) * multiplier))
            
            # Only add unique designs
            if scaled_design not in tracker:
                tracker.append(scaled_design)
                designs[multiplier] = scaled_design
    else:
        # Calculate possible multipliers for each baseline value
        possible_multipliers = []
        for i in range(num_pipe):
            baseline = baseline_design[i] if baseline_design[i] > 0 else 0.0001
            # Calculate multipliers that scale baseline to target values
            multipliers = [target / baseline for target in size_choice[i]]
            possible_multipliers.append(multipliers)
        
        # Get unique multipliers across all pipes
        unique_multipliers = sorted(set(value for sublist in possible_multipliers for value in sublist))
        for multiplier in unique_multipliers:
            scaled_design = list(get_diameter_all_per_pipe(size_choice, np.array(baseline_design) * multiplier))
            
            # Only add unique designs
            if scaled_design not in tracker:
                tracker.append(scaled_design)
                designs[multiplier] = scaled_design
    
    return designs

def FunctionSF_pipe(min_diameter, para_length, para_strahler, mean_length, mean_strahler, length, strahler, para_c = 1):
    return min_diameter * (para_length * (length - mean_length) / mean_length + para_strahler * (strahler - mean_strahler) / mean_strahler +  para_c)

def FunctionSF(commercial_diameter_per_pipe, min_diameter_list, sf, pipe_length, pipe_strahler):
    para_length, para_strahler, para_c= sf[0], sf[1], sf[2]
    mean_length = np.mean(pipe_length)
    mean_strahler = np.mean(pipe_strahler)

    num = len(min_diameter_list)
    results = np.zeros(num)
    for i in range(num):
        diameter = FunctionSF_pipe(min_diameter_list[i], para_length, para_strahler, mean_length, mean_strahler, pipe_length[i], pipe_strahler[i], para_c)
        temp =  get_diameter(commercial_diameter_per_pipe[i], diameter)
        if temp:
            results[i] = temp[1]
        else:
            results[i] = max(commercial_diameter_per_pipe[i])
    return results

def safety_factors_functional(num_pipe, baseline_design, size_choice, pipe_length, pipe_strahler, 
                            sf_range=[[0, 2], [0, 2], [0.2, 5.0]], a_num=100, b_num=100, c_num=1000):
    """
    Generate functional safety factor designs based on pipe characteristics.
    
    Args:
        num_pipe (int): Number of pipes in the network.
        baseline_design (list): Baseline pipe diameters.
        size_choice (list): Available pipe sizes for each pipe.
        pipe_length (list): Length of each pipe.
        pipe_strahler (list): Strahler number for each pipe.
        sf_range (list): Parameter ranges for functional safety factors [[a_min, a_max], [b_min, b_max], [c_min, c_max]].
        a_num (int): Number of grid points for parameter a.
        b_num (int): Number of grid points for parameter b.
        c_num (int): Number of grid points for parameter c.
    
    Returns:
        tuple: (sf_params_list, designs) - Safety factor parameters and corresponding designs.
    """
    # Generate parameter combinations
    params = list(itertools.product(
        np.linspace(sf_range[0][0], sf_range[0][1], a_num),
        np.linspace(sf_range[1][0], sf_range[1][1], b_num),
        np.linspace(sf_range[2][0], sf_range[2][1], c_num)
    ))
    
    sf_params_list = []
    designs = []
    costs = []
    
    for param in params:
        design = list(FunctionSF(size_choice, baseline_design, param, pipe_length, pipe_strahler))
        
        # Only add unique designs
        if design not in designs:
            sf_params_list.append(param)
            cost = system_cost(design, pipe_length)
            designs.append(design)
            costs.append(cost)
    
    # Sort by cost (ascending order)
    sorted_indices = np.argsort(costs)
    designs = [designs[i] for i in sorted_indices]
    sf_params_list = [sf_params_list[i] for i in sorted_indices]
    costs = [costs[i] for i in sorted_indices]
    
    return sf_params_list, designs

def safety_factors_functional_ssf(num_pipe, baseline_design, ssf, size_choice, pipe_length, pipe_strahler, 
                             a_num=10, b_num=10, a_range=[0, 1], b_range=[0, 1]):
    """
    Generate functional safety factor designs based on pipe characteristics with system-wide results. Instead of searching all
    parameters, only search parameters that are around the given system safety factor (ssf).

    Args:
        num_pipe (int): Number of pipes in the network.
        baseline_design (list): Baseline pipe diameters.
        ssf (list): System safety factor results (a list of safety factors).
        size_choice (list): Available pipe sizes for each pipe.
        pipe_length (list): Length of each pipe.
        pipe_strahler (list): Strahler number for each pipe.
        a_num (int): Number of grid points for parameter a.
        b_num (int): Number of grid points for parameter b.
        a_range (list): Range for parameter a [min, max].
        b_range (list): Range for parameter b [min, max].
        
    
    Returns:
        tuple: (sf_params_list, designs) - Safety factor parameters and corresponding designs.
    """
    # Generate parameter combinations
    params = list(itertools.product(
        np.linspace(a_range[0], a_range[1], a_num),
        np.linspace(b_range[0], b_range[1], b_num),
        ssf
    ))
    
    sf_params_list = []
    designs = []
    costs = []
    
    for param in params:
        design = list(FunctionSF(size_choice, baseline_design, param, pipe_length, pipe_strahler))
        
        # Only add unique designs
        if design not in designs:
            sf_params_list.append(param)
            cost = system_cost(design, pipe_length)
            designs.append(design)
            costs.append(cost)
    
    # Sort by cost (ascending order)
    sorted_indices = np.argsort(costs)
    designs = [designs[i] for i in sorted_indices]
    sf_params_list = [sf_params_list[i] for i in sorted_indices]
    costs = [costs[i] for i in sorted_indices]
    
    return sf_params_list, designs

# def sf_functional_result_uncertainty(network_info, sf_params_list, designs, inp_folder, file_name, 
#                                    output_folder, reference_point, n_obj=2, num_pro=8, num_scenario=8, 
#                                    save=False, save_name='', save_path='eval_results/', parallel=True, 
#                                    coarse_size=10, fine_size=100, later_size=20):
#     """
#     Evaluate functional safety factor designs under uncertainty using adaptive grid search.
    
#     Args:
#         network_info (dict): Network information.
#         sf_params_list (list): List of safety factor parameters.
#         designs (list): List of corresponding designs.
#         inp_folder (str): Path to SWMM input files folder.
#         file_name (list): List of input file names.
#         output_folder (str): Folder to save results.
#         reference_point (array): Reference point for hypervolume calculation.
#         n_obj (int): Number of objectives.
#         num_pro (int): Number of processes for parallel execution.
#         num_scenario (int): Number of scenarios.
#         save (bool): Whether to save intermediate results.
#         save_name (str): Prefix for saved file names.
#         save_path (str): Path to save intermediate results.
#         parallel (bool): Whether to use parallel processing.
#         coarse_size (int): Number of points in coarse grid search.
#         fine_size (int): Number of points in fine grid search.
#         later_size (int): Number of points in final grid search.
    
#     Returns:
#         str: Path to cleaned Pareto front results.
#     """
#     num_total_sample = len(file_name)
#     start_idx = 0
#     end_idx = len(sf_params_list) - 1
    
#     results = []
#     reliability_list = []
#     params_list = []

#     def eval_design(idx):
#         """Evaluate a single design by index."""
#         key, design = sf_params_list[idx], designs[idx]
        
#         # Check if already evaluated
#         if key in params_list:
#             return reliability_list[params_list.index(key)]
        
#         eval_result = design_evaluation(
#             design, network_info, inp_folder, file_name,
#             num_total_sample=num_total_sample, num_scenario=num_scenario, n_obj=n_obj, num_pro=num_pro,
#             save=save, save_name=save_name, save_path=save_path, parallel=parallel
#         )
        
#         reliability = eval_result[0]
#         results.append(list(key) + list(design) + list(eval_result))
#         reliability_list.append(reliability)
#         params_list.append(key)
#         return reliability

#     def grid_search_reliability(start_idx, end_idx, step=10):
#         """
#         Perform grid search over the given index range.
        
#         Returns:
#             tuple: (least_zero_idx, first_100_idx) - Indices of first 0% and 100% reliability designs.
#         """
#         least_zero_idx = None
#         first_100_idx = None
        
#         for idx in range(start_idx, min(end_idx + 1, len(sf_params_list)), step):
#             reliability = eval_design(idx)
#             if reliability == 0 and least_zero_idx is None:
#                 least_zero_idx = idx
#             if reliability == 100 and first_100_idx is None:
#                 first_100_idx = idx
        
#         return least_zero_idx, first_100_idx

#     # Three-stage adaptive grid search
    
#     # Stage 1: Coarse grid search
#     coarse_step = max(1, (end_idx - start_idx) // coarse_size)
#     coarse_zero_idx, coarse_100_idx = grid_search_reliability(start_idx, end_idx, step=coarse_step)
#     coarse_zero_idx = coarse_zero_idx if coarse_zero_idx is not None else start_idx
#     coarse_100_idx = coarse_100_idx if coarse_100_idx is not None else end_idx

#     # Stage 2: Fine grid search between coarse bounds
#     fine_step = max(1, (coarse_100_idx - coarse_zero_idx) // fine_size)
#     fine_zero_idx, fine_100_idx = grid_search_reliability(coarse_zero_idx, coarse_100_idx, step=fine_step)
#     fine_zero_idx = fine_zero_idx if fine_zero_idx is not None else coarse_zero_idx
#     fine_100_idx = fine_100_idx if fine_100_idx is not None else coarse_100_idx

#     # Stage 3: Final search in the middle-to-high reliability range
#     mid_idx = (fine_zero_idx + fine_100_idx) // 2
#     third_step = max(1, (fine_100_idx - mid_idx) // later_size)
#     grid_search_reliability(mid_idx, fine_100_idx, step=third_step)

#     # Save results to CSV
#     header = (['Para Length', 'Para Strahler', 'Para C'] + 
#               [f"x{i+1}" for i in range(network_info['num_pipe'])] + 
#               [f"f{i+1}" for i in range(n_obj)])
    
#     # Remove duplicates
#     seen = set()
#     unique_rows = []
#     for row in results:
#         row_tuple = tuple(row)
#         if row_tuple not in seen:
#             seen.add(row_tuple)
#             unique_rows.append(row)

#     results_path = os.path.join(output_folder, 'FunctionSF_results.csv')
#     with open(results_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#         writer.writerows(unique_rows)

#     # Set minimization directions
#     mins_direction = [-1, 1] if n_obj == 2 else [-1, -1, 1, 1]
    
#     # Find non-dominated solutions
#     df_clean_path = utils.clean_pf(results_path, n_obj, mins=mins_direction)
#     return df_clean_path

def sf_functional_result_uncertainty_ssf(network_info, ssf_designs, sf_params_list, designs, inp_folder, file_name, 
                                   output_folder, reference_point, n_obj=2, num_pro=8, num_scenario=8, 
                                   save=False, save_name='', save_path='eval_results/', parallel=True, 
                                   coarse_size=10, fine_size=100, later_size=20, rel_thresholds=80, vol_thresholds=500):
    """
    Evaluate functional safety factor designs under uncertainty using adaptive grid search.
    
    Args:
        network_info (dict): Network information.
        ssf_designs (DataFrame): DataFrame of system safety factor designs.
        sf_params_list (list): List of safety factor parameters.
        designs (list): List of corresponding designs.
        inp_folder (str): Path to SWMM input files folder.
        file_name (list): List of input file names.
        output_folder (str): Folder to save results.
        reference_point (array): Reference point for hypervolume calculation.
        n_obj (int): Number of objectives.
        num_pro (int): Number of processes for parallel execution.
        num_scenario (int): Number of scenarios.
        save (bool): Whether to save intermediate results.
        save_name (str): Prefix for saved file names.
        save_path (str): Path to save intermediate results.
        parallel (bool): Whether to use parallel processing.
        coarse_size (int): Number of points in coarse grid search.
        fine_size (int): Number of points in fine grid search.
        later_size (int): Number of points in final grid search.
    
    Returns:
        str: Path to cleaned Pareto front results.
    """
    num_total_sample = len(file_name)
    start_idx = 0
    end_idx = len(sf_params_list) - 1
    
    results = []
    reliability_list = []
    params_list = []

    def eval_design(idx):
        """Evaluate a single design by index."""
        key, design = sf_params_list[idx], designs[idx]
        
        # Check if already evaluated
        if key in params_list:
            return reliability_list[params_list.index(key)]
        
        eval_result = design_evaluation(
            design, network_info, inp_folder, file_name,
            num_total_sample=num_total_sample, num_scenario=num_scenario, n_obj=n_obj, num_pro=num_pro,
            save=save, save_name=save_name, save_path=save_path, parallel=parallel, rel_thresholds=rel_thresholds, vol_thresholds=vol_thresholds
        )
        
        reliability = eval_result[0]
        results.append(list(key) + list(design) + list(eval_result))
        reliability_list.append(reliability)
        params_list.append(key)
        return reliability

    def grid_search_reliability(start_idx, end_idx, step=10):
        """
        Perform grid search over the given index range.
        
        Returns:
            tuple: (least_zero_idx, first_100_idx) - Indices of first 0% and 100% reliability designs.
        """
        least_zero_idx = None
        first_100_idx = None
        
        for idx in range(start_idx, min(end_idx + 1, len(sf_params_list)), step):
            reliability = eval_design(idx)
            if reliability == 0 and least_zero_idx is None:
                least_zero_idx = idx
            if reliability == 100 and first_100_idx is None:
                first_100_idx = idx
        
        return least_zero_idx, first_100_idx

    # Three-stage adaptive grid search
    
    # Stage 1: Coarse grid search
    coarse_step = max(1, (end_idx - start_idx) // coarse_size)
    coarse_zero_idx, coarse_100_idx = grid_search_reliability(start_idx, end_idx, step=coarse_step)
    coarse_zero_idx = coarse_zero_idx if coarse_zero_idx is not None else start_idx
    coarse_100_idx = coarse_100_idx if coarse_100_idx is not None else end_idx

    # Stage 2: Fine grid search between coarse bounds
    fine_step = max(1, (coarse_100_idx - coarse_zero_idx) // fine_size)
    fine_zero_idx, fine_100_idx = grid_search_reliability(coarse_zero_idx, coarse_100_idx, step=fine_step)
    fine_zero_idx = fine_zero_idx if fine_zero_idx is not None else coarse_zero_idx
    fine_100_idx = fine_100_idx if fine_100_idx is not None else coarse_100_idx

    # Stage 3: Final search in the middle-to-high reliability range
    mid_idx = (fine_zero_idx + fine_100_idx) // 2
    third_step = max(1, (fine_100_idx - mid_idx) // later_size)
    grid_search_reliability(mid_idx, fine_100_idx, step=third_step)

    # Save results to CSV
    header = (['Para Length', 'Para Strahler', 'Para C'] + 
              [f"x{i+1}" for i in range(network_info['num_pipe'])] + 
              [f"f{i+1}" for i in range(n_obj)])
    
    # Add SSF designs to results with two 0s at the beginning to match parameter count
    for _, row in ssf_designs.iterrows():
        row_values = row.tolist()
        # Add two 0s at the beginning for Para Length and Para Strahler, then add the row values
        ssf_row = [0, 0] + row_values
        results.append(ssf_row)

    # Remove duplicates
    seen = set()
    unique_rows = []
    for row in results:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)

    results_path = os.path.join(output_folder, 'FunctionSF_results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(unique_rows)

    # Set minimization directions
    mins_direction = [-1, 1] if n_obj == 2 else [-1, -1, 1, 1]
    
    # Find non-dominated solutions
    df_clean_path = utils.clean_pf(results_path, n_obj, mins=mins_direction)
    return df_clean_path

def sf_result_uncertainty(network_info, designs, inp_folder, file_name, output_folder, reference_point, 
                         n_obj=2, num_pro=8, num_scenario=8, save=False, save_name='', 
                         save_path='eval_results/', parallel=True, sf_range=[0.2, 5.0], 
                         coarse_size=10, fine_size=100, later_size=20, rel_thresholds=80, vol_thresholds=500, grid_search=False):
    """
    Evaluate system safety factor designs under uncertainty using adaptive grid search.
    
    Args:
        network_info (dict): Network information.
        designs (dict): Dictionary mapping safety factors to designs.
        inp_folder (str): Path to SWMM input files folder.
        file_name (list): List of input file names.
        output_folder (str): Folder to save results.
        reference_point (array): Reference point for hypervolume calculation.
        n_obj (int): Number of objectives.
        num_pro (int): Number of processes for parallel execution.
        num_scenario (int): Number of scenarios.
        save (bool): Whether to save intermediate results.
        save_name (str): Prefix for saved file names.
        save_path (str): Path to save intermediate results.
        parallel (bool): Whether to use parallel processing.
        sf_range (list): Range of safety factors to evaluate [min, max].
        coarse_size (int): Number of points in coarse grid search.
        fine_size (int): Number of points in fine grid search.
        later_size (int): Number of points in final grid search.
        rel_thresholds (float): Reliability threshold for evaluation.
        vol_thresholds (float): Volume threshold for evaluation.
        grid_search (bool): Whether to perform adaptive grid search.
    Returns:
        str: Path to cleaned Pareto front results.
    """
    num_total_sample = len(file_name)
    
    # Filter and sort designs within the specified sf_range
    designs_range = {k: v for k, v in designs.items() if sf_range[0] <= k <= sf_range[1]}
    sorted_keys = sorted(designs_range.keys())
    sorted_designs = [designs_range[k] for k in sorted_keys]

    start_idx = 0
    end_idx = len(sorted_keys) - 1
    
    results = []
    reliability_list = []

    def eval_design(idx):
        """Evaluate a single design by index."""
        key, design = sorted_keys[idx], sorted_designs[idx]
        
        # Check if already evaluated
        existing_keys = [row[0] for row in results]
        if key in existing_keys:
            return reliability_list[existing_keys.index(key)]
        
        eval_result = design_evaluation(
            design, network_info, inp_folder, file_name,
            num_total_sample=num_total_sample, num_scenario=num_scenario, n_obj=n_obj, num_pro=num_pro,
            save=save, save_name=save_name, save_path=save_path, parallel=parallel, rel_thresholds=rel_thresholds, vol_thresholds=vol_thresholds
        )
        
        reliability = eval_result[0]
        results.append([key] + list(design) + list(eval_result))
        reliability_list.append(reliability)
        return reliability

    def grid_search_reliability(start_idx, end_idx, step=10):
        """
        Perform grid search over the given index range.
        
        Returns:
            tuple: (least_zero_idx, first_100_idx) - Indices of first 0% and 100% reliability designs.
        """
        least_zero_idx = None
        first_100_idx = None
        
        for idx in range(start_idx, min(end_idx + 1, len(sorted_keys)), step):
            reliability = eval_design(idx)
            if reliability == 0 and least_zero_idx is None:
                least_zero_idx = idx
            if reliability == 100 and first_100_idx is None:
                first_100_idx = idx
        
        return least_zero_idx, first_100_idx

    if grid_search:
        # Three-stage adaptive grid search
        
        # Stage 1: Coarse grid search
        coarse_step = max(1, (end_idx - start_idx) // coarse_size)
        coarse_zero_idx, coarse_100_idx = grid_search_reliability(start_idx, end_idx, step=coarse_step)
        coarse_zero_idx = coarse_zero_idx if coarse_zero_idx is not None else start_idx
        coarse_100_idx = coarse_100_idx if coarse_100_idx is not None else end_idx

        # Stage 2: Fine grid search between coarse bounds
        fine_step = max(1, (coarse_100_idx - coarse_zero_idx) // fine_size)
        fine_zero_idx, fine_100_idx = grid_search_reliability(coarse_zero_idx, coarse_100_idx, step=fine_step)
        fine_zero_idx = fine_zero_idx if fine_zero_idx is not None else coarse_zero_idx
        fine_100_idx = fine_100_idx if fine_100_idx is not None else coarse_100_idx

        # Stage 3: Final search in the middle-to-high reliability range
        mid_idx = (fine_zero_idx + fine_100_idx) // 2
        third_step = max(1, (fine_100_idx - mid_idx) // later_size)
        grid_search_reliability(mid_idx, fine_100_idx, step=third_step)

    else:
        # Evaluate all designs without grid search
        for idx in range(len(sorted_keys)):
            eval_design(idx)

    # Save results to CSV
    header = (['System SF'] + 
              [f"x{i+1}" for i in range(network_info['num_pipe'])] + 
              [f"f{i+1}" for i in range(n_obj)])
    
    # Remove duplicates
    seen = set()
    unique_rows = []
    for row in results:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)

    results_path = os.path.join(output_folder, 'sf_results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(unique_rows)

    # Set minimization directions
    mins_direction = [-1, 1] if n_obj == 2 else [-1, -1, 1, 1]
    
    # Find non-dominated solutions
    df_clean_path = utils.clean_pf(results_path, n_obj, mins=mins_direction)
    return df_clean_path



def sf_result_nouncertainty(network_info, designs):
    """
    Find the minimum safety factor design that achieves zero overflow (deterministic case).
    
    Args:
        network_info (dict): Network information.
        designs (dict): Dictionary mapping safety factors to designs.
    
    Returns:
        tuple: (safety_factor, design, cost) - Minimum safety factor, design, and cost.
    """
    designs_list = [[key, design] for key, design in designs.items()]
    return exponential_search(network_info, designs_list)

def exponential_search(network_info, designs_list):
    """
    Find the first design with zero overflow using exponential search followed by binary search.
    
    Args:
        network_info (dict): Network information.
        designs_list (list): List of [safety_factor, design] pairs sorted by safety factor.
    
    Returns:
        tuple: (safety_factor, design, cost) - First design achieving zero overflow.
    """
    # Check if the first design already achieves zero overflow
    first_overflow = overflow_swmm(
        network_info['num_pipe'], designs_list[0][1], 
        network_info['filepath'], network_info['pipe_name']
    )['system']
    
    if first_overflow <= 0:
        cost = system_cost(designs_list[0][1], network_info['pipe_length'])
        return designs_list[0][0], designs_list[0][1], cost

    n = len(designs_list)
    index = 1

    # Exponentially increase index while overflow > 0
    while index < n:
        overflow = overflow_swmm(
            network_info['num_pipe'], designs_list[index][1], 
            network_info['filepath'], network_info['pipe_name']
        )['system']
        
        if overflow <= 0:
            break
        index *= 2

    # Perform binary search in the valid range
    return binary_search(network_info, designs_list, index // 2, min(index, n - 1))

def binary_search(network_info, designs_list, low, high):
    """
    Binary search to find the first design with zero overflow.
    
    Args:
        network_info (dict): Network information.
        designs_list (list): List of [safety_factor, design] pairs.
        low (int): Lower bound index.
        high (int): Upper bound index.
    
    Returns:
        tuple: (safety_factor, design, cost) - First design achieving zero overflow.
    """
    first_zero = -1
    
    while low <= high:
        mid = (low + high) // 2
        overflow = overflow_swmm(
            network_info['num_pipe'], designs_list[mid][1], 
            network_info['filepath'], network_info['pipe_name']
        )['system']
        
        if overflow <= 0:
            first_zero = mid
            high = mid - 1  # Search for earlier occurrence
        else:
            low = mid + 1  # Move to higher safety factors

    # Calculate final results
    final_overflow = overflow_swmm(
        network_info['num_pipe'], designs_list[first_zero][1], 
        network_info['filepath'], network_info['pipe_name'], 
        file_suffix='_sf'
    )['system']
    
    cost = system_cost(designs_list[first_zero][1], network_info['pipe_length'])
    return designs_list[first_zero][0], designs_list[first_zero][1], cost
