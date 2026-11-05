"""
Functions not specific to a single module, but used across multiple modules in the pipe network project.

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""
import json
import numpy as np
import pandas as pd
import os
from pymoo.indicators.hv import Hypervolume
from .load_pipe import read_pipe


def get_total_pipe_length(swmm_file_path):
    """
    Calculate the total length of all pipes in the SWMM file.

    Args:
        swmm_file_path (str): Path to the SWMM input file.
    Returns:
        tuple: A list of individual pipe lengths and the total pipe length.
    """
    # Read the pipe data from the SWMM file
    swmm_info, network_info = read_pipe(swmm_file_path, swmm_file_path[:-3] + 'rpt')

    # Extract lengths and convert to float
    pipe_lengths = network_info['pipe_length']

    # Calculate total length
    total_length = sum(pipe_lengths)

    return pipe_lengths, total_length

def get_avg_area(swmm_file_path):
    """
    Calculate the average area of subcatchments in the SWMM file.

    Args:
        swmm_file_path (str): Path to the SWMM input file.
    Returns:
        float: Average area (ft^2) of subcatchments.
    """
    # Read the pipe data from the SWMM file
    swmm_info, network_info = read_pipe(swmm_file_path, swmm_file_path[:-3] + 'rpt')

    # Get the average area from the network info
    avg_area = network_info['avg_area']

    # From Acres to ft^2
    avg_area = avg_area * 43560

    return avg_area

def load_json(file_path):
    """
    Load pipe data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Loaded data.
    """
    with open(file_path, "r") as file:
        return json.load(file)
    

def find_non_dominated(points):
    """
    Identifies the non-dominated points among a set of points.

    Args:
        points (np.ndarray): 2D array where each row is a point in objective space.

    Returns:
        List[int]: Indices of non-dominated points.
    """
    num_points = points.shape[0]
    dominated = np.zeros(num_points, dtype=bool)

    for i in range(num_points):
        for j in range(num_points):
            if i != j and all(points[j] <= points[i]) and any(points[j] < points[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]

def clean_pf(file_path, n_obj, mins = [-1,1], gens = None, drop_duplicates_columns = None):
    """
    Reads a Pareto front DataFrame from a file and finds all non-dominated points.

    Args:
        file_path (str): Path to the `df_pf` file (CSV format).
        n_obj (int): Number of objectives (assumed to be the last n_obj columns).
        mins (list, optional): List indicating if the objectives are to be minimized (1) or maximized (-1).
        gens (int, optional): If provided, only consider the last `gens` generations for non-dominated sorting.

    Returns:
        pd.DataFrame: A new DataFrame containing only the non-dominated rows.
    """
    # 1. Read the Pareto front DataFrame
    df_pf = pd.read_csv(file_path)

    # 2. Extract the objective values (last n_obj columns)
    if gens is not None and 'generation' in df_pf.columns:
        max_gen = df_pf['generation'].max()
        df_pf = df_pf[df_pf['generation'] >= max_gen - gens + 1].reset_index(drop=True)
        objectives = df_pf.iloc[:, -n_obj-1:-1].values # take the last n_obj columns before 'generation'
    else:
        objectives = df_pf.iloc[:, -n_obj:].values

    for i in range(n_obj):
        if mins[i] == -1:
            objectives[:, i] = -objectives[:, i]
        

    # 3. Find non-dominated indices
    idx_nd = find_non_dominated(objectives)

    # 4. Filter the DataFrame
    new_df = df_pf.iloc[idx_nd].reset_index(drop=True)

    for i in range(n_obj):
        if mins[i] == -1:
            new_df.iloc[:, -n_obj + i] = -new_df.iloc[:, -n_obj + i]

    if drop_duplicates_columns is not None:
        new_df = new_df.drop_duplicates(subset=drop_duplicates_columns).reset_index(drop=True)
    else:
        new_df = new_df.drop_duplicates().reset_index(drop=True)

    # 5. Save to a new CSV alongside the original
    base, ext = os.path.splitext(file_path)
    out_path = f"{base}_cleaned{ext}"
    new_df.to_csv(out_path, index=False)
    # print(f"Saved cleaned Pareto front ({len(new_df)} points) to: {out_path}")

    return out_path





def calculate_hv_from_df_pf(file_path, ref_point, mins = [-1,1], columns = None):
    """
    Reads a Pareto front DataFrame from a file and calculates the hypervolume (HV).

    Args:
        file_path (str): Path to the `df_pf` file (CSV format).
        ref_point (list): Reference point for hypervolume calculation.

    Returns:
        float: The calculated hypervolume.
    """
    # Read the Pareto front DataFrame
    df_pf = pd.read_csv(file_path)

    # Extract the objective values (assuming they are the last columns in the DataFrame)
    # Adjust the column slicing if necessary
    if columns is None:
        objectives = df_pf.iloc[:, -len(ref_point):].values
    else:
        objectives = df_pf.iloc[:, columns].values

    for i in range(len(ref_point)):
        if mins[i] == -1:
            objectives[:, i] = -objectives[:, i]

    # Calculate the hypervolume
    hv = Hypervolume(ref_point=ref_point)
    hypervolume_value = hv.do(objectives)

    return hypervolume_value



def _distance(file_a, file_b, reliability_ind, cost_ind, negative = [True,True], rel_range = [70,100]):
    """
    Calculate the 2D distance between two Pareto fronts for cost and reliability.

    Args:
        file_a (str): Path to the first Pareto front file (CSV format).
        file_b (str): Path to the second Pareto front file (CSV format).
        reliability_ind (int): Index of the reliability column in the DataFrame.
        cost_ind (int): Index of the cost column in the DataFrame.
        negative (list): List indicating if the objectives are to be minimized (True) or maximized (False).
        rel_range (list): List containing the min and max reliability values to consider

    Returns:
        float: The calculated 2D distance.
    """
    # Read the Pareto front DataFrames
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    # Extract the objective values (assuming they are the last two columns in the DataFrame)
    objectives_a_rel = df_a.iloc[:, reliability_ind].values
    objectives_a_cost = df_a.iloc[:, cost_ind].values
    objectives_b_rel = df_b.iloc[:, reliability_ind].values
    objectives_b_cost = df_b.iloc[:, cost_ind].values
    if negative[0]:
        objectives_a_rel = -objectives_a_rel
    if negative[1]:
        objectives_b_rel = -objectives_b_rel

    # Resort by reliability, then by cost (both ascending)
    a_sort_idx = np.lexsort((objectives_a_cost, objectives_a_rel))
    objectives_a_rel = objectives_a_rel[a_sort_idx]
    objectives_a_cost = objectives_a_cost[a_sort_idx]

    b_sort_idx = np.lexsort((objectives_b_cost, objectives_b_rel))
    objectives_b_rel = objectives_b_rel[b_sort_idx]
    objectives_b_cost = objectives_b_cost[b_sort_idx]
    

    # Create a list of reliability thresholds from 0 to 100 with interval 1
    reliability_range = np.arange(rel_range[0], rel_range[1]+1, 1)
    differences_cost = []
    percentages_cost = []
    for rel_threshold in reliability_range:
        # For A: find the solution with the lowest reliability >= threshold
        mask_a = objectives_a_rel >= rel_threshold
        if np.any(mask_a):
            idx_a = np.argmin(objectives_a_rel[mask_a])
            indices_a = np.where(mask_a)[0]
            idx_a_final = indices_a[idx_a]
            cost_a = objectives_a_cost[idx_a_final]
        else:
            cost_a = np.nan

        # For B: find the solution with the lowest reliability >= threshold
        mask_b = objectives_b_rel >= rel_threshold
        if np.any(mask_b):
            idx_b = np.argmin(objectives_b_rel[mask_b])
            indices_b = np.where(mask_b)[0]
            idx_b_final = indices_b[idx_b]
            cost_b = objectives_b_cost[idx_b_final]
        else:
            cost_b = np.nan

        # Compare the difference in cost
        if not np.isnan(cost_a) and not np.isnan(cost_b):
            differences_cost.append(cost_b - cost_a)
            percentages_cost.append((cost_b - cost_a) / cost_a * 100 if cost_a != 0 else np.nan)
        else:
            differences_cost.append(np.nan)
    # differences_cost is a list of cost differences for each reliability threshold
    return reliability_range, differences_cost, percentages_cost


def get_cost_for_reliability(file_path, min_reliability, reliability_ind, cost_ind, negative=[True, True]):
    """
    Calculate the minimum cost needed to achieve a given minimum reliability requirement.

    Args:
        file_path (str): Path to the Pareto front file (CSV format).
        min_reliability (float): Minimum reliability requirement to achieve.
        reliability_ind (int): Index of the reliability column in the DataFrame.
        cost_ind (int): Index of the cost column in the DataFrame.
        negative (list): List indicating if the objectives are to be minimized (True) or maximized (False).

    Returns:
        tuple: (cost, actual_reliability) where:
            - cost (float): Minimum cost to achieve the reliability requirement, or np.nan if not achievable
            - actual_reliability (float): Actual reliability of the selected solution, or np.nan if not achievable
    """
    # Read the Pareto front DataFrame
    df = pd.read_csv(file_path)

    # Extract the objective values
    objectives_rel = df.iloc[:, reliability_ind].values
    objectives_cost = df.iloc[:, cost_ind].values
    
    # Apply negative transformation if needed
    if negative[0]:
        objectives_rel = -objectives_rel
    if negative[1]:
        objectives_cost = -objectives_cost

    # Find solutions that meet the minimum reliability requirement
    mask = objectives_rel >= min_reliability
    
    if not np.any(mask):
        # No solution meets the reliability requirement
        return np.nan, np.nan
    
    # Among solutions that meet the reliability requirement, find the one with minimum cost
    feasible_costs = objectives_cost[mask]
    feasible_reliabilities = objectives_rel[mask]
    
    min_cost_idx = np.argmin(feasible_costs)
    min_cost = feasible_costs[min_cost_idx]
    actual_reliability = feasible_reliabilities[min_cost_idx]
    
    # Transform back if negative was applied
    if negative[1]:
        min_cost = -min_cost
    if negative[0]:
        actual_reliability = -actual_reliability
    
    return min_cost, actual_reliability


def get_cost_for_reliability_range(file_path, reliability_range, reliability_ind, cost_ind, negative=[True, True]):
    """
    Calculate the minimum cost needed for a range of reliability requirements.

    Args:
        file_path (str): Path to the Pareto front file (CSV format).
        reliability_range (array-like): Array of reliability thresholds to evaluate.
        reliability_ind (int): Index of the reliability column in the DataFrame.
        cost_ind (int): Index of the cost column in the DataFrame.
        negative (list): List indicating if the objectives are to be minimized (True) or maximized (False).

    Returns:
        tuple: (reliability_thresholds, costs, actual_reliabilities) where:
            - reliability_thresholds (array): Input reliability requirements
            - costs (list): Minimum costs for each reliability requirement (np.nan if not achievable)
            - actual_reliabilities (list): Actual reliabilities achieved (np.nan if not achievable)
    """
    costs = []
    actual_reliabilities = []
    
    for rel_threshold in reliability_range:
        cost, actual_rel = get_cost_for_reliability(
            file_path, rel_threshold, reliability_ind, cost_ind, negative
        )
        costs.append(cost)
        actual_reliabilities.append(actual_rel)
    
    return reliability_range, costs, actual_reliabilities