"""
Brute Force Optimization for Pipe Network Design.

This script performs an exhaustive search over all possible pipe diameter combinations
to find the Pareto-optimal solutions for pipe network design under uncertainty.

Usage:
    python 4_BruteForce.py [--setup SETUP] [--udd_results PATH] [--network NETWORK]
                           [--swmm_path PATH] [--output NAME] [--num_total_sample N]
                           [--n_obj N] [--parallel] [--num_processes N]
                           [--rel_thresholds FLOAT] [--vol_thresholds FLOAT]

Arguments:
    --setup: Path to UDD setup JSON file (default: data/processed/setup.json)
    --udd_results: Path to UDD results directory (default: results/SOWs)
    --network: Network to be evaluated (default: 5)
    --swmm_path: Base directory for SWMM input files (SOWs) (default: results/SOWs)
    --output: Output folder name (default: BruteForce)
    --num_total_sample: Number of samples for uncertainty evaluation (default: 1000)
    --n_obj: Number of objectives, 2 or 4 (default: 4)
    --parallel: Run in parallel mode (flag)
    --num_processes: Number of parallel processes to use (default: 60)
    --rel_thresholds: Reliability threshold for evaluation (default: 80)
    --vol_thresholds: Volume threshold for evaluation; if None, uses value from UDD results
        (default: None)

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import argparse
import json
import itertools
import time
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure the parent directory is in the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pnetwork.sow as sow
import pnetwork.utils as utils
import pnetwork.ga as ga
import pnetwork.load_pipe as load_pipe
from pnetwork.evaluation import design_evaluation

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Brute force optimization for pipe network design.")

    parser.add_argument("--setup", type=str, default='data/processed/setup.json',
                        help="Path to UDD setup JSON file")
    parser.add_argument("--udd_results", type=str, default="results/SOWs",
                        help="Path to UDD results directory")
    parser.add_argument("--network", type=str, default="5",
                        help="Network to be evaluated (e.g. --network 5)")
    parser.add_argument("--swmm_path", type=str, default='results/SOWs',
                        help="Base directory for SWMM input files (SOWs)")
    parser.add_argument("--output", type=str, default="BruteForce", 
                        help="Output folder name")
    parser.add_argument("--num_total_sample", type=int, default=1000,
                        help="Number of samples for uncertainty evaluation (default: 1000)")
    parser.add_argument("--n_obj", type=int, default=4, 
                        help="Number of objectives (2 or 4)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel mode")
    parser.add_argument("--num_processes", type=int, default=60,
                        help="Number of parallel processes to use (default: 60)")
    parser.add_argument("--rel_thresholds", type=float, default=80,
                        help="Reliability threshold for evaluation (default: 80)")
    parser.add_argument("--vol_thresholds", type=float, default=None,
                        help="Volume threshold for evaluation (default: None, will use value from UDD results)")
    return parser.parse_args()

def generate_all_candidates(size_list_per_pipe):
    """
    Generate all possible pipe diameter combinations.

    Args:
        size_list_per_pipe (list): List of available pipe sizes for each pipe.

    Returns:
        np.ndarray: Array containing all possible diameter combinations.
    """
    combinations = itertools.product(*size_list_per_pipe)
    return np.array(list(combinations))

def evaluate_candidates_brute_force(candidates, network_info, inp_folder, file_name, 
                                   num_total_sample, num_scenario, n_obj=4, num_pro=8, 
                                   parallel=True, rel_thresholds=80, vol_thresholds=5000):
    """
    Evaluate all candidates using brute force approach.

    Args:
        candidates (np.ndarray): Array of candidate designs.
        network_info (dict): Network information.
        inp_folder (str): Path to SWMM input files folder.
        file_name (list): List of input file names.
        num_total_sample (int): Number of samples for evaluation.
        num_scenario (int): Number of scenarios.
        n_obj (int): Number of objectives (2 or 4).
        num_pro (int): Number of processes for parallel execution.
        parallel (bool): Whether to use parallel processing.
        rel_thresholds (float): Reliability threshold for evaluation.
        vol_thresholds (float): Volume threshold for evaluation.

    Returns:
        list: List of objective values for each candidate.
    """
    results = []
    for candidate in tqdm(candidates, desc="Evaluating candidates", disable=True):
        result = design_evaluation(
            candidate, network_info, inp_folder, file_name, 
            num_total_sample, num_scenario, n_obj, num_pro, 
            save=False, save_name='', save_path='eval_results/', 
            parallel=parallel, rel_thresholds=rel_thresholds, vol_thresholds=vol_thresholds
        )
        results.append(result)
    return results

def brute_force_optimization(setup, udd_results, network, swmm_path, output,
                           num_total_sample=1000, n_obj=4, parallel=True, num_processes=32, rel_thresholds=80, vol_thresholds=None):
    """
    Perform brute force optimization for pipe network design.

    Args:
        setup (str): Path to setup JSON file.
        udd_results (str): Path to UDD results directory.
        network (str): Network identifier.
        swmm_path (str): Base directory for SWMM input files.
        output (str): Output folder name.
        num_total_sample (int): Number of samples for uncertainty evaluation.
        n_obj (int): Number of objectives (2 or 4).
        parallel (bool): Whether to use parallel processing.
        num_processes (int): Number of parallel processes.
        rel_thresholds (float): Reliability threshold for evaluation.
        vol_thresholds (float): Volume threshold for evaluation.
    """
    # Load setup data
    setup_data = utils.load_json(setup)
    commercial_diameter = setup_data["commercial_diameter"]
    climate_models = setup_data["climate_models"]
    aging_models = setup_data["aging_models"]
    num_scenario = len(climate_models) * len(aging_models)

    # Set up paths
    udd_results_path = os.path.join(udd_results, network, "UDD", "udd.json")
    ini_swmm_path = os.path.join(udd_results, network, f'{network}_udd.inp')
    swmm_folder = os.path.join(swmm_path, network)
    swmm_inp_folder = os.path.join(swmm_folder, "inp_files")
    inp_file_name_path = os.path.join(swmm_folder, "inp_files_list.txt")
    output_folder = os.path.join(swmm_folder, f'{output}_{n_obj}')
    os.makedirs(output_folder, exist_ok=True)

    # Load UDD results
    udd_data = utils.load_json(udd_results_path)
    results_min_diameter = udd_data['results_min_diameter']
    reference_point = udd_data['reference_point']
    if vol_thresholds is None:
        vol_thresholds = udd_data['flooding_volume_threshold']
    
    # Set minimization directions and reference point based on number of objectives
    mins_direction = [-1, -1, 1, 1] if n_obj == 4 else [-1, 1]
    if n_obj == 2:
        reference_point = np.array([reference_point[0], reference_point[-1]])

    # Load network information
    swmm_info, network_info = load_pipe.read_pipe(ini_swmm_path, ini_swmm_path[:-3] + 'rpt')

    # Determine pipe size choices for each pipe
    pipe_size_per_pipe = ga.size_choice_per_pipe_udd(
        num_pipe=network_info['num_pipe'],
        min_diameters=results_min_diameter,
        max_pipe_size=network_info['pipe_max_size'],
        size_choice=commercial_diameter,
        num_choice_small=2,
        num_choice_large=4
    )

    time_start = time.time()
    # Generate all possible designs
    # print("Generating all possible designs...")
    designs = generate_all_candidates(pipe_size_per_pipe)
    # print(f"Total number of designs: {len(designs)}")

    # Perform brute force evaluation
    # print("Starting brute force evaluation...")
    
    
    results = evaluate_candidates_brute_force(
        designs, network_info, swmm_inp_folder,
        sow.read_inp_files_list(inp_file_name_path),
        num_total_sample, num_scenario, n_obj=n_obj,
        num_pro=num_processes, parallel=parallel, rel_thresholds=rel_thresholds, vol_thresholds=vol_thresholds
    )
    
    
    # Create DataFrame with results
    header = [f"x{i+1}" for i in range(network_info['num_pipe'])] + [f"f{i+1}" for i in range(n_obj)]
    df = pd.DataFrame(np.hstack([designs, np.array(results)]), columns=header)
    
    # Save results
    df_path = os.path.join(output_folder, "bf_results.csv")
    df.to_csv(df_path, index=False)
    
    # Find non-dominated solutions
    df_clean_path = utils.clean_pf(df_path, n_obj, mins=mins_direction)

    time_end = time.time()
    runtime = time_end - time_start
    
    # Calculate hypervolume
    hv = utils.calculate_hv_from_df_pf(df_clean_path, reference_point, mins=mins_direction)
    
    # print(f"Brute force optimization completed in {runtime:.2f} seconds")
    # print(f"Hypervolume: {hv}")
    
    # Save summary results
    summary_results = {
        "hypervolume": hv,
        "runtime_seconds": runtime,
        "total_designs_evaluated": len(designs),
        "num_objectives": n_obj,
        "num_total_samples": num_total_sample,
        "num_scenarios": num_scenario
    }
    
    with open(os.path.join(output_folder, "BruteForce.json"), "w") as f:
        json.dump(summary_results, f, indent=4)
    
    # print(f"Results saved to {output_folder}")

if __name__ == "__main__":
    args = parse_arguments()
    brute_force_optimization(
        setup=args.setup,
        udd_results=args.udd_results,
        network=args.network,
        swmm_path=args.swmm_path,
        output=args.output,
        num_total_sample=args.num_total_sample,
        n_obj=args.n_obj,
        parallel=args.parallel,
        num_processes=args.num_processes,
        rel_thresholds=args.rel_thresholds,
        vol_thresholds=args.vol_thresholds
    )
