"""
Sobol sensitivity analysis for pipe network designs.

This script performs Sobol sensitivity analysis to identify influential parameters and their
interactions in pipe network design performance under deep uncertainty.

Usage:
    python 8_Sobol_analysis.py [OPTIONS]

Arguments:
    --swmm_path: Base directory for SWMM input files (default: 'results/SOWs')
    --network: Network to analyze (default: 16)
    --setup: Path to UDD setup JSON file (default: 'data/processed/setup.json')
    --params_bounds_file: Path to Sobol parameters JSON file (default: 'data/processed/para_bounds.json')
    --samples_folder: Folder to save Sobol samples (default: 'results/SOWs_sobol')
    --num_sample: Number of Sobol samples (default: 2048)
    --parallel: Run in parallel mode (default: False)
    --num_processes: Number of parallel processes (default: 32)
    --per_scenario: Perform analysis per climate/aging scenario (default: False)
    --num_sample_per_scenario: Number of samples per scenario (default: 2048)
    --num_inp: Number of SWMM input files generated for parallel evaluation (if None, generated 5*num_processes)
    --seed: Random seed for reproducibility (default: 0)
    --design: Path to JSON file containing pipe sizes for a specific design (optional)

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import argparse
import itertools
from SALib.analyze.sobol import analyze
from SALib.sample import sobol
import numpy as np
import json
import os
import sys

# Ensure the parent directory is in the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pnetwork.utils as utils
import pnetwork.sow as sow
import pnetwork.load_pipe as load_pipe
from pnetwork.evaluation import overflow_swmm

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Sobol Sensitivity Analysis for Pipe Network Designs")
    parser.add_argument("--swmm_path", type=str, default='results/SOW_samples',
                        help="Base directory for SWMM input files")
    parser.add_argument("--network",  type=str, default="16", help="Network to be evaluated (e.g. --network 16)")
    parser.add_argument("--setup", type=str, default='data/processed/setup.json',
                        help="Path to UDD setup JSON file")
    parser.add_argument("--params_bounds_file", type=str, default='data/processed/para_bounds.json',
                        help="Path to Sobol parameters JSON file")
    parser.add_argument("--samples_folder", type=str, default='results/sobol_samples',
                        help="Folder to save Sobol samples")
    parser.add_argument("--num_sample", type=int, default=2,
                        help="Number of Sobol samples")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel mode (default: False)")
    parser.add_argument("--num_processes", type=int, default=32,
                        help="Number of parallel processes to use (default: 32)")
    parser.add_argument("--per_scenario", action="store_true",
                        help="Perform analysis per scenario (default: False)")
    parser.add_argument("--num_sample_per_scenario", type=int, default=2048,
                        help="Number of samples per scenario (default: 2048)")
    parser.add_argument("--num_inp", type=int, default=None,
                        help="Number of SWMM input files generated for parallel evaluation (if None, generated 5*num_processes)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--design", type=str, default=None,
                        help="Path to JSON file containing pipe sizes for a specific design (optional)")
    return parser.parse_args()


# Convert numpy arrays to lists for JSON serialization
def convert_numpy_to_list(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        # Handle NaN and inf values in arrays
        result = obj.tolist()
        return result
    elif isinstance(obj, (np.int64, np.int32)):
        # Integer types 
        return obj.item()
    elif isinstance(obj, (np.float64, np.float32)):
        # Float types - check for NaN/inf
        if np.isnan(obj) or np.isinf(obj):
            return None  # or handle as needed
        return obj.item()
    else:
        return obj

def sobol_analysis(swmm_path, network, setup, params_bounds_file, samples_folder,
             num_sample, parallel, num_processes, per_scenario=True, num_sample_per_scenario=2048, num_inp=None, seed=0, design=None):
    """
    Sobol sensitivity analysis for pipe network designs under deep uncertainty.

    Args:
        swmm_path (str): Base directory for SWMM input files.
        network (str): Network name.
        setup (str): Path to UDD setup JSON file.
        samples_folder (str): Folder to save SOW samples.
        num_sample (int): Number of sample for sobol analysis.
        parallel (bool): Whether to run in parallel.
        num_processes (int): Number of parallel processes.
        per_scenario (bool): Whether to perform analysis per scenario.
        num_sample_per_scenario (int): Number of samples per scenario.
        num_inp (int): Number of SWMM input files generated for parallel evaluation (if None, generated 5*num_processes).
        seed (int): Random seed for reproducibility.
    """
    setup_data = utils.load_json(setup)
    params_bounds = utils.load_json(params_bounds_file)
    climate_models = setup_data["climate_models"]
    aging_models = setup_data["aging_models"]
    # print(f"Loaded setup with climate models: {climate_models} and aging models: {aging_models}")

    # Aggregated results

    problem = {
        'num_vars': 3,
        'names': ['precipitation', 'impervious area', 'conduit Manning\'s roughness coeff'],
        'bounds': [params_bounds['climate_models_bounds']['ALL'],
                [-10, 10],
                params_bounds['aging_models_bounds']['ALL']]}
    np.random.seed(seed)  # For reproducibility
    # print(f"Sampling parameters for aggregated scenario with {num_sample} samples...")
    param_values = sobol.sample(problem, num_sample, calc_second_order=True)
    # print(f"Evaluating aggregated scenario...")
    swmm_file_path = os.path.join(swmm_path, f'{network}/{network}_udd.inp')
    swmm_info, network_info = load_pipe.read_pipe(swmm_file_path, swmm_file_path[:-3] + 'rpt')
    if design is not None:
        pipe_sizes = utils.load_json(design)['pipe_sizes']
        overflow = overflow_swmm(network_info['num_pipe'], pipe_sizes, network_info['filepath'], network_info['pipe_name'], file_suffix='_sf90')['system']
        swmm_file_path = os.path.join(swmm_path, f'{network}/{network}_udd_sf90.inp')
    results = sow.generate_SOW_sensitivity_v2(
        param_values,
        swmm_file_path,
        samples_folder=os.path.join(samples_folder, f'{network}', f'aggregated'),
        eval_file_name='sobol_eval_',
        parallel=parallel,
        num_pro=num_processes,
        num_inp=num_inp
    )
    # print(f"Performing Sobol sensitivity analysis for aggregated scenario...")
    si_aggregated = {
        'System overflow volume': analyze(problem, results, print_to_console=True)
    }

    si_aggregated = convert_numpy_to_list(si_aggregated)
    with open(os.path.join(samples_folder, f'sobol_analysis_network_{network}_aggregated_{num_sample}_seed_{seed}.json'), 'w') as f:
        json.dump(si_aggregated, f, indent=4)

    if per_scenario:
        scenarios = list(itertools.product(climate_models, aging_models))
    
        si_per_scenario = {}

        for climate, aging in scenarios:
            # print(f"Climate model: {climate}, Aging model: {aging}")

            # Define problem for Sobol analysis
            problem = {
                'num_vars': 3,
                'names': ['precipitation', 'impervious area', 'conduit Manning\'s roughness coeff'],
                'bounds': [params_bounds['climate_models_bounds'][climate],
                        [-10, 10],
                        params_bounds['aging_models_bounds'][aging]]
            }

            # Sample parameters using Sobol's method
            np.random.seed(0)  # For reproducibility
            param_values = sobol.sample(problem, num_sample_per_scenario, calc_second_order=True)

            # Evaluate the model for each parameter set

            # print(f"Evaluating for {climate} and {aging}...")
            swmm_file_path = os.path.join(swmm_path, f'{network}/{network}_udd.inp')

            results = sow.generate_SOW_sensitivity_v2(
                param_values,
                swmm_file_path,
                samples_folder=os.path.join(samples_folder, f'{network}', f'{climate}_{aging}'),
                eval_file_name='sobol_eval_',
                parallel=parallel,
                num_pro=num_processes,
                num_inp=num_inp
            )

            # print(f"Performing Sobol sensitivity analysis for {climate} and {aging}...")

            si_results = {
                'System overflow volume': analyze(problem, results, print_to_console=True),
            }
            
            
            
            si_per_scenario[f'{climate}_{aging}'] = convert_numpy_to_list(si_results)


        with open(os.path.join(samples_folder, f'sobol_analysis_network_{network}_{num_sample_per_scenario}_seed_{seed}.json'), 'w') as f:
            json.dump(si_per_scenario, f, indent=4)

    

if __name__ == "__main__":
    args = parse_arguments()
    # print(f"Starting Sobol sensitivity analysis for network {args.network} with {args.num_sample} samples (per scenario: {args.per_scenario})...")
    sobol_analysis(
        swmm_path=args.swmm_path,
        network=args.network,
        setup=args.setup,
        params_bounds_file=args.params_bounds_file,
        samples_folder=args.samples_folder,
        num_sample=args.num_sample,
        parallel=args.parallel,
        num_processes=args.num_processes,
        per_scenario=args.per_scenario,
        num_sample_per_scenario=args.num_sample_per_scenario,
        num_inp = args.num_inp,
        seed=args.seed,
        design=args.design,
    )
    