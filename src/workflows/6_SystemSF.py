"""
System-wide Safety Factor (SystemSF) Approach.

This script implements the System-wide Safety Factor approach with three search strategies: interval search, brute-force search, and adaptive grid search.
which scales pipe diameters using safety factors to handle uncertainties in pipe network design.

Usage:
    python 6_SystemSF.py [--setup SETUP] [--udd_results PATH] [--network NETWORK]
                         [--network_path PATH] [--swmm_path PATH] [--output NAME]
                         [--n_obj N] [--parallel] [--num_processes N]
                         [--sf_range RANGE] [--num_points N] [--grid_search]
                         [--coarse_size N] [--fine_size N] [--later_size N]
                         [--rel_thresholds FLOAT] [--vol_thresholds FLOAT]

Arguments:
    --setup: Path to UDD setup JSON file (default: data/processed/setup.json)
    --udd_results: Path to UDD results directory (default: results/SOWs)
    --network: Network to be evaluated (default: 5)
    --network_path: Path to the network folder; if None, uses --network as folder name
        (default: None)
    --swmm_path: Base directory for SWMM input files (SOWs) (default: results/SOWs)
    --output: Output folder name (default: SystemSF)
    --n_obj: Number of objectives, 2 or 4 (default: 4)
    --parallel: Run in parallel mode (flag)
    --num_processes: Number of parallel processes (default: 60)
    --sf_range: Range of safety factors to consider for system-wide search (default: [0.5, 3.0], change to 'None' to deactivate the interval search and adopt a brute-force search with all effective safety factors)
    --num_points: Number of points to sample within the safety factor range (default: 200, change to 'None' to search through all effective safety factors)
    --rel_thresholds: Reliability threshold for evaluation (default: 80)
    --vol_thresholds: Volume threshold for evaluation; if None, uses value from UDD results
        (default: None)
    --grid_search: Whether to perform adaptive grid search over all effective safety factors (flag)
    --coarse_size: Number of coarse grid points for safety factor search (default: 15)
    --fine_size: Number of fine grid points for safety factor search (default: 200)
    --later_size: Number of later grid points for safety factor search (default: 30)


Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import argparse
import json
import time
import numpy as np
import sys
import os

# Ensure the parent directory is in the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pnetwork.sow as sow
import pnetwork.utils as utils
import pnetwork.ga as ga
import pnetwork.load_pipe as load_pipe
import pnetwork.sf as sf

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="System Safety Factor method using grid search.")

    parser.add_argument("--setup", type=str, default='data/processed/setup.json',
                        help="Path to UDD setup JSON file")
    parser.add_argument("--udd_results", type=str, default="results/SOWs",
                        help="Path to UDD results directory")
    parser.add_argument("--network", type=str, default="5",
                        help="Network to be evaluated (e.g. --network 5)")
    parser.add_argument("--network_path", type=str, 
                        default=None,
                        help="Path to the network folder. Default if None: will look for the file with the name as --network")
    parser.add_argument("--swmm_path", type=str, default='results/SOWs',
                        help="Base directory for SWMM input files (SOWs)")
    parser.add_argument("--output", type=str, default="SystemSF", 
                        help="Output folder name")
    parser.add_argument("--n_obj", type=int, default=4, 
                        help="Number of objectives (2 or 4)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel mode")
    parser.add_argument("--num_processes", type=int, default=60,
                        help="Number of parallel processes to use (default: 60)")
    parser.add_argument("--sf_range", type=float, nargs=2, default=[0.5, 3.0],
                        help="Range of safety factors to consider for system-wide search (default: [0.5, 3.0])")
    parser.add_argument("--num_points", type=int, default=200,
                        help="Number of points to sample within the safety factor range (default: 200)")
    parser.add_argument("--coarse_size", type=int, default=15,
                        help="Number of coarse grid points for safety factor search (default: 15)")
    parser.add_argument("--fine_size", type=int, default=200,
                        help="Number of fine grid points for safety factor search (default: 200)")
    parser.add_argument("--later_size", type=int, default=30,
                        help="Number of later grid points for safety factor search (default: 30)")
    parser.add_argument("--rel_thresholds", type=float, default=80,
                        help="Reliability threshold for evaluation (default: 80)")
    parser.add_argument("--vol_thresholds", type=float, default=None,
                        help="Volume threshold for evaluation (default: None, will use value from UDD results)")
    parser.add_argument("--grid_search", action="store_true",
                        help="Whether to perform adaptive grid search")
    return parser.parse_args()

def SystemSF_GridSearch(
        swmm_path,
        network,
        network_path,
        setup,
        udd_results,
        output,
        n_obj=4,
        parallel=True,
        num_processes=60,
        sf_range=[0.5, 3.0],
        num_points=200,
        coarse_size=15,
        fine_size=200,
        later_size=30,
        rel_thresholds=80,
        vol_thresholds=None,
        grid_search=False
    ):
    """
    Perform System Safety Factor optimization using grid search.

    Args:
        swmm_path (str): Base directory for SWMM input files.
        network (str): Network identifier.
        network_path (str): Path to the network folder. Default if None: will look for the file with the name as --network.
        setup (str): Path to setup JSON file.
        udd_results (str): Path to UDD results directory.
        output (str): Output folder name.
        n_obj (int): Number of objectives (2 or 4).
        parallel (bool): Whether to use parallel processing.
        num_processes (int): Number of parallel processes.
        coarse_size (int): Number of coarse grid points for safety factor search.
        fine_size (int): Number of fine grid points for safety factor search.
        later_size (int): Number of later grid points for safety factor search.
        rel_thresholds (float): Reliability threshold for evaluation.
        vol_thresholds (float): Volume threshold for evaluation.
        grid_search (bool): Whether to perform adaptive grid search.
    """
    # Load setup data
    setup_data = utils.load_json(setup)
    commercial_diameter = setup_data["commercial_diameter"]
    climate_models = setup_data["climate_models"]
    aging_models = setup_data["aging_models"]
    num_scenario = len(climate_models) * len(aging_models)

    # Set up paths
    network_folder = network if network_path is None else network_path
    udd_results_path = os.path.join(udd_results, network_folder, "UDD", "udd.json")
    ini_swmm_path = os.path.join(udd_results, network_folder, f'{network}_udd.inp')
    swmm_folder = os.path.join(swmm_path, network_folder)
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

    # Perform grid search/brute force evaluation for safety factors
    # print("Starting System Safety Factor approach...")
    time_start = time.time()

    # Generate safety factor designs
    designs = sf.safety_factors(
        num_pipe=network_info['num_pipe'],
        baseline_design=results_min_diameter,
        size_choice=pipe_size_per_pipe,
        sf_range=sf_range,
        num_points=num_points
    )
    # print(f"Generated {len(designs)} designs.")
    # Evaluate designs under uncertainty using grid search/brute force
    df_clean_path = sf.sf_result_uncertainty(
        network_info, designs,
        swmm_inp_folder,
        sow.read_inp_files_list(inp_file_name_path),
        output_folder,
        reference_point, n_obj,
        num_pro=num_processes,
        num_scenario=num_scenario,
        parallel=parallel,
        sf_range=sf_range,
        coarse_size=coarse_size,
        fine_size=fine_size,
        later_size=later_size,
        rel_thresholds=rel_thresholds,
        vol_thresholds=vol_thresholds,
        grid_search=grid_search
    )

    time_end = time.time()
    runtime = time_end - time_start
    
    # Calculate hypervolume
    hv = utils.calculate_hv_from_df_pf(df_clean_path, reference_point, mins=mins_direction)

    # print(f"System Safety Factor grid search completed in {runtime:.2f} seconds")
    # print(f"Hypervolume: {hv}")

    # Save summary results
    if grid_search:
        results = {
            "hypervolume": hv,
            "runtime_seconds": runtime,
            "num_objectives": n_obj,
            "num_scenarios": num_scenario,
            "sf_range": sf_range,
            "num_points": num_points,
            "coarse_size": coarse_size,
            "fine_size": fine_size,
            "later_size": later_size,
            "rel_thresholds": rel_thresholds,
            "vol_thresholds": vol_thresholds
        }
    else:
        results = {
            "hypervolume": hv,
            "runtime_seconds": runtime,
            "num_objectives": n_obj,
            "num_scenarios": num_scenario,
            "sf_range": sf_range,
            "num_points": num_points,
            "num_designs": len(designs),
            "rel_thresholds": rel_thresholds,
            "vol_thresholds": vol_thresholds
        }
    
    with open(os.path.join(output_folder, "SystemSF.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # print(f"Results saved to {output_folder}")

if __name__ == "__main__":
    args = parse_arguments()
    SystemSF_GridSearch(
        swmm_path=args.swmm_path,
        network=args.network,
        network_path=args.network_path,
        setup=args.setup,
        udd_results=args.udd_results,
        output=args.output,
        n_obj=args.n_obj,
        parallel=args.parallel,
        num_processes=args.num_processes,
        sf_range=args.sf_range,
        num_points=args.num_points,
        coarse_size=args.coarse_size,
        fine_size=args.fine_size,
        later_size=args.later_size,
        rel_thresholds=args.rel_thresholds,
        vol_thresholds=args.vol_thresholds,
        grid_search=args.grid_search,
    )

