"""
Functional Safety Factor (FunctionSF) method - Grid Search version.

This script implements a grid search approach for the Functional Safety Factor method,
which applies different safety factors based on pipe characteristics (length, Strahler number)
to handle uncertainties in pipe network design. The method is an extension of the System Safety Factor method.

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Functional Safety Factor method using grid search.")

    parser.add_argument("--setup", type=str, default='data/processed/setup.json',
                        help="Path to UDD setup JSON file")
    parser.add_argument("--udd_results", type=str, default="results/SOW_samples",
                        help="Path to UDD results directory")
    parser.add_argument("--ssf_results", type=str, default="sf_results_cleaned.csv",
                        help="Path to System Safety Factor results CSV file")
    parser.add_argument("--network", type=str, default="5",
                        help="Network to be evaluated (e.g. --network 5)")
    parser.add_argument("--network_path", type=str, 
                        default=None,
                        help="Path to the network folder. Default if None: will look for the file with the name as --network")
    parser.add_argument("--swmm_path", type=str, default='results/SOW_samples',
                        help="Base directory for SWMM input files (SOWs)")
    parser.add_argument("--output", type=str, default="FunctionSF", 
                        help="Output folder name")
    parser.add_argument("--n_obj", type=int, default=4, 
                        help="Number of objectives (2 or 4)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel mode")
    parser.add_argument("--num_processes", type=int, default=60,
                        help="Number of parallel processes to use (default: 60)")
    
    # Functional safety factor specific parameters
    parser.add_argument("--a_num", type=int, default=10,
                        help="Number of grid points for parameter a (default: 10)")
    parser.add_argument("--b_num", type=int, default=10,
                        help="Number of grid points for parameter b (default: 10)")
    parser.add_argument("--a_range", type=float, nargs='+', default=[0, 1],
                        help="Range for parameter a (default: [0, 1])")
    parser.add_argument("--b_range", type=float, nargs='+', default=[0, 1],
                        help="Range for parameter b (default: [0, 1])")
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
    return parser.parse_args()

def FunctionSF_GridSearch(
        swmm_path,
        network,
        network_path,
        setup,
        udd_results,
        ssf_results,
        output,
        n_obj=4,
        parallel=True,
        num_processes=60,
        a_num = 10,
        b_num = 10,
        a_range = [0, 1],
        b_range = [0, 1],
        coarse_size=15,
        fine_size=200,
        later_size=30,
        rel_thresholds=80,
        vol_thresholds=None
    ):
    """
    Perform Functional Safety Factor optimization using grid search.

    Args:
        swmm_path (str): Base directory for SWMM input files.
        network (str): Network identifier.
        setup (str): Path to setup JSON file.
        udd_results (str): Path to UDD results directory.
        ssf_results (str): Path to System Safety Factor results CSV file.
        output (str): Output folder name.
        n_obj (int): Number of objectives (2 or 4).
        parallel (bool): Whether to use parallel processing.
        num_processes (int): Number of parallel processes.
        a_num (int): Number of grid points for parameter a.
        b_num (int): Number of grid points for parameter b.
        a_range (list): Range for parameter a.
        b_range (list): Range for parameter b.
        coarse_size (int): Number of coarse grid points for safety factor search.
        fine_size (int): Number of fine grid points for safety factor search.
        later_size (int): Number of later grid points for safety factor search.
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
    if network_path is not None:
        network_folder = network_path
    else:
        network_folder = network
    udd_results_path = os.path.join(udd_results, network_folder, "UDD", "udd.json")
    ssf_results_path = os.path.join(swmm_path, network_folder, f"SystemSF_{n_obj}", ssf_results)
    ssf_runtime_path = os.path.join(swmm_path, network_folder, f"SystemSF_{n_obj}", "SystemSF.json")
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

    time_start = time.time()

    # SF design

    # Perform grid search for safety factors
    # print("Starting reading results for System Safety Factor designs...")

    # Reuse System Safety Factor results to limit the search space
    ssf_designs = pd.read_csv(ssf_results_path)
    ssf = ssf_designs['System SF'].values

    # print(f"Generated {len(ssf)} System Safety Factor designs.")


    # Build network structure and calculate Strahler numbers
    network_structure = load_pipe.build_pipe_network(swmm_info, network_info)
    root = network_structure['nodes'][network_structure['root'][0]]
    load_pipe.calculate_strahler_numbers(root)
    pipe_strahler_numbers = load_pipe.get_strahler_numbers_by_inds(root)
    

    # Perform grid search for functional safety factors
    # print("Starting Functional Safety Factor grid search...")

    # Generate functional safety factor designs
    sf_list, designs = sf.safety_factors_functional_ssf(
        num_pipe=network_info['num_pipe'],
        baseline_design=results_min_diameter,
        ssf=ssf,
        size_choice=pipe_size_per_pipe,
        pipe_length=network_info['pipe_length'],
        pipe_strahler=pipe_strahler_numbers,
        a_num=a_num,
        b_num=b_num,
        a_range=a_range,
        b_range=b_range
    )
    # print(f"Generated {len(sf_list)} functional safety factor designs.")

    # Evaluate designs under uncertainty using grid search
    df_clean_path = sf.sf_functional_result_uncertainty_ssf(
        network_info, ssf_designs, sf_list, designs,
        swmm_inp_folder,
        sow.read_inp_files_list(inp_file_name_path),
        output_folder,
        reference_point, n_obj,
        num_pro=num_processes,
        num_scenario=num_scenario,
        parallel=parallel,
        coarse_size=coarse_size,
        fine_size=fine_size,
        later_size=later_size,
        rel_thresholds=rel_thresholds,
        vol_thresholds=vol_thresholds
    )

    time_end = time.time()
    runtime = time_end - time_start

    # Add ssf runtime to the total runtime
    if os.path.exists(ssf_runtime_path):
        with open(ssf_runtime_path, "r") as f:
            ssf_runtime_data = json.load(f)
            ssf_runtime = ssf_runtime_data.get("runtime_seconds", 0)
            runtime += ssf_runtime
            # print(f"Included System Safety Factor runtime: {ssf_runtime:.2f} seconds. Total runtime: {runtime:.2f} seconds.")

    # Calculate hypervolume
    hv = utils.calculate_hv_from_df_pf(df_clean_path, reference_point, mins=mins_direction)

    # Save summary results
    results = {
        "ssf_based": True,
        "hypervolume": hv,
        "runtime_seconds": runtime,
        "num_objectives": n_obj,
        "num_scenarios": num_scenario,
        "a_num": a_num,
        "b_num": b_num,
        "a_range": a_range,
        "b_range": b_range,
        "coarse_size": coarse_size,
        "fine_size": fine_size,
        "later_size": later_size,
        "rel_thresholds": rel_thresholds,
        "vol_thresholds": vol_thresholds
    }
    
    with open(os.path.join(output_folder, "FunctionSF.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # print(f"Results saved to {output_folder}")

if __name__ == "__main__":
    args = parse_arguments()
    FunctionSF_GridSearch(
        swmm_path=args.swmm_path,
        network=args.network,
        network_path=args.network_path,
        setup=args.setup,
        udd_results=args.udd_results,
        ssf_results=args.ssf_results,
        output=args.output,
        n_obj=args.n_obj,
        parallel=args.parallel,
        num_processes=args.num_processes,
        a_num=args.a_num,
        b_num=args.b_num,
        a_range=args.a_range,
        b_range=args.b_range,
        coarse_size=args.coarse_size,
        fine_size=args.fine_size,
        later_size=args.later_size,
        rel_thresholds=args.rel_thresholds,
        vol_thresholds=args.vol_thresholds
    )