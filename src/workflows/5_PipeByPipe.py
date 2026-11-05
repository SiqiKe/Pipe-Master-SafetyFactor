"""
PipeByPipe NSGA-II optimization for pipe sizing.

This script implements the PipeByPipe method using NSGA-II genetic algorithm 
to optimize pipe diameters for stormwater networks under uncertainty.

Usage:
    python 5_PipeByPipe.py [--setup SETUP] [--udd_results PATH] [--network NETWORK]
                           [--network_path PATH] [--swmm_path PATH] [--output NAME]
                           [--n_obj N] [--num_processes N] [--pop_size N] [--tol FLOAT]
                           [--period N] [--seed N] [--gen N] [--rel_thresholds FLOAT]
                           [--vol_thresholds FLOAT]

Arguments:
    --setup: Path to UDD setup JSON file (default: data/processed/setup.json)
    --udd_results: Path to UDD results directory (default: results/SOW_samples)
    --network: Network to be evaluated (default: 5)
    --network_path: Path to the network folder; if None, uses --network as folder name
        (default: None)
    --swmm_path: Base directory for SWMM input files (SOWs) (default: results/SOW_samples)
    --output: Output folder name (default: PipeByPipe)
    --n_obj: Number of objectives, 2 or 4 (default: 4)
    --num_processes: Number of parallel processes to use (default: 60)
    --pop_size: Population size for NSGA-II (default: 100)
    --tol: Convergence tolerance (default: 0.0075)
    --period: Period for convergence check (default: 20)
    --seed: Random seed for reproducibility (default: 1)
    --gen: Maximum number of generations (default: 50)
    --rel_thresholds: Reliability threshold for evaluation (default: 80)
    --vol_thresholds: Volume threshold for evaluation; if None, uses value from UDD results
        (default: None)

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

from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PipeByPipe NSGA-II optimization for pipe network design.")

    parser.add_argument("--setup", type=str, default='data/processed/setup.json',
                        help="Path to UDD setup JSON file")
    parser.add_argument("--udd_results", type=str, default="results/SOW_samples",
                        help="Path to UDD results directory")
    parser.add_argument("--network", type=str, default="5",
                        help="Network to be evaluated (e.g. --network 5)")
    parser.add_argument("--network_path", type=str, 
                        default=None,
                        help="Path to the network folder. Default if None: will look for the file with the name as --network")
    parser.add_argument("--swmm_path", type=str, default='results/SOW_samples',
                        help="Base directory for SWMM input files (SOWs)")
    parser.add_argument("--output", type=str, default="PipeByPipe", 
                        help="Output folder name")
    parser.add_argument("--n_obj", type=int, default=4, 
                        help="Number of objectives (2 or 4)")
    parser.add_argument("--num_processes", type=int, default=60,
                        help="Number of parallel processes to use (default: 60)")
    
    # NSGA-II parameters
    parser.add_argument("--pop_size", type=int, default=100, 
                        help="Population size for NSGA-II (default: 100)")
    parser.add_argument("--tol", type=float, default=0.0075, 
                        help="Convergence tolerance (default: 0.0075)")
    parser.add_argument("--period", type=int, default=20, 
                        help="Period for convergence check (default: 20)")
    parser.add_argument("--seed", type=int, default=1, 
                        help="Random seed for reproducibility (default: 1)")
    parser.add_argument("--gen", type=int, default=50, 
                        help="Maximum number of generations (default: 50)")
    
    # Objectives threshold parameters (optional)
    parser.add_argument("--rel_thresholds", type=float, default=80,
                        help="Reliability threshold for evaluation (default: 80)")
    parser.add_argument("--vol_thresholds", type=float, default=None,
                        help="Volume threshold for evaluation (default: None, will use value from UDD results)")

    # Number of pipe size choices for smaller and larger pipes than the minimum diameter
    parser.add_argument("--num_choice_small", type=int, default=2,
                        help="Number of pipe size choices smaller than the minimum diameter (default: 2)")
    parser.add_argument("--num_choice_large", type=int, default=4,
                        help="Number of pipe size choices larger than the minimum diameter (default: 4)")
    return parser.parse_args()

def PipeByPipe(
        swmm_path,
        network,
        network_path,
        setup,
        udd_results,
        output,
        n_obj=4,
        num_processes=60,
        pop_size=100,
        tol=0.0075,
        period=20,
        seed=1,
        max_gen=50,
        rel_thresholds=80,
        vol_thresholds=None,
        num_choice_small=2,
        num_choice_large=4
    ):
    """
    Perform PipeByPipe optimization using NSGA-II genetic algorithm.

    Args:
        swmm_path (str): Base directory for SWMM input files.
        network (str): Network identifier.
        network_path (str): Path to the network folder. If None, will look for the file with the name as --network.
        setup (str): Path to setup JSON file.
        udd_results (str): Path to UDD results directory.
        output (str): Output folder name.
        n_obj (int): Number of objectives (2 or 4).
        num_processes (int): Number of parallel processes.
        pop_size (int): Population size for NSGA-II.
        tol (float): Convergence tolerance.
        period (int): Period for convergence check.
        seed (int): Random seed for reproducibility.
        max_gen (int): Maximum number of generations.
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
    network_folder = network_path if network_path is not None else network
    udd_results_path = os.path.join(udd_results, network_folder, "UDD", "udd.json")
    ini_swmm_path = os.path.join(udd_results, network_folder, f'{network}_udd.inp')
    swmm_folder = os.path.join(swmm_path, network_folder)
    swmm_inp_folder = os.path.join(swmm_folder, "inp_files")
    inp_file_name_path = os.path.join(swmm_folder, "inp_files_list.txt")
    output_folder = os.path.join(swmm_folder, f'{output}_{n_obj}', f'pop_{pop_size}_ftol_{tol}')
    csv_result_folder = os.path.join(output_folder, 'csv_results')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(csv_result_folder, exist_ok=True)

    # Load UDD results
    udd_data = utils.load_json(udd_results_path)
    results_min_diameter = udd_data['results_min_diameter']
    reference_point = udd_data['reference_point']
    if vol_thresholds is None:
        vol_thresholds = udd_data['flooding_volume_threshold']
    
    # Set minimization directions and reference point based on number of objectives
    
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
        num_choice_small=num_choice_small,
        num_choice_large=num_choice_large
    )

    time_start = time.time()
    # Build network structure for genetic algorithm
    network_structure = load_pipe.build_pipe_network(swmm_info, network_info)
    root = network_structure['nodes'][network_structure['root'][0]]
    load_pipe.calculate_strahler_numbers(root)
    sub_network = load_pipe.get_all_subnetwork(network_structure['nodes'])

    # Define optimization problem
    problem = ga.ProblemUncertainty_udd(
        n_var=network_info['num_pipe'],
        n_obj=n_obj,
        network_info=network_info, 
        network_nodes=network_structure,
        sub_network=sub_network,
        size_choice_per_pipe=pipe_size_per_pipe,
        inp_folder=swmm_inp_folder,
        inp_file_name=sow.read_inp_files_list(inp_file_name_path),
        swmm_file_suffix='_eval',
        num_processes=num_processes,
        rel_thresholds=rel_thresholds,
        vol_thresholds=vol_thresholds
    )

    # Configure NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=ga.UDDSampling(), 
        crossover=ga.SubTreeCrossover(),
        mutation=PM(prob=0.8, eta=3.0, vtype=float, repair=RoundingRepair()),
        repair=RoundingRepair()
    )

    # Set termination criteria
    termination = DefaultMultiObjectiveTermination(
        ftol=tol,
        xtol=0.005, # used a large xtol to disable xtol convergence check. Focus on ftol convergence instead.
        period=period,
        n_max_gen=max_gen
    )

    # print("Starting PipeByPipe NSGA-II optimization...")
    
    hvs = []
    n_evals = []
    # Run initial optimization
    res = minimize(problem, algorithm, termination, seed=seed, save_history=True, verbose=True)
    
    # Save results and calculate hypervolume
    run_name = f"{network}_seed_{seed}_0"
    a, b, hv, n_eval = ga.GA_history(
        problem, res, run_name,
        network_info['num_pipe'], 
        commercial_diameter=pipe_size_per_pipe,
        save_path=csv_result_folder + '/',
        ref_point=reference_point
    )
    hvs.append(hv)
    n_evals.append(n_eval)

    # Check if convergence was achieved
    ftol_meet = res.algorithm.n_gen < max_gen
    if not ftol_meet:
        print(f"Algorithm stopped at max generation: {res.algorithm.n_gen} without reaching ftol")

    # Continue optimization if convergence not achieved
    resume_count = 0
    while not ftol_meet:
        resume_count += 1
        # print(f"Resuming optimization (attempt {resume_count})...")
        
        # Use final population as starting point
        pop = res.algorithm.pop
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=pop, 
            crossover=ga.SubTreeCrossover(),
            mutation=PM(prob=0.8, eta=3.0, vtype=float, repair=RoundingRepair()),
            repair=RoundingRepair()
        )

        res = minimize(problem, algorithm, termination, seed=seed, save_history=True, verbose=True)
        
        # Save resumed results
        run_name = f"{network}_seed_{seed}_{resume_count}"
        a, b, hv, n_eval = ga.GA_history(
            problem, res, run_name,
            network_info['num_pipe'], 
            commercial_diameter=pipe_size_per_pipe,
            save_path=csv_result_folder + '/',
            ref_point=reference_point
        )
        hvs.append(hv)
        n_evals.append(n_eval)

        if res.algorithm.n_gen < max_gen:
            # print(f"Algorithm converged at generation: {res.algorithm.n_gen}")
            ftol_meet = True
        else:
            # print(f"Algorithm stopped at max generation: {res.algorithm.n_gen} without reaching ftol")
            pass


    # Find the best Pareto front from the history of the last runs
    try:
        mins_direction = [1]*n_obj
        drop_duplicates_columns = [f'x{i}' for i in range(1,1+network_info['num_pipe'])]
        objective_columns = [-n_obj - 1 + i for i in range(n_obj)]
        path_to_history = os.path.join(csv_result_folder, f"pop_{network}_seed_{seed}_{resume_count}.csv")
        df_clean_path = utils.clean_pf(path_to_history, n_obj, mins=mins_direction, gens = 10, drop_duplicates_columns=drop_duplicates_columns)
        final_hv = utils.calculate_hv_from_df_pf(df_clean_path, reference_point, mins=mins_direction, columns=objective_columns)
    except Exception as e:
        # print(f"Error occurred while calculating final hypervolume: {e}")
        final_hv = None

    time_end = time.time()
    runtime = time_end - time_start

    # print(f"PipeByPipe optimization completed in {runtime:.2f} seconds")

    # Save summary results
    summary_results = {
        "hypervolume": hvs,
        "final_hypervolume": final_hv,
        "n_evaluations": n_evals,
        "runtime_seconds": runtime,
        "num_objectives": n_obj,
        "num_scenarios": num_scenario,
        "population_size": pop_size,
        "convergence_ftolerance": tol,
        "seed": seed,
        "rel_thresholds": rel_thresholds,
        "vol_thresholds": vol_thresholds
    }
    
    with open(os.path.join(output_folder, "PipeByPipe.json"), "w") as f:
        json.dump(summary_results, f, indent=4)
    
    # print(f"Results saved to {output_folder}")

if __name__ == "__main__":
    args = parse_arguments()
    PipeByPipe(
        swmm_path=args.swmm_path,
        network=args.network,
        network_path=args.network_path,
        setup=args.setup,
        udd_results=args.udd_results,
        output=args.output,
        n_obj=args.n_obj,
        num_processes=args.num_processes,
        pop_size=args.pop_size,
        tol=args.tol,
        period=args.period,
        seed=args.seed,
        max_gen=args.gen,
        rel_thresholds=args.rel_thresholds,
        vol_thresholds=args.vol_thresholds,
        num_choice_small=args.num_choice_small,
        num_choice_large=args.num_choice_large
    )

