"""
Prepare SOW samples for future evaluation and evaluate the UDD design under SOWs.

This script generates SOW samples with different climate and aging models for future evaluation,
and evaluates pipe network designs under these scenarios of change. The number of SOWs is
determined by the convergence analysis in step 2 (2_SOW_n_conv.py).

Usage:
    python 3_SOW_prep_udd_eval.py [--swmm_path PATH] [--networks N1 N2 ...] [--setup SETUP]
                                  [--prec_source_folder FOLDER] [--samples_folder FOLDER]
                                  [--num_sample N] [--with_eval] [--parallel]
                                  [--num_processes N] [--already_eval]

Arguments:
    --swmm_path: Base directory for SWMM input files (default: results/SOWs/)
    --networks: Network IDs to process (default: 16)
    --setup: Path to UDD setup JSON file (default: data/processed/setup.json)
    --prec_source_folder: Precipitation projection folder
        (default: data/processed/precipitation/projection/results/)
    --samples_folder: Folder to save SOW samples (default: results/SOWs/)
    --num_sample: Number of SOW samples per scenario (default: 5)
    --with_eval: Include evaluation in results (flag)
    --parallel: Run in parallel mode (flag)
    --num_processes: Number of parallel processes (default: 32)
    --already_eval: Use pre-evaluated results (flag)
    --rel_threshold: Reliability threshold for evaluation (default: 0.80)
    --eval_20prec: Evaluate the UDD design under 20% precipitation increase scenarios (flag)

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import itertools
import sys
import os
import argparse
import json
import numpy as np

# Ensure the parent directory is in the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pnetwork.sow as sow
import pnetwork.utils as utils
from pnetwork.evaluation import design_evaluation
import pnetwork.load_pipe as load_pipe

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare and evaluate SOWs for UDD design.")

    parser.add_argument("--swmm_path", type=str, default='results/SOWs/',
                        help="Base directory for SWMM input files")
    parser.add_argument("--networks", nargs="+", type=int, default=[37],
                        help="Networks to be evaluated (e.g. --networks 5 9 16)")
    parser.add_argument("--setup", type=str, default='data/processed/setup.json',
                        help="Path to UDD setup JSON file")
    parser.add_argument("--prec_source_folder", type=str, default='data/processed/precipitation/projection/results/',
                        help="Precipitation projection folder")
    parser.add_argument("--samples_folder", type=str, default='results/SOWs/',
                        help="Folder to save SOW samples")
    parser.add_argument("--num_sample", type=int, default=5,
                        help="Number of SOW samples to generate per scenario")
    parser.add_argument("--with_eval", action="store_true",
                        help="Include evaluation in the results")
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel mode (default: False)")
    parser.add_argument("--num_processes", type=int, default=32,
                        help="Number of parallel processes to use (default: 32)")
    parser.add_argument("--already_eval", action="store_true",
                        help="Flag indicating if evaluation has already been performed")
    parser.add_argument("--rel_threshold", type=float, default=80,
                        help="Reliability threshold for evaluation (default: 80)")
    parser.add_argument("--eval_20prec", action="store_true",
                        help="Evaluate the UDD design under 20% precipitation increase scenarios, default: False")
    return parser.parse_args()

def SOW_prep(swmm_path, networks, setup, prec_source_folder, samples_folder,
             num_sample, with_eval, parallel, num_processes, already_eval=False, rel_threshold=80, eval_20prec=False):
    """
    Generate SOWs and evaluate UDD design for specified networks.

    Args:
        swmm_path (str): Base directory for SWMM input files.
        networks (list): List of network IDs to process.
        setup (str): Path to UDD setup JSON file.
        prec_source_folder (str): Precipitation projection folder.
        samples_folder (str): Folder to save SOW samples.
        num_sample (int): Number of SOW samples per scenario.
        with_eval (bool): Whether to evaluate and store results.
        parallel (bool): Whether to run in parallel.
        num_processes (int): Number of parallel processes.
        already_eval (bool): Whether evaluation has already been performed before.
        rel_threshold (float): Threshold for reliability calculation (default: 80)
        eval_20prec (bool): Whether to evaluate the UDD design under 20% precipitation increase scenarios
    """
    setup_data = utils.load_json(setup)
    climate_models = setup_data["climate_models"]
    aging_models = setup_data["aging_models"]
    land_use_model_concentration=setup_data['land_use_model_concentration']
    num_scenario = len(climate_models) * len(aging_models)

    sampled = False
    for i in networks:
        # print(f"Processing network {i}...")
        swmm_file_path = os.path.join(swmm_path, f'{i}/{i}_udd.inp')
        if already_eval:
            i_results = np.load(os.path.join(swmm_path, f'{i}/udd_eval_results.npy'), allow_pickle=True)
            # print(i_results.shape)
            # print(f"Using pre-evaluated results for network {i}. using results from {os.path.join(swmm_path, f'{i}/udd_eval_results.npy')}")
            pass
        else:
            i_results = sow.generate_SOW_with_one_folder(
                i,
                swmm_file_path, climate_models, aging_models,
                land_use_model_concentration=land_use_model_concentration,
                num_sample=num_sample,
                samples_folder=samples_folder,
                sampled=sampled,
                return_period=10, year_name=2075,
                source_file=prec_source_folder,
                with_eval=with_eval,
                eval_file_name='udd_eval_',
                parallel=parallel,
                num_pro=num_processes
            )
        if with_eval:

            # Compute the total pipe length for normalization
            pipe_length_list, total_pipe_length = utils.get_total_pipe_length(swmm_file_path)

            # Compute the total area for normalization
            avg_area = utils.get_avg_area(swmm_file_path)

            # Flooding volume threshold: rough estimate 0.5 ft of flooding depth over the avg subcatchment area with 20% street coverage.
            vol_threshold = avg_area*0.2*0.5
            
            
            reliability_list = []
            flooding_volume_list = []
            robustness_list = []
            for group_idx in range(num_scenario):
                start_idx = group_idx * num_sample
                end_idx = (group_idx + 1) * num_sample
                reliability, flooding_volume, robustness = sow.read_result(
                    i_results[start_idx:end_idx],
                    num_sample=num_sample,
                    num_all_sample=num_sample,
                    rel=rel_threshold, vol=vol_threshold
                )
                reliability_list.append(reliability)
                flooding_volume_list.append(flooding_volume)
                robustness_list.append(robustness)
            # Compute averages
            reliability = sum(reliability_list) / num_scenario
            flooding_volume = sum(flooding_volume_list) / num_scenario
            robustness = sum(robustness_list) / num_scenario

            

            # Path to the existing JSON file for this network
            json_file_path = os.path.join(swmm_path, f"{i}/UDD/udd.json")

            # Load existing data if file exists, else create a new dict
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    try:
                        results_data = utils.load_json(json_file_path)
                    except Exception:
                        results_data = {}
            else:
                results_data = {}
            cost = results_data.get("cost", 0)  # Retrieve cost if it exists, else default to 0
            # Update with new results
            # print(f"Updating results in {json_file_path}...")
            results_data.update({
                "reliability": reliability,
                "flooding_volume": flooding_volume,
                "robustness": robustness,
                "reference_point": [0, 0, max(25000, flooding_volume * 10), cost * 3],
                "num_scenarios": num_scenario,
                "scenario": list(itertools.product(climate_models, aging_models)),
                "reliability_per_scenario": reliability_list,
                "flooding_volume_per_scenario": flooding_volume_list,
                "robustness_per_scenario": robustness_list,
                "pipe_length_list": pipe_length_list,
                "total_pipe_length": total_pipe_length,
                "avg_area (ft^2)": avg_area,
                "flooding_volume_threshold": vol_threshold,  # Example threshold for flooding volume: rough estimate 0.5 ft of flooding depth over the avg subcatchment area with 20% street coverage
            })

            # Write back to the JSON file
            with open(json_file_path, "w") as f:
                json.dump(results_data, f, indent=4)
        
        if eval_20prec:
            # Evaluate the UDD design under 20% precipitation increase scenarios
            # Path to the existing JSON file for this network
            json_file_path = os.path.join(swmm_path, f"{i}/UDD_20prec/udd.json")
            threshold_path = os.path.join(swmm_path, f"{i}/UDD/udd.json")
            threshold_data = utils.load_json(threshold_path)
            vol_threshold = threshold_data.get("flooding_volume_threshold", 0)  # Retrieve flooding volume threshold if it exists, else default to 0

            # Load existing data if file exists, else create a new dict
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    try:
                        results_data = utils.load_json(json_file_path)
                    except Exception:
                        results_data = {}
            else:
                results_data = {}
            cost = results_data.get("cost", 0)  # Retrieve cost if it exists, else default to 0
            design = results_data.get("results_diameter", None)  # Retrieve design if it exists, else default to None
            if design is None:
                # print(f"No design found in {json_file_path}. Skipping evaluation for network {i} with 20% precipitation increase.")
                continue
            else:
                ini_swmm_path = os.path.join(swmm_path, f"{i}/{i}_udd_20prec.inp")
                swmm_info, network_info = load_pipe.read_pipe(ini_swmm_path, ini_swmm_path[:-3] + 'rpt')
                
                swmm_folder = os.path.join(swmm_path, f"{i}")
                swmm_inp_folder = os.path.join(swmm_folder, f"inp_files")
                inp_file_name_path = os.path.join(swmm_folder, "inp_files_list.txt")
                
                i_results_20prec = design_evaluation(
            design, network_info, swmm_inp_folder, sow.read_inp_files_list(inp_file_name_path),
            num_total_sample=num_sample*num_scenario, num_scenario=num_scenario, n_obj=4, num_pro=num_processes,parallel=parallel, rel_thresholds=rel_threshold, vol_thresholds=vol_threshold)
                


                # Update with new results
                # print(f"Updating results in {json_file_path}...")
                results_data.update({
                    "reliability": i_results_20prec[0],
                    "flooding_volume": i_results_20prec[2],
                    "robustness": i_results_20prec[1],
                    })
                # Write back to the JSON file
                with open(json_file_path, "w") as f:
                    json.dump(results_data, f, indent=4)
            
            
        sampled = True

if __name__ == "__main__":
    args = parse_arguments()
    SOW_prep(
        swmm_path=args.swmm_path,
        networks=args.networks,
        setup=args.setup,
        prec_source_folder=args.prec_source_folder,
        samples_folder=args.samples_folder,
        num_sample=args.num_sample,
        with_eval=args.with_eval,
        parallel=args.parallel,
        num_processes=args.num_processes,
        already_eval=args.already_eval,
        rel_threshold=args.rel_threshold,
        eval_20prec=args.eval_20prec
    )
