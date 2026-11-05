"""
Pipe Network Generator following the Urban Drainage Design (UDD) manual (Kilgore et al., 2024). 

"Roger Kilgore, A. Tamim Atayee, George “Rudy” Herrmann. (2024). Urban drainage design. Federal Highway Administration. https://www.fhwa.dot.gov/engineering/hydraulics/pubs/hif24006.pdf"


This script generates a pipe network design following the Urban Drainage Design (UDD) manual.
It loads network and setup data, runs a SWMM simulation, determines minimum required pipe sizes,
and saves the results. By changing the IDF curve in the setup file, users design for different return periods.

Usage:
    python 1_UDD_design.py [--base_swmm_folder BASE] [--udd_setup SETUP] [--network NAMES ...]
                           [--swmm_folder FOLDER] [--output_folder FOLDER]

Arguments:
    --base_swmm_folder: Path to base SWMM input files (default: data/processed/swmm_inputs)
    --udd_setup: Path to UDD setup JSON file (default: data/processed/setup.json), where includes the IDF curve for 10yr events with different durations and the commercial pipe diameter options.
    --network: Network name(s) to process (default: 5 9 16)
    --swmm_folder: Folder to save generated SWMM files (default: SOW_samples)
    --output_folder: Folder to save output results (default: SOW_samples)
    --intensity_2yr_24hr: 2-year, 24-hour rainfall intensity (inch/hr) (default: 0.138)
    --pipe_roughness_coef: Manning's roughness coefficient for pipes (default: 0.012)

Note: Copilot helps with the clean up of documentation.

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import argparse
import os
import sys

# Ensure the parent directory is in the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


import time
import numpy as np
import json



from pyswmm import Simulation, SystemStats

import pnetwork.udd as udd
import pnetwork.utils as utils
import pnetwork.load_pipe as load_pipe


def parse_arguments():
    """
    Parse command-line arguments for the UDD design workflow.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Urban Drainage Design (UDD)")
    parser.add_argument("--base_swmm_folder", type=str, default="data/processed/swmm_inputs",
                        help="Path to base SWMM input files")
    parser.add_argument("--udd_setup", type=str, default='data/processed/setup.json',
                        help="Path to UDD setup JSON file")
    parser.add_argument("--network", nargs="+", type=str, default=["5", "9", "16", "23", "37", "49", "62", "76", "84"], help="networks to be evaluated (e.g. --network 5 9 16)")
    parser.add_argument("--swmm_folder", type=str, default="results/SOWs",
                        help="Folder to save generated SWMM files")
    parser.add_argument("--output_folder", type=str, default="results/SOWs",
                        help="Folder to save output results")
    parser.add_argument("--intensity_2yr_24hr", type=float, default=0.138,
                        help="2-year, 24-hour rainfall intensity (inch/hr)")
    parser.add_argument("--pipe_roughness_coef", type=float, default=0.012,
                        help="Manning's roughness coefficient for pipes")
    return parser.parse_args()


def run_udd_design(swmm_folder, base_swmm_folder, udd_setup, network, output_folder, intensity_2yr_24hr=0.138, pipe_roughness_coef=0.012):
    """
    Run the Urban Drainage Design (UDD) process for a given pipe network.

    Args:
        swmm_folder (str): Folder containing SWMM files.
        base_swmm_folder (str): Folder containing base SWMM files.
        udd_setup_folder (str): Folder containing UDD setup files.
        network (str): Network name.
        output_folder (str): Folder to save the output results.
        intensity_2yr_24hr (float): 2-year, 24-hour rainfall intensity (inch/hr) used for estimating the time of concentration in each subcatchment following TR55.
        pipe_roughness_coef (float): Manning's roughness coefficient for pipes.
    """
    swmm_network_path = os.path.join(swmm_folder, network)
    os.makedirs(swmm_network_path, exist_ok=True)
    swmm_path = os.path.join(swmm_network_path, network + '.inp')
    base_swmm_path = os.path.join(base_swmm_folder, network + '.inp')

    # Copy base SWMM input file if available
    if base_swmm_path != 'None':
        import shutil
        shutil.copy(base_swmm_path, swmm_path)
    
    output_path = os.path.join(output_folder, network, 'UDD')
    os.makedirs(output_path, exist_ok=True)

    # Load the setup data
    
    setup_data = utils.load_json(udd_setup)
    commercial_diameter = setup_data["commercial_diameter"]
    ID_10yr = [setup_data["ID_10yr"]["durations"], setup_data["ID_10yr"]["intensities"]]

    # Run SWMM simulation
    start = time.time()
    with Simulation(swmm_path) as sim:
        system_routing = SystemStats(sim)
        for _ in sim:
            pass
        # print('System-wide overflow under default rain event is',
        #       system_routing.routing_stats['flooding'], 'cubic feet.')
    runtime_swmm = time.time() - start
    # print('Runtime of one simulation', runtime_swmm)

    # Load network info
    swmm_info, network_info = load_pipe.read_pipe(swmm_path, swmm_path[:-3] + 'rpt')

    # UDD: get the minimum required pipe size
    start_time = time.time()
    pipe_tree = udd.build_pipe_network_standard(
        swmm_info, 
        num_pipe=network_info['num_pipe'], 
        toc_intensity=intensity_2yr_24hr,  # 2-year, 24-hour rainfall intensity (inch/hr)
        IDF=ID_10yr
    )
    udd.update_values(
        swmm_info, 
        pipe_tree['nodes'][pipe_tree['root'][0]], 
        IDF=ID_10yr, 
        commercial_diameter=commercial_diameter, 
        pipe_roughness_coef=pipe_roughness_coef
    )
    runtime = time.time() - start_time
    # print("It takes", runtime, 'seconds to find the answer for UDD.')

    results_diameter, results_min_diameter, overflow, cost = udd.read_results(
        network_info, pipe_tree, commercial_diameter
    )

    # Save results
    results = {
        "results_diameter": results_diameter.tolist() if isinstance(results_diameter, np.ndarray) else results_diameter,
        "results_min_diameter": results_min_diameter.tolist() if isinstance(results_min_diameter, np.ndarray) else results_min_diameter,
        "single_SOW_overflow": overflow,
        "cost": cost,
        "runtime_one_swmm": runtime_swmm,
        "runtime_udd_design": runtime,
    }
    results_path = os.path.join(output_path, 'udd.json')
    # Load existing data if file exists, then update
    if os.path.exists(results_path):
        with open(results_path, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.update(results)
        results = existing_data
    
    # Save results to file
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


def main():
    """
    Main entry point for the UDD design workflow.
    Parses arguments and runs the design process for each specified network.
    """
    args = parse_arguments()
    for network in args.network:
        run_udd_design(args.swmm_folder, args.base_swmm_folder, args.udd_setup, network, args.output_folder, args.intensity_2yr_24hr, args.pipe_roughness_coef)


if __name__ == "__main__":
    main()