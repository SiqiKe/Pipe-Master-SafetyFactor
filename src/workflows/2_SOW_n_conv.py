"""
Generate SOW samples with resamples and analyze convergence properties.

This script generates multiple State of the World (SOW) samples with resamples, considering different
climate models and aging models. It evaluates the generated SOWs and checks convergence properties
to determine the optimal number of SOWs needed.

Usage:
    python 2_SOW_n_conv.py [--setup SETUP] [--prec_source_folder FOLDER] [--swmm_path PATH]
                           [--samples_folder FOLDER] [--network NETWORK] [--with_eval]
                           [--sampled] [--parallel] [--num_processes N]
                           [--num_SOWs N1 N2 ...] [--num_resample R1 R2 ...]
                           [--seeds N] [--already_eval]

Arguments:
    --setup: Path to UDD setup JSON file (default: data/processed/setup.json)
    --prec_source_folder: Precipitation folder
        (default: data/processed/precipitation/projection/results/)
    --swmm_path: SWMM file location (default: results/SOW_samples/37/37_udd.inp)
    --samples_folder: Folder to save samples (default: results/SOWs_conv_num/)
    --network: Network name (default: 37)
    --with_eval: Include evaluation in the results (flag)
    --sampled: Use pre-sampled data if available (flag)
    --parallel: Run in parallel mode (flag)
    --num_processes: Number of parallel processes to use (default: 16)
    --num_SOWs: List of SOW counts to generate (default: 1 5 10)
    --num_resample: Number of resamples for each SOW count (default: 3 3 3)
    --seeds: Number of random seeds for reproducibility (default: 2)
    --already_eval: Skip SOW generation and only read existing evaluations (flag)

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

import argparse
import sys
import os
import numpy as np

# Ensure the parent directory is in the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pnetwork.sow as sow
import pnetwork.utils as utils
import itertools

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate SOWs with resamples for convergence analysis.")

    parser.add_argument("--setup", type=str, default='data/processed/setup.json',
                        help="Path to UDD setup JSON file")
    parser.add_argument("--prec_source_folder", type=str, default='data/processed/precipitation/projection/results/', 
                        help="Precipitation folder")
    parser.add_argument("--swmm_path", type=str, default='results/SOW_samples/37/37_udd.inp',
                        help="SWMM file location")
    parser.add_argument("--samples_folder", type=str, default='results/SOWs_conv_num/',
                        help="Folder to save samples")
    parser.add_argument("--network", type=str, default="37", help="Network name")
    parser.add_argument("--with_eval", action='store_true', help="Include evaluation in the results (e.g. --with_eval sets args.with_eval to True)")
    parser.add_argument("--sampled", action='store_true', help="Use pre-sampled data if available (e.g. --sampled sets args.sampled to True)")
    parser.add_argument("--parallel", action='store_true', help="Run in parallel mode (e.g. --parallel sets args.parallel to True)")
    parser.add_argument("--num_processes", type=int, default=16, help="Number of parallel processes to use")
    parser.add_argument("--num_SOWs", nargs="+", type=int, default=[1, 5, 10],
                        help="List of SOW counts to generate (e.g. --num_SOWs 1 12 25 50)")
    parser.add_argument("--num_resample", nargs="+", type=int, default=[3, 3, 3],
                        help="Number of resamples for each SOW count")
    parser.add_argument("--seeds", type=int, default=2,
                        help="number of seeds")
    parser.add_argument("--already_eval", action='store_true', help="Skip SOW generation and only read existing evaluations (e.g. --already_eval sets args.already_eval to True)")
    return parser.parse_args()

def generate_sow_with_args(swmm_path, setup, samples_folder, num_sample, num_resample,
                           with_eval, sampled, prec_source_folder, parallel, num_processes, seeds=1):
    """
    Generate SOWs with resamples based on provided arguments.

    Args:
        swmm_path (str): SWMM file location.
        setup (str): Path to setup JSON file.
        samples_folder (str): Folder to save samples.
        num_sample (int): Number of SOWs to generate.
        num_resample (int): Number of resamples.
        with_eval (bool): Include evaluation in results.
        sampled (bool): Use pre-sampled data if available.
        prec_source_folder (str): Precipitation source folder.
        parallel (bool): Run in parallel mode.
        num_processes (int): Number of parallel processes.
        seeds (int): Number of random seeds for reproducibility.
    """
    # print(f"Generating SOWs: swmm_path={swmm_path}, samples_folder={samples_folder}, "
    #       f"num_sample={num_sample}, num_resample={num_resample}, with_eval={with_eval}, "
    #       f"sampled={sampled}, parallel={parallel}, num_processes={num_processes}, seeds={seeds}")

    setup_data = utils.load_json(setup)
    climate_models = setup_data["climate_models"]
    aging_models = setup_data["aging_models"]
    land_use_model_concentration = setup_data['land_use_model_concentration']

    return sow.generate_SOW_with_resamples(
        swmm_path, climate_models, aging_models, land_use_model_concentration=land_use_model_concentration,
        num_sample=num_sample, num_resample=num_resample,
        samples_folder=samples_folder,
        with_eval=with_eval, sampled=sampled,
        return_period=10, year_name=2075,
        source_file=prec_source_folder,
        parallel=parallel, num_pro=num_processes, seeds=seeds
    )

def read_data(samples_folder, seeds, num_resample=20, scenarios=None):
    """
    Read sample data from the specified folder and combine into a single numpy array.

    Args:
        samples_folder (str): Path to the folder containing sample files.
        seeds (list): List of random seeds used in sampling.
        num_resample (int): Number of resample folders to read.
        scenarios (list): List of (climate, aging) scenario tuples.

    Returns:
        np.ndarray: Data array of shape (num_seed*num_resample, num_scenarios, ...).
    """
    data = []
    for seed in range(seeds):
        for i in range(num_resample):
            results = []
            for climate, aging in scenarios:
                file_name = f"{climate}_{aging}_results.npy"
                file_path = os.path.join(samples_folder, f"resample_{i}_seed_{seed}", file_name)
                if os.path.exists(file_path):
                    loaded_data = np.load(file_path, allow_pickle=True)
                    results.append(loaded_data)
                else:
                    # print(f"File not found: {file_path}")
                    pass
            data.append(results)
    return np.array(data)

def read_results_all_scenarios(data, rel=80):
    """
    Aggregate results for all scenarios and resamples.

    Args:
        data (np.ndarray): Data array of shape (num_resample, num_scenarios, num_sample).
        rel (float): Reliability threshold.

    Returns:
        np.ndarray: Array of averaged results for each resample.
    """
    num_seed_resample, num_scenarios, num_sample = data.shape
    results = []
    for resample in range(num_seed_resample):
        resample_results = []
        for scenario in range(num_scenarios):
            result_per_scenario = sow.read_result(
                data[resample, scenario, :], num_sample, num_all_sample=num_sample, rel=rel
            )
            resample_results.append(result_per_scenario)
        resample_results = np.mean(np.array(resample_results), axis=0)
        resample_results[2] = resample_results[2] * 100  # Convert robustness to percentage
        results.append(resample_results)
    return np.array(results)

def determine_sow_n_rel(objs_resample, num_SOWs, scenarios=None):
    """
    Print convergence diagnostics for SOW sample sizes. The convergence criterion is based on the 95% confidence interval of the reliability metric for each SOW count.

    Args:
        objs_resample (dict): Dictionary of results arrays keyed by SOW count.
        num_SOWs (list): List of SOW counts.
        scenarios (list): List of scenario tuples.
    """
    num_scenarios = len(scenarios)
    objs_values = objs_resample.copy()
    for metric_ind in [0,1]:  # Loop through 0: reliability, 1:flooding_volume
        # print(f"\nConvergence diagnostics for metric index {metric_ind} ({'Reliability' if metric_ind == 0 else 'Flooding Volume'}):")
        box_data = [objs_values[j][:, metric_ind] for j in num_SOWs]  # Reliability

        # +-2.5% for reliability, +-10% for flooding volume
        mean_value = np.mean(box_data[-1])
        lower_bound = mean_value - 0.025*mean_value if metric_ind == 0 else mean_value - 0.1*mean_value
        upper_bound = mean_value + 0.025*mean_value if metric_ind == 0 else mean_value + 0.1*mean_value
        # print(f"Reference mean value: {mean_value}, Lower bound: {lower_bound}, Upper bound: {upper_bound}")

        for j in num_SOWs:
            num_samples = j * num_scenarios
            ci_lower = np.percentile(objs_values[j][:, metric_ind], 2.5)
            ci_upper = np.percentile(objs_values[j][:, metric_ind], 97.5)
            if lower_bound <= ci_lower and ci_upper <= upper_bound:
                # print(f"Sample size {j} ({num_samples} SOWs) meets the precision requirement with 95% CI: [{ci_lower}, {ci_upper}]")
                pass
            else:
                # print(f"Sample size {j} does not meet the precision requirement. 95% CI: [{ci_lower}, {ci_upper}]")
                pass

def SOW_n(swmm_path, setup, samples_folder, network, with_eval, sampled, prec_source_folder, parallel, num_processes,
          num_SOWs=[1, 5, 10], num_resample=[3, 3, 3], seeds=2, already_eval=False):
    """
    Run the SOW generation and convergence diagnostics pipeline.

    Args:
        swmm_path (str): SWMM file location.
        setup (str): Path to setup JSON file.
        samples_folder (str): Folder to save samples.
        network (str): Network name.
        with_eval (bool): Include evaluation in results.
        sampled (bool): Use pre-sampled data if available.
        prec_source_folder (str): Precipitation source folder.
        parallel (bool): Run in parallel mode.
        num_processes (int): Number of parallel processes.
        num_SOWs (list): List of SOW counts.
        num_resample (list): List of resample counts for each SOW count.
        seeds (int): Number of random seeds for reproducibility.
    """
    setup_data = utils.load_json(setup)
    climate_models = setup_data["climate_models"]
    aging_models = setup_data["aging_models"]
    scenarios = list(itertools.product(climate_models, aging_models))

    objs_resample = {}
    # print("Starting SOW generation and analysis...")
    for i, j in zip(num_SOWs, num_resample):
        samples_folder_i = os.path.join(samples_folder, f"{network}_{i}")
        if not os.path.exists(samples_folder_i):
            os.makedirs(samples_folder_i)
        if not already_eval:
            generate_sow_with_args(
                swmm_path=swmm_path,
                setup=setup,
                samples_folder=samples_folder_i,
                num_sample=i,
                num_resample=j,
                with_eval=with_eval,
                sampled=sampled,
                prec_source_folder=prec_source_folder,
                parallel=parallel,
                num_processes=num_processes,
                seeds=seeds
            )
        samples_folder_i_data = os.path.join(samples_folder_i, network)
        udd_result_i = read_data(samples_folder=samples_folder_i_data, seeds=seeds, num_resample=j, scenarios=scenarios)
        objs_resample[i] = read_results_all_scenarios(udd_result_i, rel=80)
    # print("SOW generation and analysis completed.")

    determine_sow_n_rel(objs_resample, num_SOWs=num_SOWs, scenarios=scenarios)

if __name__ == "__main__":
    args = parse_arguments()
    SOW_n(
        swmm_path=args.swmm_path,
        setup=args.setup,
        samples_folder=args.samples_folder,
        network=args.network,
        with_eval=args.with_eval,
        sampled=args.sampled,
        prec_source_folder=args.prec_source_folder,
        parallel=args.parallel,
        num_processes=args.num_processes,
        num_SOWs=args.num_SOWs,
        num_resample=args.num_resample,
        seeds=args.seeds,
        already_eval=args.already_eval
    )