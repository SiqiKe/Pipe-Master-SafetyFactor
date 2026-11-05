"""
Generalize SOW (State of the World) samples and stress testing for the given design.

Main Functions:
- generate_SOW_with_resamples(base_path, climate_models, aging_models, land_use_model_concentration,
                            num_sample = 1000, num_resample = 20, samples_folder='samples/', 
                            with_eval = True, sampled = False, return_period = 10, year_name = 2075)
-
Note:
Copilot helps with function documentation.

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""


import random
import pandas as pd
import time
import os
import itertools
from pyswmm import Simulation, SystemStats
import pnetwork.pr_sampling as pr
import pnetwork.ag_sampling as ag
import pnetwork.load_pipe as load_pipe
import pnetwork.imp_sampling as imp
import pnetwork.evaluation as ev
import pnetwork.rtd as rtd
import numpy as np
from pnetwork.modify_inp import read_inp, delete_rainfall, add_uncertainties

def initial_(swmm_path):
    """
    This function runs a SWMM simulation to gather basic information about the model,
    including the number of pipes, subcatchments, and impervious percentages.

    Parameters:
    swmm_path : str
        Path to the SWMM input file (.inp).

    Returns:
    swmm_info : dict
        Information about the SWMM model.
    network_info : dict
        Information about the network including number of pipes.
    subs : list
        List of subcatchment names.
    impervs : list
        List of impervious percentages for each subcatchment.
    """
    # Set up the simulation
    # Run SWMM (making sure there will be a rpt file)
    start = time.time()
    with Simulation(swmm_path) as sim:
        system_routing = SystemStats(sim)
        for step in sim:
            pass       
        # print('System-wide overflow under default rain event is', system_routing.routing_stats['flooding'], 'square feet.') #feet^3
    # print('Runtime of one simulation', time.time()-start)


    # Load info
    swmm_info, network_info = load_pipe.read_pipe(swmm_path, swmm_path[:-3]+'rpt')
    
    return swmm_info, network_info, swmm_info['Subcatchments']['Name'], swmm_info['Subcatchments']['%Imperv']



def read_samples(impervs, climate_models, aging_models, land_use_model_concentration, num_sample, 
                 num_resample, samples_folder, network_folder, 
                 sampled=True, return_period=10, year_name=2075, source_file='../precipitation/projection/results/', seeds=1):
    """
    Read in CSV files for precipitation samples info and system aging samples if sampled=True
    or generate new samples if sampled=False and generate impervious area samples.
    """
    if sampled:
        climate_samples = {}
        for i in climate_models:
            climate_samples[i] = pd.read_csv(os.path.join(samples_folder, 'pr', f"{i}.csv"))
        aging_samples = {}
        for j in aging_models:
            aging_samples[j] = pd.read_csv(os.path.join(samples_folder, f"aging_{j}.csv"))
    else:
        climate_samples = pr.precipitation_sampling(
            num_sample, num_resample, return_period, climate_models, year_name,
            folder=os.path.join(samples_folder, 'pr'), source_file=source_file, seeds=seeds
        )
        aging_samples = ag.pipe_aging_sampling(
            aging_models, num_sample, num_resample, folder=samples_folder, seeds=seeds
        )
        
    impervs = [float(i) for i in impervs]
    imperv_samples = imp.imperv_sampling_sampling(
        impervs, num_sample, num_resample, concentration=land_use_model_concentration, 
        file_name='', folder=network_folder, seeds=seeds
    )

    return climate_samples, aging_samples, imperv_samples



def read_rainfall_data(sample_id, folder_path):
    """
    Reads rainfall data from a file.
    """
    with open(os.path.join(folder_path, f"{sample_id}.dat"), 'r') as file:
        rainfall = file.readlines()
    return rainfall


def write_samples_inp(base_file, aging_samples, imperv_samples, pr_model, aging_model, subs,
                      num_sample, resample, sample_folder, new_folder, seed=0):
    """
    Write the ".inp" files with uncertainties for each sample and resample.
    """
    pr_folder = os.path.join(sample_folder, 'pr', pr_model, f"resample{resample}_seed_{seed}")
    aging_df = aging_samples[aging_model]
    for i in range(num_sample):
        rainfall = read_rainfall_data(i, pr_folder)
        aging = aging_df.loc[
            (aging_df['samples_id'] == i) & (aging_df['resample_id'] == resample) & (aging_df['seed'] == seed), 'roughness'
        ].values[0]
        impervs = imperv_samples[:, seed, resample, i]
        newfile_path = os.path.join(new_folder, f"{i}.inp")
        add_uncertainties(base_file, newfile_path, rainfall, subcatch=subs, imperv=impervs, roughness=aging)

def write_samples_inp_one_folder(base_file, pr_samples, aging_samples, imperv_samples, pr_model, aging_model, subs,
                      num_sample, resample, sample_folder, new_folder, seed=0):
    """
    Write the ".inp" files with uncertainties for each sample and resample.
    """
    pr_folder = os.path.join(sample_folder, 'pr', pr_model, f"resample{resample}_seed_{seed}")
    aging_df = aging_samples[aging_model]
    
    for i in range(num_sample):
        rainfall = read_rainfall_data(i, pr_folder)
        aging = aging_df.loc[
            (aging_df['samples_id'] == i) & (aging_df['resample_id'] == resample)& (aging_df['seed'] == seed), 'roughness'
        ].values[0]
        impervs = imperv_samples[:, seed, resample, i]
        newfile_path = os.path.join(new_folder, f"{pr_model}_{aging_model}_{i}.inp")
        add_uncertainties(base_file, newfile_path, rainfall, subcatch=subs, imperv=impervs, roughness=aging)
    return [f"{pr_model}_{aging_model}_{i}.inp" for i in range(num_sample)]




def generate_SOW_with_resamples(base_path, climate_models, aging_models, land_use_model_concentration,
                                num_sample=1000, num_resample=20, samples_folder='samples/', 
                                with_eval=True, sampled=False, return_period=10, year_name=2075, 
                                source_file='../precipitation/projection/results/', parallel=True, num_pro=2, seeds=1):
    """
    Generate SOW with resamples and optionally evaluate them.

    num_sample : int number of samples to generate for each scenario
    """
    # Initial setup
    swmm_info, network_info, subs, impervs = initial_(base_path)
    num_pipe = network_info['num_pipe']
    
    network_folder = os.path.join(samples_folder, str(num_pipe))
    os.makedirs(network_folder, exist_ok=True)

    inp_folder = os.path.join(network_folder, 'inp_files')
    os.makedirs(inp_folder, exist_ok=True)
    
    # Read the file
    base_inp = read_inp(base_path)
    new_inp = delete_rainfall(base_inp, '24hr', os.path.join(network_folder, 'base.inp'), True)

    # Read the samples
    _, aging_samples, imperv_samples = read_samples(
        impervs, climate_models, aging_models, land_use_model_concentration, num_sample, num_resample, samples_folder, 
        network_folder, sampled, return_period, year_name, source_file = source_file, seeds=seeds
    )

    # Scenarios
    scenarios = list(itertools.product(climate_models, aging_models))
    
    # Write inp with evaluation per resample
    if with_eval:
        results_df = {}
    for seed in range(seeds):
        for i in range(num_resample):
            resample_path = os.path.join(network_folder, f"resample_{i}_seed_{seed}")
            os.makedirs(resample_path, exist_ok=True)
            if with_eval:
                results_df[(seed, i)] = {}
            for scen in scenarios:
                scen_folder = os.path.join(inp_folder, f"{scen[0]}_{scen[1]}") #file for each resample and seed will be overwrited, so don't need a new location
                os.makedirs(scen_folder, exist_ok=True)
                write_samples_inp(
                    new_inp,  aging_samples, imperv_samples, scen[0], scen[1], 
                    subs, num_sample, i, samples_folder, scen_folder, seed=seed
                )

                if with_eval:
                    # print(f"Evaluating scenario {scen[0]}_{scen[1]} for resample {i} and seed {seed}...")
                    results_df[(seed, i)][scen] = ev.evaluate_uncertainty_eval_only(
                        scen_folder, num_sample, num_pro=num_pro, save=True, 
                        save_name=f"{scen[0]}_{scen[1]}_", save_path=resample_path, parallel=parallel
                    )
    if with_eval:
        return results_df
    return None

def split_integer(x, y):
    base = x // y
    remainder = x % y
    result = [base + 1 if i < remainder else base for i in range(y)]
    return result

def generate_SOW_with_one_folder(network_name, base_path, climate_models, aging_models, land_use_model_concentration,
                                num_sample=1000, samples_folder='samples/', 
                                sampled=False, return_period=10, year_name=2075, 
                                source_file='../precipitation/projection/results/', with_eval=True, eval_file_name = 'eval_', parallel=True, num_pro=2):
    """
    Generate SOW and store them in a specified folder, and optionally evaluate them.

    num_sample : int number of samples to generate for each scenario
    with_eval (bool): Whether to evaluate the generated SOW samples.
    """
    # Initial setup
    swmm_info, network_info, subs, impervs = initial_(base_path)
    num_pipe = network_info['num_pipe']
    
    network_folder = os.path.join(samples_folder, str(network_name))
    os.makedirs(network_folder, exist_ok=True)

    inp_folder = os.path.join(network_folder, 'inp_files')
    os.makedirs(inp_folder, exist_ok=True)
    
    # Read the file
    base_inp = read_inp(base_path)
    new_inp = delete_rainfall(base_inp, '24hr', os.path.join(network_folder, 'base.inp'), True)

    # Read the samples
    num_resample = 1
    pr_samples, aging_samples, imperv_samples = read_samples(
        impervs, climate_models, aging_models, land_use_model_concentration, num_sample, num_resample, samples_folder, 
        network_folder, sampled, return_period, year_name, source_file = source_file
    )

    # Scenarios
    scenarios = list(itertools.product(climate_models, aging_models))
    
    # Write inp with evaluation per resample
    
    inp_files_name = []
    for scen in scenarios:
        inp_files_name_scen = write_samples_inp_one_folder(
            new_inp, pr_samples, aging_samples, imperv_samples, scen[0], scen[1], 
            subs, num_sample, 0, samples_folder, inp_folder
        )
        inp_files_name.extend(inp_files_name_scen)
    
    # Write inp_files_name to a file
    inp_files_name_path = os.path.join(network_folder, 'inp_files_list.txt')
    with open(inp_files_name_path, 'w') as file:
        for inp_file in inp_files_name:
            file.write(f"{inp_file}\n")


    if with_eval:
        results = ev.evaluate_uncertainty_eval_only_allinp(
            inp_folder, inp_files_name, num_pro=num_pro, save=True, 
            save_name= eval_file_name, save_path=network_folder, parallel=parallel
        )
        return results

    return None



def generate_SOW_sensitivity(para_list, base_path, samples_folder='samples/', 
                                 eval_file_name = 'eval_', parallel=True, num_pro=2):
    """
    Generate SOW with given para_list and evaluate them.
    """
    # print(f"Generating SOW samples with {len(para_list)} parameters")
    # Initial setup
    swmm_info, network_info, subs, impervs = initial_(base_path)
    num_pipe = network_info['num_pipe']
    

    inp_folder = os.path.join(samples_folder, 'inp_files')
    os.makedirs(inp_folder, exist_ok=True)

    # Prepare rainfall files
    pr_folder = os.path.join(samples_folder, 'pr')
    os.makedirs(pr_folder, exist_ok=True)
    rainfall_pf = pd.DataFrame({
        'sample_id': [str(i) for i in range(len(para_list))],
        'intensity_in_hr': [para[0] for para in para_list]})
    rtd.write_dat_file_given_name(rainfall_data=rainfall_pf, name_column='sample_id', 
                                        folder_name=pr_folder, start_time='01/01/2025 00:00')
    
    # print(f"Generated rainfall files for {len(para_list)} samples in {pr_folder}")

    # Read the file
    base_inp = read_inp(base_path)
    new_inp = delete_rainfall(base_inp, '24hr', os.path.join(samples_folder, 'base.inp'), True)

    for idx, para in enumerate(para_list):
        newfile_path = os.path.join(inp_folder, f"{idx}.inp")
        rainfall = read_rainfall_data(idx, pr_folder)
        impervs = np.array([max(0, min(100, int(imp) + para[1])) for imp in impervs])
        add_uncertainties(new_inp, newfile_path, rainfall, subcatch=subs, imperv=impervs, roughness=para[2])
    
    
    inp_files_name = [f"{i}.inp" for i in range(len(para_list))]
    # Write inp_files_name to a file
    inp_files_name_path = os.path.join(inp_folder, 'inp_files_list.txt')
    with open(inp_files_name_path, 'w') as file:
        for inp_file in inp_files_name:
            file.write(f"{inp_file}\n")

    # print(f"Generated .inp files for {len(para_list)} samples in {inp_folder}")

    results = ev.evaluate_uncertainty_eval_only_allinp(
            inp_folder, inp_files_name, num_pro=num_pro, save=True, 
            save_name= eval_file_name, save_path=samples_folder, parallel=parallel
        )
    return results


def generate_SOW_sensitivity_v2(para_list, base_path, samples_folder='samples/', 
                                 eval_file_name = 'eval_', parallel=True, num_pro=2, num_inp=None):
    """
    Generate SOW with given para_list and evaluate them. 
    In this version, we change how we generate the .inp files to make it more efficient for large number of samples.
    Instead of creating separate files for each SOW, we sample .inp for each processor, and change rainfall, imp and aging to each SOW within pySWMM. 
    This way we can avoid creating too many files and also make it easier to manage the files.
    """
    if num_inp is None:
        num_inp = 5 * num_pro # number of .inp files to create, which should be larger than num_pro to make sure each processor can have different .inp files to run in parallel
    # print(f"Generating SOW samples with {len(para_list)} parameters using {num_inp} .inp files")

    def _indices(total_length, num_groups):
        base_size = total_length // num_groups
        remainder = total_length % num_groups
        indices = []
        start = 0
        for i in range(num_groups):
            end = start + base_size + (1 if i < remainder else 0)
            indices.append((start, end))
            start = end
        return indices

    groups_idx = _indices(len(para_list), num_inp)

    # Initial setup
    swmm_info, network_info, subs, impervs = initial_(base_path)
    pipe_names = network_info['pipe_name']
    

    inp_folder = os.path.join(samples_folder, 'inp_files')
    os.makedirs(inp_folder, exist_ok=True)

    import shutil
    for i in range(num_inp):
        shutil.copy(base_path, os.path.join(inp_folder, f"{i}.inp"))

    inp_files_name = [f"{i}.inp" for i in range(num_inp)]
    # Write inp_files_name to a file
    inp_files_name_path = os.path.join(inp_folder, 'inp_files_list.txt')
    with open(inp_files_name_path, 'w') as file:
        for inp_file in inp_files_name:
            file.write(f"{inp_file}\n")

    # print(f"Generated .inp files for {num_inp} samples in {inp_folder}")

    results = ev.evaluate_uncertainty_eval_only_group(
            inp_folder, inp_files_name, num_inp, groups_idx, para_list, subs, impervs, pipe_names, num_pro=num_pro, save=True, 
            save_name= eval_file_name, save_path=samples_folder, parallel=parallel
        )
    return results


def read_inp_files_list(file_path):
    """
    Reads the list of .inp file names from a file.

    Parameters:
    file_path : str
        Path to the file containing the list of .inp file names.

    Returns:
    list
        List of .inp file names.
    """
    with open(file_path, 'r') as file:
        inp_files = file.read().splitlines()
    return inp_files

import numpy as np
def read_result(results, num_sample, num_all_sample = 1000, rel = 80, vol = 500):
    rind = random.sample(range(num_all_sample),num_sample)
    ## Reliability
    system_overflow = results[rind]
    reliability = (np.count_nonzero(system_overflow == 0)/num_sample)*100
    mask = results > 0
    avg_system_flooding_volume = np.mean(results[mask]) if np.any(mask) else 0
    

    if reliability >= rel and avg_system_flooding_volume <= vol:
        robustness = 1
    else:
        robustness = 0
    return reliability, avg_system_flooding_volume, robustness

def read_results(results, num_sample, num_resample, climate_models, aging_models, num_all_sample = 1000, rel = 80, vol = 500):
    scenarios = list(itertools.product(climate_models, aging_models))
    objs = np.zeros((num_resample, len(scenarios), 3))
    for i in range(num_resample):
        for j in range(len(scenarios)):
            objs[i, j, :] = (read_result(results[i][scenarios[j]], num_sample, num_all_sample, rel, vol))
    return scenarios, objs