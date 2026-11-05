"""
This module is for pipe network's performance assessment.

Main Functions:
- overflow_swmm(n, x, model, pipe_name, file_suffix = '_eval') -> dict
- system_cost(x, length) -> float
- get_sizes(num_pipe, size_level_list, size_choice)

Note:
None

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""


from pyswmm import Simulation, SystemStats, SimulationPreConfig
import numpy as np
import os
from .rtd import nrcs_method_24hr_rainfall

def cost_factor(pipe_size) -> float:
    """
    calculate the cost factor of particular pipe size in January, 1999 USD

    ENR construction cost index for January, 1999 is 6000
    ENR construction cost index for May, 2025 is 13532.44
    Conversion factor: 13532.44 / 6000 = 2.255407

    Args:
    pipe_size (float): pipe diameter in feet

    Returns:
    float: cost factor
    """

    return 0.19*((pipe_size*12)**1.6259)*2.255407 #concrete pipe in 1999 USD adjusted to 2025 USD

def overflow_swmm_eval_only(file_name) -> dict:
    """Eval without changing pipes' size"""
    output = {}
    with Simulation(file_name) as sim:
        system_routing = SystemStats(sim)
        for step in sim:
            pass
        flood = system_routing.routing_stats['flooding']
  
    output['system'] = flood
    
    
    return output

def overflow_swmm_eval_new_para(model, rainfall_intensity, subs, impervs_values, pipe_name, pipe_roughness, file_suffix = '_eval') -> dict:
    """
    Using swmm to report the overflow volume of the system with parameter changes in precipitation, impervious area and conduit roughness.

    Args:
    model (str): filepath for the SWMM inp file
    rainfall_intensity (float): rainfall intensity in inch/hr
    subs (list): name of subcatchments
    impervs_values (list): new imperviousness values for the subcatchments, in percentage (0-100)
    pipe_name (list): name of pipes
    pipe_roughness (float): a new roughness value for the pipes
    file_suffix (str): file suffix for the SWMM inp with new parameters, default is '_eval'

    Returns:
    dict: overflow volume within the system
    """
    output = {}
    try:
        sim_conf = SimulationPreConfig()
        rainfall = nrcs_method_24hr_rainfall(rainfall_intensity)
        for i in range(len(rainfall)):
            sim_conf.add_update_by_token("TIMESERIES", "24hr", 3, rainfall[i], i) # 3 is the coloumn number of rainfall in TIMESERIES, i is the row number for each time step in "24hr" timeseries
        for i in range(len(subs)):
            #print(f"Updating subcatchment {subs[i]} with imperviousness {impervs_values[i]}%")
            sim_conf.add_update_by_token("SUBCATCHMENTS", subs[i], 4, impervs_values[i])
        for i in range(len(pipe_name)):
            #print(f"Updating pipe {pipe_name[i]} with roughness {pipe_roughness}")
            sim_conf.add_update_by_token("CONDUITS", pipe_name[i], 4, pipe_roughness) # 4 is the coloumn number of roughness in CONDUITS
        sim_conf.filename_suffix = file_suffix
        
        with Simulation(model, sim_preconfig = sim_conf) as sim:
            system_routing = SystemStats(sim)
            for step in sim:
                pass

            output['system'] = system_routing.routing_stats['flooding'] #cubic feet
        return output
    except Exception as e:
        print(f"An error occurred during SWMM simulation: {e}")
        return {'system': float('inf')}
    finally:
        try:
            sim.close()
        except:
            pass

def overflow_swmm(n, x, model, pipe_name, file_suffix = '_eval') -> dict:
    """
    Using swmm to report the overflow volume of the system and each node

    Args:
    n (int): the number of pipe in the network
    x (list or array): size of pipes
    model (str): filepath for the SWMM inp file
    pipe_name (list): name of pipes
    file_suffix (str): file suffix for the SWMM inp with new pipe sizes

    Returns:
    dict: overflow volume within the system
    """
    output = {}
    try:
        sim_conf = SimulationPreConfig()
        for i in range(n):
            sim_conf.add_update_by_token("XSECTIONS", pipe_name[i], 2, x[i]) # 2 is the coloumn number of diameter in Xsections
        sim_conf.filename_suffix = file_suffix
        
        with Simulation(model, sim_preconfig = sim_conf) as sim:
            system_routing = SystemStats(sim)
            for step in sim:
                pass

            output['system'] = system_routing.routing_stats['flooding'] #cubic feet
        return output
    except Exception as e:
        print(f"An error occurred during SWMM simulation: {e}")
        return {'system': float('inf')}
    finally:
        try:
            sim.close()
        except:
            pass


def system_cost(x, length) -> float:      
    """
    calculate the cost of the system

    Args:
    x (list or array): size of pipes in feet
    length (list of array: length of pipes in feet

    Returns:
    float: system cost
    """
    sum_cost = 0
    for iter_x, iter_length in zip(x,length):
        sum_cost += cost_factor(iter_x)*iter_length
    return sum_cost





def get_sizes(num_pipe, size_level_list, size_choice):
    """
    get the list of diameters from a list of size_level

    Args:
    num_pipe (int): the number of pipes
    size_level_list (list/array): size level of corresponding pipe
    size_choice (list/array): diameter choice

    Returns:
    list/array: diameters of each pipe
    """
    diameters = np.zeros(num_pipe)
    for i in range(num_pipe):
        diameters[i] = size_choice[int(size_level_list[i])]
    return diameters



import os
def evaluate_scenario_SWMM_eval_only(sample, inp_folder):
    """
    Args:
    sample (int): sample id 
    network_info (dict)
    inp_folder (str): source path of the folder which stores all inp

    Returns:
    dict: overflow volume within the system
    """
    
    result = overflow_swmm_eval_only(os.path.join(inp_folder, f"{sample}.inp"))
    

    return result['system']

def evaluate_scenario_SWMM_path(swmm_path):
    """
    Args:
    swmm_path (str): source path of the swmm inp file

    Returns:
    dict: overflow volume within the system
    """
    
    result = overflow_swmm_eval_only(swmm_path)
    
    return result['system']
    
def evaluate_scenario_SWMM_new_design(i, sample, network_info, inp_folder, swmm_file_suffix = '_eval'):
    """
    Args:
    i (list or array): size of pipes
    sample (int): sample id 
    network_info (dict)
    inp_folder (str): source path of the folder which stores all inp
    swmm_file_suffix (str): a file suffix to avoid replacing the source file

    Returns:
    dict: overflow volume within the system
    """
    result = overflow_swmm(network_info['num_pipe'], i, os.path.join(inp_folder, f"{sample}"), 
                                 network_info['pipe_name'],
                             file_suffix = swmm_file_suffix)

    return result['system']


def x_filename(x):
    filename = '_'.join(map(lambda i: str(i).replace('.', '-'), x))
    filename_with_extension = f"{filename}.npy"
    return filename_with_extension



def evaluate_uncertainty_eval_only(inp_folder, num_total_sample, num_pro=8, save=False, save_name='', save_path='eval_results/', parallel=True):
    """
    Evaluate uncertainty with an option to disable parallelization.

    Args:
    inp_folder (str): Source path of the folder which stores all inp files.
    num_total_sample (int): Number of samples to evaluate.
    num_pro (int): Number of processes for parallelization.
    save (bool): Whether to save the results.
    save_name (str): Name for the saved results file.
    save_path (str): Path to save the results.
    parallel (bool): Whether to use parallel processing.

    Returns:
    np.ndarray: Results of the evaluation.
    """
    tasks = [(sample, inp_folder) for sample in range(num_total_sample)]

    if parallel:
        from joblib import Parallel, delayed
        from tqdm import tqdm
        results = Parallel(n_jobs = num_pro)(delayed(evaluate_scenario_SWMM_eval_only)(*task) for task in tqdm(tasks, disable=True))
    else:
        # Sequential processing
        results = [evaluate_scenario_SWMM_eval_only(*task) for task in tasks]

    results = np.array(results)

    if save:
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, save_name + 'results.npy')
        np.save(path, results)

    return results

def evaluate_uncertainty_eval_only_allinp(inp_folder, file_name, num_pro=8, save=False, save_name='', save_path='eval_results/', parallel=True):
    """
    Evaluate uncertainty with an option to disable parallelization.

    Args:
    inp_folder (str): Source path of the folder which stores all inp files.
    file_name (list): names of all files
    num_pro (int): Number of processes for parallelization.
    save (bool): Whether to save the results.
    save_name (str): Name for the saved results file.
    save_path (str): Path to save the results.
    parallel (bool): Whether to use parallel processing.

    Returns:
    np.ndarray: Results of the evaluation.
    """

    tasks = [os.path.join(inp_folder, i) for i in file_name]

    if parallel:

        from joblib import Parallel, delayed
        from tqdm import tqdm
        results = Parallel(n_jobs = num_pro)(delayed(evaluate_scenario_SWMM_path)(task) for task in tqdm(tasks, disable=True))
    else:
        # Sequential processing
        results = [evaluate_scenario_SWMM_path(task) for task in tasks]

    results = np.array(results)

    if save:
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, save_name + 'results.npy')
        np.save(path, results)

    return results

def evaluate_uncertainty_eval_only_group(inp_folder, file_name, num_inp, group_idx, para_list, subs, subs_value, pipe_names, num_pro=8, save=False, save_name='', save_path='eval_results/', parallel=True):
    """
    Evaluate uncertainty with parameters set grouped by rainfall values.

    Args:
    inp_folder (str): Source path of the folder which stores all inp files with different rainfall values.
    file_name (list): names of all files
    num_inp (int): number of inp files
    group_idx (list): list of indices to group by
    para_list (list): list of parameter values
    subs (list): name of subcatchments
    subs_value (list): impervious values of subcatchments
    pipe_names (list): name of pipes
    num_pro (int): Number of processes for parallelization.
    save (bool): Whether to save the results.
    save_name (str): Name for the saved results file.
    save_path (str): Path to save the results.
    parallel (bool): Whether to use parallel processing.

    Returns:
    np.ndarray: Results of the evaluation.
    """

    def evaluate_scenario_SWMM_group(task):
        model = task[0]
        subs = task[1]
        subs_value = task[2]
        pipe_names = task[3]
        param_values = task[4]
        results = []
        for i in range(len(param_values)):
            subs_value_new = np.array([max(0, min(100, int(imp) + param_values[i][1])) for imp in subs_value])
            result = overflow_swmm_eval_new_para(model, param_values[i][0], subs, subs_value_new, pipe_names, param_values[i][2], file_suffix = f"_eval")   
            results.append(result['system'])
        return results

    tasks = [(os.path.join(inp_folder, file_name[i]), subs, subs_value, pipe_names, para_list[group_idx[i][0]:group_idx[i][1]]) for i in range(num_inp)]


    if parallel:

        from joblib import Parallel, delayed
        from tqdm import tqdm
        results = Parallel(n_jobs = num_pro)(delayed(evaluate_scenario_SWMM_group)(task) for task in tqdm(tasks, disable=True))
    else:
        # Sequential processing
        results = [evaluate_scenario_SWMM_group(task) for task in tasks]

    # Flatten list of lists into a 1D numpy array
    results = np.array([item for sublist in results for item in sublist])

    if save:
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, save_name + 'results.npy')
        np.save(path, results)

    return results


def evaluate_uncertainty_new_design(i, network_info, inp_folder, file_name, num_pro=8, save=False, save_name='', save_path='eval_results/', parallel=True):
    """
    Evaluate uncertainty with an option to disable parallelization.

    Args:
    i (list or array): size of pipes
    network_info (dict): Network information including pipe names.
    inp_folder (str): Source path of the folder which stores all inp files.
    file_name (list): names of all files
    num_pro (int): Number of processes for parallelization.
    save (bool): Whether to save the results.
    save_name (str): Name for the saved results file.
    save_path (str): Path to save the results.
    parallel (bool): Whether to use parallel processing.

    Returns:
    np.ndarray: Results of the evaluation.
    """
    tasks = [(i, sample, network_info, inp_folder) for sample in file_name]

    if parallel:
        from joblib import Parallel, delayed
        from tqdm import tqdm
        results = Parallel(n_jobs = num_pro)(delayed(evaluate_scenario_SWMM_new_design)(*task) for task in tasks)
    else:
        # Sequential processing
        results = [evaluate_scenario_SWMM_new_design(*task) for task in tasks]

    results = np.array(results)

    if save:
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, save_name + 'results.npy')
        np.save(path, results)

    return results

def results_analysis(i, results, network_info, n_obj, num_total_sample, num_scenario, rel_thresholds = 80, vol_thresholds = 500):
    """
    Analyze the results of the evaluation.
    Args:
    i (list or array): size of pipes
    results (np.ndarray): Results of the evaluation. (evaluate_uncertainty_eval_only_allinp, evaluate_uncertainty_new_design)
    network_info (dict): Network information including pipe names.
    n_obj (int): Number of objectives.
    num_total_sample (int): Number of samples.
    num_scenario (int): Number of scenarios.
    rel_thresholds (int): Reliability threshold.
    vol_thresholds (float): Volume threshold.
    """
    ## Reliability
    system_overflow = results
    reliability = (np.count_nonzero(system_overflow == 0)/num_total_sample)*100

    ## Robustness
    rob_count = 0
    num_sow = num_total_sample//num_scenario
    flooding_volume_list = []
    for k in range(num_scenario):
        tmp = system_overflow[k*num_sow:(k+1)*num_sow].copy()
        tmp_reliability = (np.count_nonzero(tmp == 0)/num_sow)*100
        tmp_mask = tmp > 0
        tmp_avg_system_flooding_volume = np.mean(tmp[tmp_mask]) if np.any(tmp_mask) else 0
        flooding_volume_list.append(tmp_avg_system_flooding_volume)
        if (tmp_reliability >= rel_thresholds) and (tmp_avg_system_flooding_volume <= vol_thresholds): 
            rob_count += 1
    avg_system_flooding_volume = np.mean(flooding_volume_list)
    robustness = rob_count/num_scenario*100

    cost = system_cost(i, network_info['pipe_length'])
    
    if n_obj == 2:
        return [reliability, cost]
    else:
        return [reliability, robustness, avg_system_flooding_volume, cost]
    
def design_evaluation(design, network_info, inp_folder, file_name, num_total_sample, num_scenario, n_obj=4, num_pro=8, save=False, save_name='', save_path='eval_results/', parallel=True, rel_thresholds=80, vol_thresholds=500):
    """Evaluate the designs of the pipe network."""
    results = evaluate_uncertainty_new_design(design, network_info, inp_folder, file_name, num_pro, save, save_name, save_path, parallel)
    objs = results_analysis(design, results, network_info, n_obj, num_total_sample, num_scenario, rel_thresholds=rel_thresholds, vol_thresholds=vol_thresholds)
    return objs




