"""
GA methods

Main Class:
- 
-
Note:
None

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""
from pymoo.core.problem import ElementwiseProblem
from .evaluation import system_cost, evaluate_uncertainty_new_design


class ProblemUncertainty_udd(ElementwiseProblem):
    """
    Problem define
    """

    def __init__(self, n_var, n_obj, network_info, network_nodes, sub_network, size_choice_per_pipe, 
    inp_folder, inp_file_name, rel_thresholds = 80, vol_thresholds = 500, num_scenario = 8,
    swmm_file_suffix = '_ga',  num_processes = 4, save = False, save_path = 'saved_npy/', **kwargs):
        self.network_info = network_info
        self.network_nodes = network_nodes
        self.sub_network = sub_network
        self.size_choice_per_pipe = size_choice_per_pipe
        self.inp_folder = inp_folder
        self.inp_file_name = inp_file_name
        self.num_total_sample = len(inp_file_name)
        self.rel_thresholds = rel_thresholds
        self.vol_thresholds = vol_thresholds
        self.num_scenario = num_scenario
        self.swmm_file_suffix = swmm_file_suffix
        self.num_processes = num_processes
        self.save = save
        self.save_path = save_path
        super().__init__(n_var=n_var, n_obj=n_obj, xl=[0]*n_var, xu=[len(sub) - 1 for sub in size_choice_per_pipe], vtype = int, **kwargs)
        ## decision variables are the size level of pipes
        self.kwargs = kwargs

    def _evaluate(self, x, out, *args, **kwargs):
        i = get_sizes_per_pipe(self.n_var, x, self.size_choice_per_pipe)

        results = evaluate_uncertainty_new_design(i, 
            network_info = self.network_info, 
            inp_folder = self.inp_folder, file_name = self.inp_file_name, num_pro= self.num_processes, save=self.save, 
            save_name = f"eval_", save_path=self.save_path
        )

        ## Reliability
        system_overflow = results
        reliability = (np.count_nonzero(system_overflow == 0)/self.num_total_sample)*100
        
        # This method is not correct because the flooding volume of different scenario can not be averaged together.
        # mask = results > 0
        # avg_system_flooding_volume = np.mean(results[mask]) if np.any(mask) else 0
        
        ## Robustness
        rob_count = 0
        num_SOW = self.num_total_sample//self.num_scenario
        # for k in range(SOW):
        #     tmp = system_overflow[k*self.num_scenario:k*self.num_scenario+self.num_scenario].copy()
        #     tmp_reliability = (np.count_nonzero(tmp == 0)/self.num_scenario)*100
        #     tmp_volume = results[k*self.num_scenario:k*self.num_scenario+self.num_scenario].copy()
        #     tmp_avg_system_flooding_volume = np.mean(tmp_volume[tmp_volume > 0]) if np.any(tmp_volume > 0) else 0
        #     if (tmp_reliability >= self.rel_thresholds) and (tmp_avg_system_flooding_volume <= self.vol_thresholds): 
        #         rob_count += 1
        # robustness = rob_count/SOW*100
        flooding_volume_list = []
        for k in range(self.num_scenario):
            tmp = system_overflow[k*num_SOW:(k+1)*num_SOW].copy()
            tmp_reliability = (np.count_nonzero(tmp == 0)/num_SOW)*100
            tmp_mask = tmp > 0
            tmp_avg_system_flooding_volume = np.mean(tmp[tmp_mask]) if np.any(tmp_mask) else 0
            flooding_volume_list.append(tmp_avg_system_flooding_volume)
            if (tmp_reliability >= self.rel_thresholds) and (tmp_avg_system_flooding_volume <= self.vol_thresholds): 
                rob_count += 1
        avg_system_flooding_volume = np.mean(flooding_volume_list)
        robustness = rob_count/self.num_scenario*100
        
        cost = system_cost(i, self.network_info['pipe_length'])

        if self.n_obj == 2:
            out["F"] = [-reliability, cost]
        else:
            out["F"] = [-reliability, -robustness, avg_system_flooding_volume, cost]
        
class ProblemUncertainty_udd_FunctionSF(ElementwiseProblem):
    """
    Problem define
    """

    def __init__(self, network_info, network_nodes, size_choice_per_pipe, min_diameter_list, pipe_strahler,
    inp_folder, inp_file_name, rel_thresholds = 80, vol_thresholds = 500, num_scenario = 8, n_var = 3, n_obj = 4, min_para_sf = [0,0,0.2], max_para_sf = [2,2,5.0],
    swmm_file_suffix = '_ga',  num_processes = 4, save = False, save_path = 'saved_npy/', ssf_results = None,**kwargs):
        self.network_info = network_info
        self.network_nodes = network_nodes
        self.size_choice_per_pipe = size_choice_per_pipe
        self.min_diameter_list = np.array(min_diameter_list)
        self.pipe_strahler = pipe_strahler
        self.inp_folder = inp_folder
        self.inp_file_name = inp_file_name
        self.num_total_sample = len(inp_file_name)
        self.rel_thresholds = rel_thresholds
        self.vol_thresholds = vol_thresholds
        self.num_scenario = num_scenario
        self.swmm_file_suffix = swmm_file_suffix
        self.num_processes = num_processes
        self.save = save
        self.save_path = save_path
        self.ssf_results = ssf_results
        super().__init__(n_var=n_var, n_obj=n_obj, xl=min_para_sf, xu=max_para_sf, **kwargs)
        ## decision variables are the size level of pipes
        self.kwargs = kwargs

    def _evaluate(self, x, out, *args, **kwargs):
        i = FunctionSF(self.size_choice_per_pipe, self.min_diameter_list, x, self.network_info['pipe_length'], self.pipe_strahler)

        results = evaluate_uncertainty_new_design(i, 
            network_info = self.network_info, 
            inp_folder = self.inp_folder, file_name = self.inp_file_name, num_pro= self.num_processes, save=self.save, 
            save_name = f"eval_", save_path=self.save_path
        )

        ## Reliability
        system_overflow = results
        reliability = (np.count_nonzero(system_overflow == 0)/self.num_total_sample)*100
        
        ## Robustness
        rob_count = 0
        num_SOW = self.num_total_sample//self.num_scenario
        flooding_volume_list = []
        for k in range(self.num_scenario):
            tmp = system_overflow[k*num_SOW:(k+1)*num_SOW].copy()
            tmp_reliability = (np.count_nonzero(tmp == 0)/num_SOW)*100
            tmp_mask = tmp > 0
            tmp_avg_system_flooding_volume = np.mean(tmp[tmp_mask]) if np.any(tmp_mask) else 0
            flooding_volume_list.append(tmp_avg_system_flooding_volume)
            if (tmp_reliability >= self.rel_thresholds) and (tmp_avg_system_flooding_volume <= self.vol_thresholds): 
                rob_count += 1
        avg_system_flooding_volume = np.mean(flooding_volume_list)
        robustness = rob_count/self.num_scenario*100
        
        cost = system_cost(i, self.network_info['pipe_length'])

        if self.n_obj == 2:
            out["F"] = [-reliability, cost]
        else:
            out["F"] = [-reliability, -robustness, avg_system_flooding_volume, cost]


def FunctionSF_pipe(min_diameter, para_length, para_strahler, mean_length, mean_strahler, length, strahler, para_c = 1):
    return min_diameter * (para_length * (length - mean_length) / mean_length + para_strahler * (strahler - mean_strahler) / mean_strahler +  para_c)


def FunctionSF(commercial_diameter_per_pipe, min_diameter_list, sf, pipe_length, pipe_strahler):
    para_length, para_strahler, para_c= sf[0], sf[1], sf[2]
    mean_length = np.mean(pipe_length)
    mean_strahler = np.mean(pipe_strahler)

    num = len(min_diameter_list)
    results = np.zeros(num)
    for i in range(num):
        diameter = FunctionSF_pipe(min_diameter_list[i], para_length, para_strahler, mean_length, mean_strahler, pipe_length[i], pipe_strahler[i], para_c)
        temp =  get_diameter(commercial_diameter_per_pipe[i], diameter)
        if temp:
            results[i] = temp[1]
        else:
            results[i] = max(commercial_diameter_per_pipe[i])
    return results
        


def get_sizes_per_pipe(num_pipe, size_level_list, size_choice_per_pipe):
    """
    get the list of diameters from a list of size_level

    Args:
    num_pipe (int): the number of pipes
    size_level_list (list/array): size level of corresponding pipe
    size_choice_per_pipe (a list of lists): diameter choice per pipe

    Returns:
    list/array: diameters of each pipe
    """
    diameters = np.zeros(num_pipe)
    for i in range(num_pipe):
        diameters[i] = size_choice_per_pipe[i][int(size_level_list[i])]
    return diameters

def get_sizes_per_pipe_reverse(num_pipe, diameters, commercial_diameter):
    """
    Given a list of diameters, return the corresponding index for each pipe.
    
    Args:
    num_pipe (int): Number of pipes.
    diameters (list/array): List of diameters for each pipe.
    commercial_diameter (list): List of commercial diameters available for each pipe.

    Returns:
    list: List of diameters corresponding to the size levels.
    """
    size_levels = np.zeros(num_pipe)
    for i in range(num_pipe):
        if diameters[i] in commercial_diameter[i]:
            size_levels[i] = np.where(commercial_diameter[i] == diameters[i])[0][0]
        else:
            size_levels[i] = len(commercial_diameter[i]) - 1
            print(f"Warning: Diameter {diameters[i]} not found in commercial_diameter[{i}]. Using largest available size.")
    return size_levels

import bisect
    
def size_choice_per_pipe_udd(num_pipe, min_diameters, max_pipe_size, size_choice, num_choice_small, num_choice_large):
    size_choice_per_pipe = []
    num_all_choice = len(size_choice)
    for i in range(num_pipe):
        dia = min_diameters[i]
        max_dia = max_pipe_size[i]
        #print('\n', dia, max_dia)
        idx = bisect.bisect_left(size_choice, dia) # find the ind of a size which is the smallest size larger than min_diameter; return the last ind if dia is larger than all choice
        max_idx = max(bisect.bisect(size_choice, max_dia) - 1, 0)
        if max_dia <= dia:
            size_choice_per_pipe.append(size_choice[max(0, max_idx - num_choice_small - num_choice_large + 1) :max_idx + 1])
            #print('here',size_choice[max(0, max_idx - num_choice_small - num_choice_large + 1):max_idx + 1])
        else:
            start_idx = max(0, idx-num_choice_small)
            end_idx = min(start_idx + num_choice_small + num_choice_large, min(max_idx + 1, num_all_choice))
            # can not largger than the max pipe size
            if end_idx - start_idx == num_choice_small + num_choice_large:
                size_choice_per_pipe.append(size_choice[start_idx:end_idx])
                #print('2', size_choice[start_idx:end_idx])
            else:
                size_choice_per_pipe.append(size_choice[max(end_idx - 6, 0):end_idx])
                
                #print('3', size_choice[max(end_idx - 6, 0):end_idx])
    return size_choice_per_pipe

from .udd import get_diameter

def get_diameter_all(commercial_diameter, min_diameter_list):
    """
    Given a list of minimum diameters required, find the commercial size fulfilling the requirements. If no enough commercial diameter, choose the biggest one.
    """
    num = len(min_diameter_list)
    results = np.zeros(num)
    for i in range(num):
        temp =  get_diameter(commercial_diameter, min_diameter_list[i])
        if temp:
            results[i] = temp[1]
        else:
            results[i] = max(commercial_diameter)

    return results


def get_diameter_all_per_pipe(commercial_diameter_per_pipe, min_diameter_list):
    """
    Given a list of minimum diameters required, find the commercial size fulfilling the requirements. If no enough commercial diameter, choose the biggest one.
    """
    num = len(min_diameter_list)
    results = np.zeros(num)
    for i in range(num):
        temp =  get_diameter(commercial_diameter_per_pipe[i], min_diameter_list[i])
        if temp:
            results[i] = temp[1]
        else:
            results[i] = max(commercial_diameter_per_pipe[i])

    return results
        



from pymoo.core.sampling import Sampling
import numpy as np
    
class UDDSampling(Sampling):
    # UDD sampling: All pipes choose the size level from choices suggested by UDD design. 
    def _do(self, problem, n_samples, **kwargs):
        
        n, n_choices= problem.n_var, len(problem.size_choice_per_pipe[0])
        if n_samples <= n_choices:
            return np.vstack([np.ones(n, dtype = int)* k for k in range(n_samples)])
        else:
            same_level = np.vstack([np.ones(n, dtype = int)* k for k in range(n_choices)])
            rand = np.vstack([np.random.randint(0, n_choices, size=n) for k in range(n_samples-n_choices)])
            return np.vstack((same_level, rand))

class UDD_FloatRandomSampling(Sampling):

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):

        X = np.random.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X

        if problem.ssf_results is not None:
            ssf = problem.ssf_results
            n_ssf = len(ssf)
            if n_samples <= 3*n_ssf:
                X[:n_samples//3, :2] = 0
                X[:n_samples//3, 2] = ssf[:n_samples//3]
            else: 
                X[:n_ssf, :2] = 0
                X[:n_ssf, 2] = ssf
        
        return X


class UDDSamplingWithRepair(Sampling):
    # UDD sampling with repair: All pipes choose the size level from choices suggested by UDD design. 
    # After sampling, we check the feasibility by strahler number and repair if not feasible. The repairing method is to change the size level to be smaller than the parent node's size level + a relaxing level (e.g. 2).

    def feasibility_check_by_strahler(self, i, node, larger_strahler_record, smaller_strahler_record, relaxing_level = 2):
        # example: feasibility_check_by_strahler(i, tree_root, [a_big_size_level], [a_big_size_level])
        smaller_strahler_record.append(i[node.ind])
        if not i[node.ind] <= min(larger_strahler_record) + relaxing_level:
            return False      
        for child in node.children:
            if node.strahler_number > child.strahler_number:
                if not self.feasibility_check_by_strahler(i.copy(), child, smaller_strahler_record.copy(), smaller_strahler_record.copy(), relaxing_level):
                    return False
            else:
                if not self.feasibility_check_by_strahler(i.copy(), child, larger_strahler_record.copy(), smaller_strahler_record.copy(), relaxing_level):
                    return False          
        return True

    def repair_lowerCost_by_strahler(self, i, node, larger_strahler_record, smaller_strahler_record, relaxing_level = 2):
        # repair_lowerCost_by_strahler(tree_root,[a_big_size_level],[a_big_size_level])  
        if not i[node.ind] <= min(larger_strahler_record) + relaxing_level:
            i[node.ind] = int(min(larger_strahler_record)+relaxing_level)
        smaller_strahler_record.append(i[node.ind])
        
        for child in node.children:
            if node.strahler_number > child.strahler_number:
                i = self.repair_lowerCost_by_strahler(i.copy(), child, smaller_strahler_record.copy(), smaller_strahler_record.copy(), relaxing_level)
            else:
                i = self.repair_lowerCost_by_strahler(i.copy(), child, larger_strahler_record.copy(), smaller_strahler_record.copy(), relaxing_level)
        return i 

    def _do(self, problem, n_samples, **kwargs):
        n, n_choices= problem.n_var, len(problem.size_choice_per_pipe[0])
        if n_samples <= n_choices:
            return np.vstack([np.ones(n, dtype = int)* k for k in range(n_samples)])
        else:
            same_level = np.vstack([np.ones(n, dtype = int)* k for k in range(n_choices)])
            rand = np.vstack([np.random.randint(0, n_choices, size=n) for k in range(n_samples-n_choices)])

            # fix rand

            nodes = problem.network_nodes['nodes']
            root = nodes[problem.network_nodes['root'][0]] #the outfall node don't have ind, we need its child node
            root_child = root.children[0]
            max_level = int(n_choices + 1)
            #print(Z.shape)

            # now repair each indvidiual i
            for i in range(len(rand)):
                # the packing plan for i
                z = rand[i]
    
                if not self.feasibility_check_by_strahler(rand[i], root_child, [max_level], [max_level]):
                    #print('False', rand[i])
                    rand[i] = self.repair_lowerCost_by_strahler(rand[i], root_child, [max_level], [max_level])
                    #print('New', rand[i])
                
            return np.vstack((same_level, rand))





from pymoo.core.crossover import Crossover

class SubTreeCrossover(Crossover):
    # Sub tree exchange
    def __init__(self, **kwargs):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var) **n_matings is the population size
        n_parents, n_matings, n_var = X.shape
        sub_network_len = len(problem.sub_network)

        # The output with the shape (n_offsprings, n_matings, n_var)
        Y = X.copy()

        # for each mating provided
        for k in range(n_matings):
            crossover_loc = problem.sub_network[np.random.randint(0, sub_network_len)]

            # get the value from first and the second parent
            a, b = X[0, k, crossover_loc], X[1, k, crossover_loc]

            Y[0, k, crossover_loc], Y[1, k, crossover_loc] = b,a

        return Y



import numpy as np
from pymoo.core.repair import Repair

class StrahlerRepair(Repair):

    def __init__(self, relaxing_level = 0):
        super().__init__()
        self.relaxing_level = relaxing_level
        

    def feasibility_check_by_strahler(self, i, node, larger_strahler_record, smaller_strahler_record, relaxing_level):
        # example: feasibility_check_by_strahler(i, tree_root, [a_big_size_level], [a_big_size_level])
        smaller_strahler_record.append(i[node.ind])
        if not i[node.ind] <= min(larger_strahler_record) + relaxing_level:
            return False      
        for child in node.children:
            if node.strahler_number > child.strahler_number:
                if not self.feasibility_check_by_strahler(i.copy(), child, smaller_strahler_record.copy(), smaller_strahler_record.copy(), relaxing_level):
                    return False
            else:
                if not self.feasibility_check_by_strahler(i.copy(), child, larger_strahler_record.copy(), smaller_strahler_record.copy(), relaxing_level):
                    return False          
        return True

    def repair_lowerCost_by_strahler(self, i, node, larger_strahler_record, smaller_strahler_record, relaxing_level):
        # repair_lowerCost_by_strahler(tree_root,[a_big_size_level],[a_big_size_level])  
        if not i[node.ind] <= min(larger_strahler_record) + relaxing_level:
            i[node.ind] = int(min(larger_strahler_record)+relaxing_level)
        smaller_strahler_record.append(i[node.ind])
        
        for child in node.children:
            if node.strahler_number > child.strahler_number:
                i = self.repair_lowerCost_by_strahler(i.copy(), child, smaller_strahler_record.copy(), smaller_strahler_record.copy(), relaxing_level)
            else:
                i = self.repair_lowerCost_by_strahler(i.copy(), child, larger_strahler_record.copy(), smaller_strahler_record.copy(), relaxing_level)
        return i 
    

        

    def _do(self, problem, Z, **kwargs):
        nodes = problem.network_nodes['nodes']
        root = nodes[problem.network_nodes['root'][0]] #the outfall node don't have ind, we need its child node
        root_child = root.children[0]
        max_level = int(len(problem.size_choice) + 1)
        #print(Z.shape)

        # now repair each indvidiual i
        for i in range(len(Z)):
            # the packing plan for i
            z = Z[i]

            if not self.feasibility_check_by_strahler(Z[i], root_child, [max_level], [max_level], self.relaxing_level):
                #print('False', Z[i])
                Z[i] = self.repair_lowerCost_by_strahler(Z[i], root_child, [max_level], [max_level], self.relaxing_level)
                #print('New', Z[i])


        return Z




## Termination

### Copy from pymoo source code. Adding a print commend to allow checking value during runing.

import numpy as np

from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.util.normalization import normalize
from pymoo.termination.delta import DeltaToleranceTermination


def calc_delta(a, b):
    return np.max(np.abs((a - b)))


def calc_delta_norm(a, b, norm):
    return np.max(np.abs((a - b) / norm))


class MultiObjectiveSpaceTermination(DeltaToleranceTermination):

    def __init__(self, tol=0.0025, only_feas=True, **kwargs):
        super().__init__(tol, **kwargs)
        self.delta_ideal = None
        self.delta_nadir = None
        self.delta_f = None
        self.only_feas = only_feas

    def _data(self, algorithm):
        feas, F = algorithm.opt.get("feas", "F")

        if self.only_feas:
            F = F[feas]

        if len(F) > 0:
            return dict(ideal=F.min(axis=0), nadir=F.max(axis=0), F=F, feas=True)
        else:
            return dict(ideal=None, nadir=None, F=F, feas=False)

    def _delta(self, prev, current):

        if not (prev["feas"] and current["feas"]):
            return np.inf

        # this is the range between the nadir and the ideal point
        norm = current["nadir"] - current["ideal"]

        # if the range is degenerated (very close to zero) - disable normalization by dividing by one
        norm[norm < 1e-32] = 1.0

        # calculate the change from last to current in ideal and nadir point
        delta_ideal = calc_delta_norm(current["ideal"], prev["ideal"], norm)
        delta_nadir = calc_delta_norm(current["nadir"], prev["nadir"], norm)

        # get necessary data from the current population
        c_F, c_ideal, c_nadir = current["F"], current["ideal"], current["nadir"]
        p_F = prev["F"]

        # normalize last and current with respect to most recent ideal and nadir
        c_N = normalize(c_F, c_ideal, c_nadir)
        p_N = normalize(p_F, c_ideal, c_nadir)

        # calculate IGD from one to another
        delta_f = IGD(c_N).do(p_N)

        # store the delta values to the object
        self.delta_ideal, self.delta_nadir, self.delta_f = delta_ideal, delta_nadir, delta_f

        # print(delta_ideal, delta_nadir, delta_f)

        return max(delta_ideal, delta_nadir, delta_f)



from pymoo.core.termination import Termination
from pymoo.termination.cv import ConstraintViolationTermination
#from pymoo.termination.ftol import SingleObjectiveSpaceTermination, MultiObjectiveSpaceTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.xtol import DesignSpaceTermination


class DefaultTermination(Termination):

    def __init__(self, x, cv, f, n_max_gen=1000, n_max_evals=100000) -> None:
        super().__init__()
        self.x = x
        self.cv = cv
        self.f = f

        self.max_gen = MaximumGenerationTermination(n_max_gen)
        self.max_evals = MaximumFunctionCallTermination(n_max_evals)

        self.criteria = [self.x, self.cv, self.f, self.max_gen, self.max_evals]

    def _update(self, algorithm):
        p = [criterion.update(algorithm) for criterion in self.criteria]
        return max(p)



class DefaultMultiObjectiveTermination(DefaultTermination):

    def __init__(self, xtol=0.0005, cvtol=1e-3, ftol=0.005, n_skip=3, period=50, **kwargs) -> None:
        x = RobustTermination(DesignSpaceTermination(tol=xtol, n_skip=n_skip), period)
        cv = RobustTermination(ConstraintViolationTermination(cvtol, terminate_when_feasible=False, n_skip=n_skip), period)
        f = RobustTermination(MultiObjectiveSpaceTermination(ftol, only_feas=True, n_skip=n_skip), period)
        super().__init__(x, cv, f, **kwargs)



import pandas as pd
import os
import numpy as np
from pymoo.indicators.hv import Hypervolume
import json

def GA_history(problem, res, name, num_pipe, commercial_diameter, save_path = 'ga_history/', ref_point = [0,0,1000,1000000]):
    # Access the history
    runtime = res.exec_time
    history = res.history
    X_opt, F_opt = res.opt.get("X", "F")
    
    # Initialize lists to collect data
    data = []
    history_F = []
    n_evals = []

    # Collect all decision variables and objective values from the history
    for generation in history:
        # `X` is the matrix of decision variables in that generation
        # `F` is the matrix of objective values in that generation
        X = generation.pop.get("X")
        F = generation.pop.get("F")

        history_F.append(generation.opt.get("F"))

        n_evals.append(generation.evaluator.n_eval)

        # Combine X and F into a single list of rows and add a generation identifier
        for x, f in zip(X, F):
            x_size = get_sizes_per_pipe(num_pipe, x, commercial_diameter)
            data.append(list(x_size) + list(f) + [generation.n_gen])
    
    approx_ideal = F_opt.min(axis=0)
    approx_nadir = F_opt.max(axis=0)
    #print(approx_ideal, approx_nadir)
    metric = Hypervolume(ref_point= ref_point,
                     norm_ref_point=False,
                     ideal=approx_ideal,
                     nadir=approx_nadir)

    hv = [metric.do(_f) for _f in history_F]

    # Create a DataFrame with appropriate column names
    # The number of decision variables and objectives may vary based on the problem
    # Adjust the column names accordingly

    n_var = problem.n_var  # Number of decision variables
    n_obj = problem.n_obj  # Number of objectives

    # Define column names for decision variables and objective values
    columns = [f"x{i+1}" for i in range(n_var)] + [f"f{i+1}" for i in range(n_obj)] + ["generation"]

    # Create DataFrame
    df_all = pd.DataFrame(data, columns=columns)
    
    data_pf = []
    for x, f in zip(res.X, res.F):
        x_size = get_sizes_per_pipe(num_pipe, x, commercial_diameter)
        data_pf.append(list(x_size) + list(f))
        
    df_pf = pd.DataFrame(data_pf, columns = columns[:-1])
    os.makedirs(save_path, exist_ok=True)
    df_all.to_csv(save_path + 'pop_' + name + '.csv')
    df_pf.to_csv(save_path + 'pf_' + name + '.csv')
    # Save runtime, hv, and n_evals to a file
    summary = {
        "runtime": runtime,
        "hv": hv,
        "n_evals": n_evals
    }
    with open(os.path.join(save_path, f'summary_{name}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return df_all, df_pf, hv, n_evals


def GA_FunctionSF_history(problem, res, name, num_pipe, min_diameters, pipe_length, pipe_strahler, commercial_diameter, save_path='ga_history/', ref_point=[0, 0, 1000, 1000000]):
    """
    Collects the history of the GA process for the FunctionSF approach and calculates hypervolume (HV).

    Args:
        problem: The optimization problem.
        res: The result object from the GA process.
        name (str): Name for saving the results.
        num_pipe (int): Number of pipes.
        min_diameters (list): Minimum diameters for pipes.
        pipe_length (list): Length of pipes.
        pipe_strahler (list): Strahler numbers for pipes.
        commercial_diameter (list): Commercial diameter options.
        save_path (str): Path to save the results.
        ref_point (list): Reference point for hypervolume calculation.

    Returns:
        tuple: DataFrames for all generations, Pareto front, HV values, and evaluations.
    """
    runtime = res.exec_time
    history = res.history
    X_opt, F_opt = res.opt.get("X", "F")

    # Initialize lists to collect data
    data = []
    history_F = []
    n_evals = []

    # Collect all decision variables and objective values from the history
    for generation in history:
        X = generation.pop.get("X")
        F = generation.pop.get("F")

        history_F.append(generation.opt.get("F"))
        n_evals.append(generation.evaluator.n_eval)

        for x, f in zip(X, F):
            if problem.n_var == 2:
                x_3 = [x[0], x[1], 0]
                x_size = FunctionSF(commercial_diameter, min_diameters, x_3, pipe_length, pipe_strahler)
            else:
                x_size = FunctionSF(commercial_diameter, min_diameters, x, pipe_length, pipe_strahler)

            data.append(list(x) + list(x_size) + list(f) + [generation.n_gen])

    # Calculate hypervolume
    approx_ideal = F_opt.min(axis=0)
    approx_nadir = F_opt.max(axis=0)
    metric = Hypervolume(ref_point=ref_point, norm_ref_point=False, ideal=approx_ideal, nadir=approx_nadir)
    hv = [metric.do(_f) for _f in history_F]

    # Create DataFrame
    n_obj = problem.n_obj
    if problem.n_var == 2:
        columns = ['Para Length', 'Para Strahler'] + [f"x{i+1}" for i in range(num_pipe)] + [f"f{i+1}" for i in range(n_obj)] + ["generation"]
    else:
        columns = ['Para Length', 'Para Strahler', 'Para C'] + [f"x{i+1}" for i in range(num_pipe)] + [f"f{i+1}" for i in range(n_obj)] + ["generation"]

    df_all = pd.DataFrame(data, columns=columns)

    data_pf = []
    for x, f in zip(res.X, res.F):
        if problem.n_var == 2:
            x_3 = [x[0], x[1], 0]
            x_size = FunctionSF(commercial_diameter, min_diameters, x_3, pipe_length, pipe_strahler)
        else:
            x_size = FunctionSF(commercial_diameter, min_diameters, x, pipe_length, pipe_strahler)
        data_pf.append(list(x) + list(x_size) + list(f))

    df_pf = pd.DataFrame(data_pf, columns=columns[:-1])
    os.makedirs(save_path, exist_ok=True)
    df_all.to_csv(os.path.join(save_path, f'FunctionSF_pop_{name}.csv'))
    df_pf.to_csv(os.path.join(save_path, f'FunctionSF_pf_{name}.csv'))
    # Save runtime, hv, and n_evals to a file
    summary = {
        "runtime": runtime,
        "hv": hv,
        "n_evals": n_evals
    }
    with open(os.path.join(save_path, f'summary_{name}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return df_all, df_pf, hv, n_evals