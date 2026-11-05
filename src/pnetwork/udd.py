"""
Standard engineering methods for sizing circular pipes using Manning's equation and the Rational equation,
following the Urban Drainage Design code (USDT Federal Highway Administration, 2024).


Main Functions:
- manning_velocity: Compute mean velocity in a full circular pipe.
- manning_flow_rate: Compute flow rate in a full circular pipe.
- manning_diameter: Compute required diameter for a given velocity or flow.
- rational_formula_flow_rate: Compute flow rate using the Rational Formula.
- TR55_tc: Compute time of concentration using TR-55 method.
- build_pipe_network_standard: Build a pipe network as a tree/graph.
- update_values: Recursively update pipe node values for design.
- read_results: Summarize design results.

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""
import numpy as np

from .load_pipe import PipeNode, get_max_size
from .evaluation import get_sizes, overflow_swmm, system_cost

# --- Manning's Equation Utilities ---

def manning_velocity(diameter, slope, roughness_coef=0.010):
    """
    Calculate mean velocity using Manning's equation for full circular pipes.

    Args:
        diameter (float): Pipe diameter (ft).
        slope (float): Slope of energy grade line (ft/ft).
        roughness_coef (float): Manning's n (default 0.010).

    Returns:
        float: Mean velocity (ft/s).
    """
    Kv = 0.59  # CU
    return (Kv / roughness_coef) * (diameter ** 0.67) * (slope ** 0.5)

def manning_flow_rate(diameter, slope, roughness_coef=0.010):
    """
    Calculate flow rate using Manning's equation for full circular pipes.

    Args:
        diameter (float): Pipe diameter (ft).
        slope (float): Slope of energy grade line (ft/ft).
        roughness_coef (float): Manning's n (default 0.010).

    Returns:
        float: Flow rate (ft^3/s).
    """
    Kq = 0.46  # CU
    return (Kq / roughness_coef) * (diameter ** 2.67) * (slope ** 0.5)

def manning_diameter(value, slope, roughness_coef=0.010, rate_of_flow=True):
    """
    Calculate required diameter using Manning's equation.

    Args:
        value (float): Desired mean velocity or flow rate.
        slope (float): Slope of energy grade line (ft/ft).
        roughness_coef (float): Manning's n (default 0.010).
        rate_of_flow (bool): If True, use flow rate; else, use velocity.

    Returns:
        float: Required diameter (ft).
    """
    Kv = 0.59  # CU
    Kq = 0.46  # CU
    if rate_of_flow:
        return ((value * roughness_coef) / (Kq * (slope ** 0.5))) ** (1 / 2.67)
    else:
        return ((value * roughness_coef) / (Kv * (slope ** 0.5))) ** (1 / 0.67)

# --- Rational Formula ---

def rational_formula_flow_rate(runoff_coef, intensity, area):
    """
    Calculate flow rate using the Rational Formula.

    Args:
        runoff_coef (float): Runoff coefficient.
        intensity (float): Rainfall intensity (in/hr).
        area (float): Area (acres).

    Returns:
        float: Flow rate (ft^3/s).
    """
    Ku = 1  # CU
    return runoff_coef * intensity * area

# --- Time of Concentration Methods ---

def sheet_flow_tc(roughness_coeff, length, intensity_2yr_24hr, slope, duration=24):
    """
    Estimate sheet flow time of concentration (UDDM Eqn 4.3, FHWA 2002).

    Args:
        roughness_coeff (float): Surface roughness coefficient.
        length (float): Flow length (ft).
        intensity_2yr_24hr (float): Rainfall intensity (in/hr).
        slope (float): Surface slope (ft/ft).
        duration (float): Rainfall duration (hr).

    Returns:
        float: Travel time (min).
    """
    Ku = 0.42  # CU
    return (Ku / (intensity_2yr_24hr * duration) ** 0.5) * ((roughness_coeff * length / (slope ** 0.5)) ** 0.8)

def shallow_concentrated_flow_tc(length, slope, k=0.619):
    """
    Estimate shallow concentrated flow time of concentration (UDDM Eqn 4.4).

    Args:
        length (float): Flow length (ft).
        slope (float): Surface slope (ft/ft).
        k (float): Intercept coefficient (default 0.619 for paved).

    Returns:
        float: Travel time (min).
    """
    Ku = 3.28  # CU
    return length / (60 * Ku * k * slope * 100)

def sum_tc(roughness_coeff_list, length_list, intensity_2yr_24hr, slope_list, num_segments=2):
    """
    Estimate time of concentration for a subcatchment by picking the smallest value.

    Args:
        roughness_coeff_list (list): List of roughness coefficients.
        length_list (list): List of flow lengths (ft).
        intensity_2yr_24hr (float): Rainfall intensity (in/hr).
        slope_list (list): List of slopes (ft/ft).
        num_segments (int): Number of segments.

    Returns:
        float: Minimum travel time (min).
    """
    tc = [
        sheet_flow_tc(roughness_coeff_list[i], length_list[i], intensity_2yr_24hr, slope_list[i])
        for i in range(num_segments)
    ]
    return min(tc)

def TR55_tc(roughness_coeff, length, intensity_2yr_24hr, slope, k=0.619):
    """
    Estimate time of concentration using TR-55 method.

    Args:
        roughness_coeff (float): Surface roughness coefficient.
        length (float): Flow length (ft).
        intensity_2yr_24hr (float): Rainfall intensity (in/hr).
        slope (float): Surface slope (ft/ft).
        k (float): Intercept coefficient (default 0.619 for paved).

    Returns:
        float: Travel time (min).
    """
    tc = []
    if length <= 77:
        tc.append(sheet_flow_tc(roughness_coeff, length, intensity_2yr_24hr, slope))
    else:
        tc.append(sheet_flow_tc(roughness_coeff, 77, intensity_2yr_24hr, slope))
        tc.append(shallow_concentrated_flow_tc(length - 77, slope, k))
    return sum(tc)

def inverse_tc(runoff_in_per_hour, IDF):
    """
    Estimate time of concentration inverse from IDF curve.

    Args:
        runoff_in_per_hour (float): Water flow intensity (in/hr).
        IDF (list): [durations, intensities].

    Returns:
        float: Travel time (min).
    """
    for i in range(len(IDF[0])):
        if runoff_in_per_hour >= IDF[1][0]:
            return IDF[0][0]
        if runoff_in_per_hour <= IDF[1][i]:
            return IDF[0][i - 1]
    return IDF[0][-1]

def get_intensity(IDF, Tc):
    """
    Find rainfall intensity from IDF curve for a given time.

    Args:
        IDF (list): [durations, intensities].
        Tc (float): Time of concentration (min).

    Returns:
        float: Rainfall intensity (in/hr).
    """
    for i in range(len(IDF[0])):
        if Tc <= IDF[0][0]:
            return IDF[1][0]
        if Tc <= IDF[0][i]:
            return IDF[1][i - 1]
    return IDF[1][-1]

def get_diameter(commercial_diameter, min_diameter):
    """
    Select the smallest commercial diameter greater than or equal to min_diameter.

    Args:
        commercial_diameter (list): List of available diameters.
        min_diameter (float): Minimum required diameter.

    Returns:
        tuple: (index, selected diameter)
    """
    for i, d in enumerate(commercial_diameter):
        if min_diameter <= d:
            return i, d
    return len(commercial_diameter) - 1, commercial_diameter[-1]



# --- Pipe Node and Network Construction ---

class standard_PipeNode(PipeNode):
    """
    Extension of PipeNode for UDD design, storing additional hydraulic and subcatchment data.
    """
    def __init__(self, id, ind=None, inlet=False, subcatchment=None, slope=None, length=None):
        super().__init__(id, None)
        self.inlet = inlet #True if it is a inlet of a subcatchment; False, if not.
        self.self_subcatchment = subcatchment  # store info about the current subcatchment the node located: subcatchment id, runoff coefficient, area
        self.slope = slope #pipe's slope
        self.length = length #pipe's length
        self.all_subcatchment = {} # store the area of all upstream subcatchments. Key: subcatchment's name; Value: area
        self.upstream_tc = [] # store times of concerntration due to its upstream pipes, surface's tc is not included in the list
        self.flow_rate = [] # flow rate from upstream
        self.diameter = None #pipe's required diameter; self.ind is the eventually selected size


def get_slope(swmm_info, ind):
    """
    Get pipe length and slope from SWMM info.

    Args:
        swmm_info (dict): SWMM data.
        ind (int): Pipe index.

    Returns:
        tuple: (length, slope)
    """
    name, _, _, _, _, length = swmm_info['Conduits'].loc[ind, ['Name', 'FromNode','ToNode', 'InOffset','OutOffset', 'Length']]
    slope = float(swmm_info['Link Summary'][swmm_info['Link Summary']['Name'] == name]['%Slope'].values[0]) / 100
    return float(length), slope

def get_subcatchment(swmm_info, child, intensity):
    """
    Get subcatchment data for a node.

    Args:
        swmm_info (dict): SWMM data.
        child (str): Node name.
        intensity (float): 2-year, 24-hour rainfall intensity (in/hr).

    Returns:
        tuple: (is_inlet, subcatchment dict, area dict)
    """
    if child not in swmm_info['Subcatchments']['Outlet'].values:
        return False, None, {}
    sub_id, area, width, slope_per = swmm_info['Subcatchments'][swmm_info['Subcatchments']['Outlet'] == child][['Name','Area','Width','%Slope']].values[0]
    runoff_coef = swmm_info['Subcatchment Runoff Summary'][swmm_info['Subcatchment Runoff Summary']['Name'] == sub_id]['Runoff_Coeff'].values[0]
    sub_length = float(area) * 43560 / float(width)  # acre to sq ft
    sub_tc = TR55_tc(float(runoff_coef), sub_length, intensity, float(slope_per) / 100, k=0.619)
    return True, {'Name': sub_id, 'Area': float(area), 'Runoff_Coeff': float(runoff_coef), 'TOC': float(sub_tc)}, {sub_id: float(area)}

def build_pipe_network_standard(swmm_info, num_pipe, toc_intensity, IDF):
    """
    Build pipe network as a tree/graph for UDD design.

    Args:
        swmm_info (dict): SWMM data.
        num_pipe (int): Number of pipes.
        toc_intensity (float): Rainfall intensity for TOC (in/hr).
        IDF (list): [durations, intensities] for the target return period (e.g., 10-year) used for design.

    Returns:
        dict: Pipe network with root and nodes.
    """
    root_id = swmm_info['Outfalls'].Name.to_list()
    edges = [
        (swmm_info['Conduits'].loc[conduit].ToNode, swmm_info['Conduits'].loc[conduit].FromNode, conduit)
        for conduit in range(num_pipe)
    ]
    nodes = {}
    # build the tree
    for parent, child, _ in edges:
        if parent not in nodes:
            nodes[parent] = standard_PipeNode(parent)
        if child not in nodes:
            nodes[child] = standard_PipeNode(child)
        nodes[parent].add_child(nodes[child])
    # update values for all the node
    # parent: ToNode, child: FromNode, all pipes values are assigned to FromNode
    for _, child, ind in edges:
        nodes[child].add_ind(ind)
        nodes[child].inlet, nodes[child].self_subcatchment, nodes[child].all_subcatchment = get_subcatchment(
            swmm_info, child, toc_intensity)
        nodes[child].length, nodes[child].slope = get_slope(swmm_info, ind)
    return {'root': root_id, 'nodes': nodes}

def largest_half_less(n):
    """
    Find the largest number less than or equal to n that is a multiple of 0.5.

    Args:
        n (float): Input number.

    Returns:
        float: Largest 0.5-multiple <= n.
    """
    return (n // 0.5) * 0.5

def update_values(swmm_info, node, IDF, commercial_diameter, pipe_roughness_coef=0.013, report=False):
    """
    Recursively update pipe node values for UDD design.

    Args:
        swmm_info (dict): SWMM data.
        node (standard_PipeNode): Current node.
        IDF (list): [durations, intensities].
        commercial_diameter (list): Available diameters.
        pipe_roughness_coef (float): Manning's n.
    """
    if node is None:
        return
    for child in node.children:
        # if the pipe have upstream pipes
        update_values(swmm_info, child, IDF, commercial_diameter, pipe_roughness_coef)
    if node.parent:
        # if the pipe if not an outfall
        
        if node.inlet:
            # if the "new" water drains to the system
            
            Runoff_coeff = float(node.self_subcatchment['Runoff_Coeff'])
            Total_area = sum(node.all_subcatchment.values())
            Tc_upstream = sum(node.upstream_tc)
            Tc_self = node.self_subcatchment['TOC']
            final_TC = max(Tc_upstream, Tc_self)
            Intensity = get_intensity(IDF, final_TC)
            flow_rate = rational_formula_flow_rate(runoff_coef=Runoff_coeff, intensity=Intensity, area=Total_area)
            if not node.children:
                # only the most upstream most storm drain's inlet TC will be added to the system tc
                node.parent.upstream_tc.append(Tc_self)
        else:
            # if no "new" water drains to the system; we cannot calculate the flow_rate; We use the flow rate from upstreams)
            
            flow_rate = sum(node.flow_rate) # sum of upstream flow rates (in case there are merging branches)

        tem_slope = node.slope
        if node.slope < 0:
            tem_slope = 0.001
        require_min_diameter = manning_diameter(value = flow_rate, slope = tem_slope, roughness_coef = pipe_roughness_coef, rate_of_flow = True)
        
        if node.slope < 0:
            upstream_size = [i.diameter for i in node.children]
            if upstream_size: 
                require_min_diameter = min(require_min_diameter, max(upstream_size)) #min of the flat pipe and max over upstream
                
        max_size = get_max_size(swmm_info, node.ind)
        allowed_max_size = largest_half_less(max_size)
        allowed_min_diameter = min(require_min_diameter, allowed_max_size)
        size, diameter = get_diameter(commercial_diameter, allowed_min_diameter)
        if max_size < require_min_diameter and report:
            print('Min requeired size is bigger than largest commercial size, using the largest commercial size instead.')
        node.size = size
        node.diameter = require_min_diameter
        V_full_pipe = manning_velocity(diameter=diameter, slope=tem_slope, roughness_coef=pipe_roughness_coef)
        if V_full_pipe < 3 and report:
            print("Warning!!! V is less than 3", V_full_pipe)
        t_downstream = node.length / V_full_pipe / 60  # min
        node.parent.all_subcatchment.update(node.all_subcatchment)
        node.parent.upstream_tc.append(t_downstream) # only the directly connected upstream node's value will be added to the list
        node.parent.flow_rate.append(flow_rate) #only the directly connected upstream node's value will be added to the list


def read_results(network_info, pipe_tree, commercial_diameter, file_suffix='_udd'):
    """
    Summarize design results for all pipes.

    Args:
        network_info (dict): Network info.
        pipe_tree (dict): Pipe network.
        commercial_diameter (list): Available diameters.

    Returns:
        tuple: (results_diameter, results_min_diameter, overflow, cost)
    """
    results = np.zeros(network_info['num_pipe'])
    results_min_diameter = np.zeros(network_info['num_pipe'])
    for node in pipe_tree['nodes'].values():
        if node.parent:
            #only the nodes with a parent node are assigned with a pipe's info
            results[node.ind] = node.size
            results_min_diameter[node.ind] = node.diameter
    results_diameter = get_sizes(network_info['num_pipe'], results, commercial_diameter)
    overflow = overflow_swmm(network_info['num_pipe'], results_diameter, network_info['filepath'], network_info['pipe_name'], file_suffix=file_suffix)['system']
    cost = system_cost(results_diameter, network_info['pipe_length'])
    return results_diameter, results_min_diameter, overflow, cost