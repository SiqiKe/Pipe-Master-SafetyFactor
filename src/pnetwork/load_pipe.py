"""
This module helps to read inp files to gather info about pipes and its network.

Main Functions:
- read_inp_section(filepath, section_name, columns_name)
- read_inp(filepath, **kwargs)
- build_pipe_network(swmm_info, num_pipe, **kwarg)
- calculate_strahler_numbers(root)

Note:
- fixed the reading error in read_pipe when we have different column names in "outfall" section
- Updated read_inp_section for sections didn't defined (return None)
- Updated read_pipe for pumps, orifices and storages section
- Updated build_pipe_network for reading pumps, orifices section

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""


import pandas as pd


def read_inp_section(filepath, section_name, columns_name):

    """
    read a specific section info from .inp file and output as a pandas DataFrame

    Args:
    filepath (str): filepath.
    section_name (str): section name
    columns_name (str): columns name of that section

    Returns:
    df: DataFrame.
    """
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    section_data = []
    in_section = False

    try:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('[') and in_section:
                break
            if in_section:
                section_data.append(stripped_line)
            if stripped_line == section_name:
                in_section = True

        
        # Filter out comments and empty lines
        section_data = [line for line in section_data if line and not line.startswith(';')]
        
        data = [line.split() for line in section_data]
        return pd.DataFrame(data, columns=columns_name)
        
    except:
        return None


def read_pipe(inp_filepath, rpt_filepath, **kwargs) -> dict:
    """
    A Pre-set function to read specific sections info from .inp and .rpt file and output as seperate pandas DataFrame

    Args:
    filepath (str)

    Returns:
    dict: a dict storing info from sections Junctions, Conduits, Xsections, and Outfalls
    """
    junctions_name = ['Name', 'Elevation', 'MaxDepth', 'InitDepth', 'SurDepth', 'Aponded']
    storages_name = ['Name', 'Elev.', 'MaxDepth', 'InitDepth', 'Shape', 'Curve', 'Name/Params', 'N/A', 'Fevap', 'Psi']
    conduits_name = ['Name', 'FromNode', 'ToNode', 'Length', 'Roughness', 'InOffset', 'OutOffset', 'InitFlow', 'MaxFlow']
    pumps_name = ['Name', 'FromNode', 'ToNode', 'PumpCurve', 'Status', 'Sartup', 'Shutoff']
    orifices_name = ['Name', 'FromNode', 'ToNode', 'Type', 'Offset', 'Qcoeff', 'Gated', 'CloseTime']
    weirs_name = ['Name', 'FromNode', 'ToNode', 'Type', 'CrestHt', 'Qcoeff', 'Gated', 'EndCon', 'EndCoeff', 'Surcharge', 'RoadWidth', 'RoadSurf']
    xsections_name = ['Link','Shape','Geom1','Geom2','Geom3','Geom4','Barrels']
    outfalls_name = ['Name','Elevation','Type','Gated']
    subcatchments_name = ['Name', 'Rain Gage', 'Outlet', 'Area','%Imperv','Width','%Slope', 'CurbLen']
    subareas_name = ['Name', 'N-Imperv',   'N-Perv' ,'S-Imperv',   ',S-Perv',     'PctZero',    'RouteTo']

    # Read file
    junctions_df = read_inp_section(inp_filepath, '[JUNCTIONS]', junctions_name)
    storages_df = read_inp_section(inp_filepath, '[STORAGE]', storages_name)
        
    conduits_df = read_inp_section(inp_filepath, '[CONDUITS]', conduits_name)
    pumps_df = read_inp_section(inp_filepath, '[PUMPS]', pumps_name)
    orifices_df = read_inp_section(inp_filepath, '[ORIFICES]', orifices_name)
    weirs_df = read_inp_section(inp_filepath, '[WEIRS]', weirs_name)
    xsections_df = read_inp_section(inp_filepath, '[XSECTIONS]', xsections_name)

    if 'new_outfalls_name' not in kwargs:
        outfalls_df = read_inp_section(inp_filepath,'[OUTFALLS]', outfalls_name)
    else:
        outfalls_df = read_inp_section(inp_filepath,'[OUTFALLS]', kwargs['new_outfalls_name'])
        
    subcatchments_df = read_inp_section(inp_filepath,'[SUBCATCHMENTS]', subcatchments_name)
    subareas_df = read_inp_section(inp_filepath,'[SUBAREAS]', subareas_name)

    runoff_name = ["Name", "Total_Precip_in", "Total_Runon_in","Total_Evap_in", "Total_Infil_in", "Imperv_Runoff_in","Perv_Runoff_in", "Total_Runoff_in",
               "Total_Runoff_10^6_gal", "Peak_Runoff_CFS", "Runoff_Coeff"]
    runoff_df = read_rpt_section(rpt_filepath, 'Subcatchment Runoff Summary', runoff_name)
    
    links_name = ["Name", "From_Node", "To_Node", "Type", "Length", "%Slope", "Roughness"]
    links_df = read_rpt_section(rpt_filepath, 'Link Summary', links_name)
    
    info = {'Junctions': junctions_df, 'Storages': storages_df, 'Outfalls': outfalls_df, 'Conduits': conduits_df,  'Orifices': orifices_df,  'Xsections': xsections_df, 'Link Summary': links_df,
            'Subcatchments': subcatchments_df, 'Subareas': subareas_df, 'Subcatchment Runoff Summary': runoff_df, 'Pumps': pumps_df,'Weirs': weirs_df}

    num_pipe = len(conduits_df)
    num_pump = len(pumps_df)
    num_ori = len(orifices_df)
    num_weir = len(weirs_df)
    pipe_id = list(conduits_df.Name)
    node_id = list(junctions_df.Name) + list(storages_df.Name) + list(outfalls_df.Name)
    pipe_length = [float(x) for x in conduits_df.Length]
    pipe_init_diameter = list(xsections_df.Geom1)
    pipe_max_size = [get_max_size(info, i) for i in range(num_pipe)]
    avg_area = sum([float(area) for area in subcatchments_df.Area]) / len(subcatchments_df)
    
    network_info = {'num_pipe': num_pipe, 'pipe_name':pipe_id, 'pipe_length': pipe_length,'pipe_size': pipe_init_diameter, 
                    'node_name': node_id, 'filepath':inp_filepath, 'num_pump': num_pump, 
                    'num_ori': num_ori, 'num_weir':num_weir, 'pipe_max_size': pipe_max_size, 'avg_area': avg_area}
    return info, network_info


def read_rpt_section(filepath, section_name, columns_name):

    """
    read a specific section info from .rpt file and output as a pandas DataFrame

    Args:
    filepath (str): filepath.
    section_name (str): section name
    columns_name (str): columns name of that section

    Returns:
    df: DataFrame.
    """
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
    start_idx = None
    temp_start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if section_name in line:
            temp_start_idx = i
        if temp_start_idx and (i>temp_start_idx) and line.strip().startswith("-"):
            start_idx = i+1
        if start_idx and (i>start_idx) and line.strip() == "":
            end_idx = i
            break
    
    if start_idx is not None and end_idx is not None:
        data = lines[start_idx:end_idx]
    else:
        raise ValueError("Could not find the ", section_name, ' section.')

    parsed_data = []
    for line in data:
        parsed_data.append(line.split())

    return pd.DataFrame(parsed_data, columns=columns_name)



### Create PipeNode class to store information of each pipe
class PipeNode:
    def __init__(self, id, ind=None):
        self.id = id # manhole's name
        self.children = []  # connected manholes in the upstream
        self.parent = None  # the connected manhole in the downstream
        self.size = None  # the size level of its downstream pipe
        self.ind = ind  # the index of its downstream pipe
        self.strahler_number = None  # the strahler order its downstream pipe

    def add_child(self, child_node):
        # add a pipe node to its children list
        child_node.parent = self
        self.children.append(child_node)

    def add_ind(self, ind):
        self.ind = ind

    def add_size(self, size_ind):
        # add index of the corresponding diameter to the pipe node
        self.size = size_ind


def get_strahler_numbers_by_inds(root):
    """
    Given the root node and a list of node IDs, output a list of Strahler numbers
    in the same order as the node IDs.

    Args:
    root (PipeNode): The root node of the pipe network.
    node_ids (list): A list of node IDs.

    Returns:
    list: A list of Strahler numbers corresponding to the node IDs.
    """
    # Perform a DFS to map node IDs to their PipeNode objects
    ind_to_node = {}

    def dfs_ind_to_node(node):
        #print(node.ind)
        ind_to_node[node.ind] = node
        for child in node.children:
            dfs_ind_to_node(child)

    dfs_ind_to_node(root)
    #print(node_inds)
    # Retrieve Strahler numbers for the given node IDs
    return [ind_to_node[node_ind].strahler_number if node_ind in ind_to_node else None for node_ind in range(len(ind_to_node)-1)]


def calculate_strahler_numbers(root):
    """
    The calculate_strahler_numbers function follows the depth-first search rule to rank the pipe in the network following the strahler order method

    Args:
    root (PipeNode Class node): the root node of the pipe network
    """
    def dfs(node):
        #print("Calculating Strahler number for node: ", node.id)
        if not node.children:
            node.strahler_number = 1
            return 1
        
        child_strahler_numbers = [dfs(child) for child in node.children]
        max_strahler = max(child_strahler_numbers)
        if child_strahler_numbers.count(max_strahler) > 1:
            node.strahler_number = max_strahler + 1
        else:
            node.strahler_number = max_strahler
        #print(node.strahler_number)
        return node.strahler_number
    
    dfs(root)

def build_pipe_network(swmm_info, network_info, **kwargs):
    """
    A Pre-set function to build pipes info as tree/grapth

    Args:
    swmm_info (dict): a dict storing info read from inp file's sections
    num_pipe (int): number of pipes in the network

    Returns:
    dict: a dict storing info for the pipe network
    """
    
    root_id = swmm_info['Outfalls'].Name.to_list() # in case there are more than one root
    edges = [(swmm_info['Conduits'].loc[conduit].ToNode, swmm_info['Conduits'].loc[conduit].FromNode, conduit) for conduit in range(network_info['num_pipe'])]
    if network_info['num_pump'] > 0 :
        edges += [(swmm_info['Pumps'].loc[link].ToNode, swmm_info['Pumps'].loc[link].FromNode, num_pipe + link) for link in range(network_info['num_pump'])]
    if network_info['num_ori'] > 0 :
        edges += [(swmm_info['Orifices'].loc[link].ToNode, swmm_info['Orifices'].loc[link].FromNode, num_pipe + kwargs['num_pump'] + link) for link in range(network_info['num_ori'])]
    nodes = {}
    for parent, child, _ in edges:
        if parent not in nodes:
            nodes[parent] = PipeNode(parent)
        if child not in nodes:
            nodes[child] = PipeNode(child)
        nodes[parent].add_child(nodes[child])
    for _, child, ind in edges:
        nodes[child].add_ind(ind)
    pipe_network = {'root': root_id, 'nodes': nodes}
    return pipe_network

def get_max_size(swmm_info, ind):

    
    #pipe's maximum size based on its up and downstream's node
    
    name, fromN, toN, fromOff, toOff, length = swmm_info['Conduits'].loc[ind, ['Name', 'FromNode','ToNode', 'InOffset','OutOffset', 'Length']]
    fromMax = float(swmm_info['Junctions'][swmm_info['Junctions']['Name'] == fromN]['MaxDepth'].values[0])

    if toN in swmm_info['Junctions']['Name'].values:
        toMax = float(swmm_info['Junctions'][swmm_info['Junctions']['Name'] == toN]['MaxDepth'].values[0])
    else:
        toMax = 1000 # a big value
        
    return min(fromMax, toMax)



def get_all_pipe_ind(root):
    """ 
    Recursively collect all pipes' ind starting from the given root node. 

    Args:
    root (PipeNode object): the root node of the sub-network

    Returns:
    pipe_ind (list): a list of all pipes started from the root
    
    """
    if root.ind != None:
        pipe_ind = [root.ind]  # Start with the root node's ind
        for child in root.children:
            pipe_ind.extend(get_all_pipe_ind(child))  # Recursively add ind from children
        return pipe_ind
    else:
        return []

def get_all_pipe_ind_lowerStrahler(root, strahler):
    """ 
    Recursively collect all pipes' ind starting from the given root node with lower Straher order. 

    Args:
    root (PipeNode object): the root node of the sub-network

    Returns:
    pipe_ind (list): a list of all pipes started from the root
    
    """
    if root.ind != None:
        pipe_ind = []  
        for child in root.children:
            if child.strahler_number < strahler:
                pipe_ind.append(child.ind)
            pipe_ind.extend(get_all_pipe_ind_lowerStrahler(child, strahler))  # Recursively add ind from children
        return pipe_ind
    else:
        return []


def get_all_subnetwork(nodes):
    """ 
    A function to generates all sub-networks. 

    Args:
    nodes (a dict of PipeNode object): all PipeNodes of the network

    Returns:
    subtrees (list of list): a list of all sub-network pipes
    
    """
    subtrees = []
    for node in nodes.values():
        temp = get_all_pipe_ind(node)
        if temp:
            subtrees.append(temp)  
    return subtrees

def get_all_lowerStrahler(nodes):
    """ 
    A function to generates all sub-networks. 

    Args:
    nodes (a dict of PipeNode object): all PipeNodes of the network

    Returns:
    subtrees (list of list): a list of all sub-network pipes
    
    """
    subtrees = {}
    for node in nodes.values():
        temp = get_all_pipe_ind_lowerStrahler(node, node.strahler_number)
        if temp:
            subtrees[node.ind] = temp
    return subtrees

