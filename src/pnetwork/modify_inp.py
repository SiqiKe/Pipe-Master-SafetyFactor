"""
Modify '.inp' file

Main Functions:
- 
-
Note:
None

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

def read_inp(path):
    with open(path, 'r') as file:
        base_inp = file.readlines()
    return base_inp



def round_to_nearest_quarter(num):
    return round(num * 4) / 4

import math
def change_pipe_shape(base_file, new_filepath, write_new = False):
    """Change all 'closed rectangular' and 'horizontal elliptical' pipes into circular pipes"""
    
    lines = base_file.copy()

    start_index = -1
    end_index = -1
    # Find the [XSECTIONS] section
    for i, line in enumerate(lines):
        if "[XSECTIONS]" in line:
            start_index = i + 3  # Skip the header lines
        elif start_index != -1 and line.startswith('['):
            end_index = i
            break
    if start_index == -1:
        print("XSECTIONS section not found.")
        return

    # Update pipes' shape and geom
    for i in range(start_index, end_index if end_index != -1 else len(lines)):
        parts = lines[i].split()
        if len(parts) > 4:  # Ensure there are enough elements in the line
            if parts[1] == 'RECT_CLOSED' or parts[1] == 'RECT_OPEN':
                cross_section_area = float(parts[2])*float(parts[3])
                diameter = (cross_section_area/math.pi)**(1/2)*2
                parts[2] = str(round_to_nearest_quarter(diameter))
                
            elif parts[1] == 'HORIZ_ELLIPSE':
                cross_section_area = float(parts[2])*float(parts[3])
                diameter = cross_section_area**(1/2)
                parts[2] = str(round_to_nearest_quarter(diameter))
            else:
                pass
                
            parts[3] = '0'
            parts[1] = 'CIRCULAR'
            lines[i] = '     '.join(parts) + '\n'  # Reconstruct the line with updated value

    if write_new:
        # Write the modified lines into a new file
        with open(new_filepath, 'w') as file:
            file.writelines(lines)

    return lines


def change_pipe_FromTo(base_file, pipe_names, new_filepath, write_new = False):
    """
    Change all pipes that labeled with inverted FromEnd and ToEnd.
    
    # Example:
    Pipes need to change in Norfolk network:
    pipe_names = ['1075', '1165', '1167', '23574', '23646']
    """
    
    
    lines = base_file.copy()

    start_index = -1
    end_index = -1
    # Find the [CONDUITS] section
    for i, line in enumerate(lines):
        if "[CONDUITS]" in line:
            start_index = i + 3  # Skip the header lines
        elif start_index != -1 and line.startswith('['):
            end_index = i
            break
    if start_index == -1:
        print("CONDUITS section not found.")
        return

    # Exchange pipes' FromEnd and ToEnd
    for i in range(start_index, end_index if end_index != -1 else len(lines)):
        parts = lines[i].split()
        if len(parts) > 5:  # Ensure there are enough elements in the line
            if parts[0] in pipe_names:
                tem = parts[1]
                parts[1] = parts[2]
                parts[2] = tem
                lines[i] = '     '.join(parts) + '\n'  # Reconstruct the line with updated value
                # print('Changed', parts[0])

    if write_new:
        # Write the modified lines back to the file
        with open(new_filepath, 'w') as file:
            file.writelines(lines)

    return lines


def delete_pipe(base_file, pipe_names, new_filepath, write_new = False):
    """
    Delete pipes
    
    # Example:
    The pipe need to be deleted in Norfolk system (make a circle)
    pipe_names = ['11890']
    """
    
    lines = base_file.copy()

    start_index = -1
    end_index = -1
    # Find the [CONDUITS] section
    for i, line in enumerate(lines):
        if "[CONDUITS]" in line:
            start_index = i + 3  # Skip the header lines
        elif start_index != -1 and line.startswith('['):
            end_index = i
            break
    if start_index == -1:
        print("CONDUITS section not found.")
        return

    # Delete info
    for i in range(start_index, end_index if end_index != -1 else len(lines)):
        parts = lines[i].split()
        if len(parts) > 5:  # Ensure there are enough elements in the line
            if parts[0] in pipe_names:
                lines[i] = ''
                # print('Deleted from CONDUITS section', parts[0])

    start_index = -1
    end_index = -1
    # Find the [XSECTIONS] section
    for i, line in enumerate(lines):
        if "[XSECTIONS]" in line:
            start_index = i + 3  # Skip the header lines
        elif start_index != -1 and line.startswith('['):
            end_index = i
            break
    if start_index == -1:
        print("XSECTIONS section not found.")
        return

    # Delete the pipes
    for i in range(start_index, end_index if end_index != -1 else len(lines)):
        parts = lines[i].split()
        if len(parts) > 5:  # Ensure there are enough elements in the line
            if parts[0] in pipe_names:
                lines[i] = ''
                # print('Deleted from XSECTIONS section', parts[0])

    if write_new:
        # Write the modified lines back to the file
        with open(new_filepath, 'w') as file:
            file.writelines(lines)

    return lines

def delete_rainfall(base_file, rainfall_name, new_filepath, write_new = False):
    """
    Delete a rainfall time series
    
    # Example:
    rainfall_name = '24hr'
    """
    
    lines = base_file.copy()

    start_index = -1
    end_index = -1
    # Find the [TIMESERIES] section
    for i, line in enumerate(lines):
        if "[TIMESERIES]" in line:
            start_index = i + 3  # Skip the header lines
        elif start_index != -1 and line.startswith('['):
            end_index = i
            break
    if start_index == -1:
        print("TIMESERIES section not found.")
        return

    # Delete info
    for i in range(start_index, end_index if end_index != -1 else len(lines)):
        parts = lines[i].split()
        if len(parts) > 2:  # Ensure there are enough elements in the line
            if parts[0] == rainfall_name:
                lines[i] = ''

    if write_new:
        # Write the modified lines back to the file
        with open(new_filepath, 'w') as file:
            file.writelines(lines)

    return lines


def change_node_elevation(base_file, change, new_filepath, write_new = False):
    """
    Change all nodes' bottom elev. to avoid changing the slope and surface's elev.
    
    # Example: decrease 2 ft for each node's bottom elev. (increase 2 ft for each node's max depth)
    change = 2
    """
    
    lines = base_file.copy()

    start_index = -1
    end_index = -1
    # Find the [JUNCTIONS] section
    for i, line in enumerate(lines):
        if "[JUNCTIONS]" in line:
            start_index = i + 3  # Skip the header lines
        elif start_index != -1 and line.startswith('['):
            end_index = i
            break
    if start_index == -1:
        print("JUNCTIONS section not found.")
        return

    # Update junctions' elev.
    for i in range(start_index, end_index if end_index != -1 else len(lines)):
        parts = lines[i].split()
        if len(parts) > 3:  # Ensure there are enough elements in the line
            parts[1] = str(float(parts[1]) - change)
            parts[2] = str(float(parts[2]) + change)
            lines[i] = '     '.join(parts) + '\n'  # Reconstruct the line with updated value

    start_index = -1
    end_index = -1
    
    for i, line in enumerate(lines):
        if "[OUTFALLS]" in line:
            start_index = i + 3  # Skip the header lines
        elif start_index != -1 and line.startswith('['):
            end_index = i
            break
    if start_index == -1:
        print("OUTFALLS section not found.")
        return

    # Update junctions' elev.
    for i in range(start_index, end_index if end_index != -1 else len(lines)):
        parts = lines[i].split()
        if len(parts) > 3:  # Ensure there are enough elements in the line
            parts[1] = str(float(parts[1]) - change)
            lines[i] = '     '.join(parts) + '\n'  # Reconstruct the line with updated value


    if write_new:
        # Write the modified lines back to the file
        with open(new_filepath, 'w') as file:
            file.writelines(lines)

    return lines


def change_pipe_roughness(base_file, roughness, new_filepath, write_new = False):
    """
    Change all pipes' roughness to a certain number
    
    # Example:
    roughness = 0.013
    """
    
    lines = base_file.copy()

    start_index = -1
    end_index = -1
    # Find the [CONDUITS] section
    for i, line in enumerate(lines):
        if "[CONDUITS]" in line:
            start_index = i + 3  # Skip the header lines
        elif start_index != -1 and line.startswith('['):
            end_index = i
            break
    if start_index == -1:
        print("CONDUITS section not found.")
        return

    # Update conduits' roughness
    for i in range(start_index, end_index if end_index != -1 else len(lines)):
        parts = lines[i].split()
        if len(parts) >= 5:  # Ensure there are enough elements in the line
            parts[4] = str(roughness)
            lines[i] = '     '.join(parts) + '\n'  # Reconstruct the line with updated value

    if write_new:
        # Write the modified lines back to the file
        with open(new_filepath, 'w') as file:
            file.writelines(lines)

    return lines

def change_timestep(base_file, newfile_path,  report_step = '00:30:00', wet_weather = '00:05:00', dry_weather = '00:05:00',
                    routing_step = '00:00:05', min_step = 5, threads = 16, write_new=True):

    lines = base_file.copy()
    for i in range(len(lines)):
        # Modify REPORT_STEP
        if lines[i].startswith("REPORT_STEP"):
            parts = lines[i].split()
            if len(parts) > 1:
                parts[1] = str(report_step)
                lines[i] = " ".join(parts) + "\n"
        # Modify WET_WEATHER_STEP
        elif lines[i].startswith("WET_STEP"):
            parts = lines[i].split()
            if len(parts) > 1:
                parts[1] = str(wet_weather)
                lines[i] = " ".join(parts) + "\n"
        # Modify DRY_WEATHER_STEP
        elif lines[i].startswith("DRY_STEP"):
            parts = lines[i].split()
            if len(parts) > 1:
                parts[1] = str(dry_weather)
                lines[i] = " ".join(parts) + "\n"
        # Modify ROUTING_STEP
        elif lines[i].startswith("ROUTING_STEP"):
            parts = lines[i].split()
            if len(parts) > 1:
                parts[1] = str(routing_step)
                lines[i] = " ".join(parts) + "\n"
        # Modify MIN_STEP
        elif lines[i].startswith("MINIMUM_STEP"):
            parts = lines[i].split()
            if len(parts) > 1:
                parts[1] = str(min_step)
                lines[i] = " ".join(parts) + "\n"
        # Modify THREDS
        elif lines[i].startswith("THREADS"):
            parts = lines[i].split()
            if len(parts) > 1:
                parts[1] = str(threads)
                lines[i] = " ".join(parts) + "\n"
    # Write the modified lines to a new file

    if write_new:
        with open(newfile_path, 'w') as f:
            f.writelines(lines)
    
    return lines


import numpy as np
def add_uncertainties(base_file, new_filepath, rainfall_data, 
                                     subcatch = None, imperv = None, roughness = None, write_new = True):
    """ 
    Add new rainfall data, change subcatchments' %imperv, change pipes' roughness
    
    """
    #old name: write_uncertainty_inp_random
    #old name2:write_uncertainty_inp_subcatches
    lines = base_file.copy()

    ## rainfall
    start_index = -1
    for i, line in enumerate(lines):
        if "[TIMESERIES]" in line:
            start_index = i + 3
            break
    if start_index == -1:
        print("TIMESERIES section not found.")
        return
    # Inserting new data right after the section header
    for data_line in reversed(rainfall_data):  # Reverse to maintain order when inserting
        lines.insert(start_index, data_line)  # Ensure each line ends with a newline

    ## impervious area
    if imperv.any():
        
        start_index = -1
        end_index = -1
        # Find the [SUBCATCHMENTS] section
        for i, line in enumerate(lines):
            if "[SUBCATCHMENTS]" in line:
                start_index = i + 3  # Skip the header lines
            elif start_index != -1 and line.startswith('['):
                end_index = i
                break
        if start_index == -1:
            print("SUBCATCHMENTS section not found.")
            return
    
        # Update %Imperv values
        for i in range(start_index, end_index if end_index != -1 else len(lines)):
            parts = lines[i].split()
            if len(parts) > 4:  # Ensure there are enough elements in the line
                #print(parts[0], subcatch)
                for sub in range(len(subcatch)):
                    if parts[0] == subcatch[sub]:
                        parts[4] = str(imperv[sub])  # Change the %Imperv value ## set maximum to be 80%!!!!
                        lines[i] = '    '.join(parts) + '\n'  # Reconstruct the line with updated value

    ## pipe roughness coefficient
    if not np.isnan(roughness):
        
        start_index = -1
        end_index = -1
        # Find the [COUNDUITS] section
        for i, line in enumerate(lines):
            if "[CONDUITS]" in line:
                start_index = i + 3  # Skip the header lines
            elif start_index != -1 and line.startswith('['):
                end_index = i
                break
        if start_index == -1:
            print("CONDUITS section not found.")
            return
    
        # Update roughness coefficient
        for i in range(start_index, end_index if end_index != -1 else len(lines)):
            parts = lines[i].split()
            if len(parts) > 4:  # Ensure there are enough elements in the line
                parts[4] = str(roughness)  # Change the %Imperv value ## set maximum to be 80%!!!!
                lines[i] = '    '.join(parts) + '\n'  # Reconstruct the line with updated value
    
    if write_new:
        # Write the modified lines back to the file
        with open(new_filepath, 'w') as file:
            file.writelines(lines)

    return lines