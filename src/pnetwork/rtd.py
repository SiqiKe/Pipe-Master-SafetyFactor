"""
Rainfall Temporal Distribution

Main Functions:
- 
-
Note:
None

Author: Siqi Ke
Contact: siqi.ke.th@dartmouth.edu
License: MIT License
"""

from datetime import datetime, timedelta
import numpy as np
import os
import re

def read_nrcs_24hr():
    string_data = """depth=0.0 0.00101 0.00202 0.00305 0.00408 0.00513 0.00618 0.00725 0.00832
    depth=0.00941 0.01050 0.01161 0.01272 0.01385 0.01498 0.01613 0.01728 0.01845
    depth=0.01962 0.02081 0.02200 0.02321 0.02442 0.02565 0.02688 0.02813 0.02938
    depth=0.03065 0.03192 0.03321 0.03450 0.03581 0.03712 0.03845 0.03978 0.04113
    depth=0.04248 0.04385 0.04522 0.04661 0.04800 0.04941 0.05084 0.05229 0.05376
    depth=0.05525 0.05676 0.05829 0.05984 0.06141 0.06300 0.06461 0.06624 0.06789
    depth=0.06956 0.07125 0.07296 0.07469 0.07644 0.07821 0.08000 0.08181 0.08364
    depth=0.08549 0.08736 0.08925 0.09116 0.09309 0.09504 0.09701 0.09900 0.10101
    depth=0.10304 0.10509 0.10716 0.10925 0.11136 0.11349 0.11564 0.11781 0.12000
    depth=0.12225 0.12460 0.12705 0.12960 0.13225 0.13500 0.13785 0.14080 0.14385
    depth=0.14700 0.15020 0.15340 0.15660 0.15980 0.16300 0.16628 0.16972 0.17332
    depth=0.17708 0.18100 0.18512 0.18948 0.19408 0.19892 0.20400 0.20940 0.21520
    depth=0.22140 0.22800 0.23500 0.24268 0.25132 0.26092 0.27148 0.28300 0.30684
    depth=0.35436 0.43079 0.56786 0.66300 0.68196 0.69864 0.71304 0.72516 0.73500
    depth=0.74344 0.75136 0.75876 0.76564 0.77200 0.77796 0.78364 0.78904 0.79416
    depth=0.79900 0.80360 0.80800 0.81220 0.81620 0.82000 0.82367 0.82726 0.83079
    depth=0.83424 0.83763 0.84094 0.84419 0.84736 0.85047 0.85350 0.85647 0.85936
    depth=0.86219 0.86494 0.86763 0.87024 0.87279 0.87526 0.87767 0.88000 0.88229
    depth=0.88455 0.88679 0.88900 0.89119 0.89335 0.89549 0.89760 0.89969 0.90175
    depth=0.90379 0.90580 0.90779 0.90975 0.91169 0.91360 0.91549 0.91735 0.91919
    depth=0.92100 0.92279 0.92455 0.92629 0.92800 0.92969 0.93135 0.93299 0.93460
    depth=0.93619 0.93775 0.93929 0.94080 0.94229 0.94375 0.94519 0.94660 0.94799
    depth=0.94935 0.95069 0.95200 0.95330 0.95459 0.95588 0.95716 0.95844 0.95971
    depth=0.96098 0.96224 0.96350 0.96475 0.96600 0.96724 0.96848 0.96971 0.97094
    depth=0.97216 0.97338 0.97459 0.97580 0.97700 0.97820 0.97939 0.98058 0.98176
    depth=0.98294 0.98411 0.98528 0.98644 0.98760 0.98875 0.98990 0.99104 0.99218
    depth=0.99331 0.99444 0.99556 0.99668 0.99779 0.99890 1.0"""
    
    # Extract numbers using regular expressions
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", string_data)
    
    # Convert the strings to floats
    cumulate_pattern = [float(num) for num in numbers]
    return cumulate_pattern

cumulate = read_nrcs_24hr()


def nrcs_method_24hr_rainfall(intensity, cumulate_pattern = read_nrcs_24hr(), return_rainfall = True):
    """
    Given the hourly average intensity, output the temporal distribution of rainfall intensity
    """
    pattern_len = len(cumulate_pattern)
    pattern_ind = np.linspace(0,1,pattern_len)   
    
    name_list = ['24hr' for i in range(pattern_len)]

    total_intensity = intensity * 24 #24hour
    rainfall = [total_intensity*(cumulate_pattern[i+1]-cumulate_pattern[i]) /1*6
                for i in range(pattern_len-1)] # per 10 min --> per 1 hour

    if return_rainfall:
        return rainfall


def nrcs_method_24hr(outputfile_name, intensity, start_date, cumulate_pattern = read_nrcs_24hr(), return_rainfall = False):
    """
    Given the hourly average intensity, output the temporal distribution of rainfall intensity
    """
    pattern_len = len(cumulate_pattern)
    pattern_ind = np.linspace(0,1,pattern_len)   
    
    name_list = ['24hr' for i in range(pattern_len)]

    total_intensity = intensity * 24 #24hour
    rainfall = [total_intensity*(cumulate_pattern[i+1]-cumulate_pattern[i]) /1*6
                for i in range(pattern_len-1)] # per 10 min --> per 1 hour

    start_time = datetime.strptime(start_date, '%m/%d/%Y %H:%M')
    end_time = start_time + timedelta(hours = 24)      #24hr     
    resolution = timedelta(minutes=10) #10min based on cumulate_pattern
    time_list = []
    current_time = start_time
    while current_time < end_time:
        time_list.append(current_time.strftime('%m/%d/%Y %H:%M'))
        current_time += resolution

    
    with open(outputfile_name, 'w') as file:
        for item1, item2, item3 in zip(name_list, time_list, rainfall):
            # Write each item to the file
            file.write(f"{item1} {item2} {item3}\n")
        file.write(f";\n")

    if return_rainfall:
        return rainfall


def nrcs_method_24hr_dat(outputfile_name, intensity, start_date, cumulate_pattern = read_nrcs_24hr(), return_rainfall = False):
    """
    Given the hourly average intensity, output the temporal distribution of rainfall intensity (in a format readable in SWMM)
    """
    pattern_len = len(cumulate_pattern)
    pattern_ind = np.linspace(0,1,pattern_len)   
    
    name_list = ['24hr' for i in range(pattern_len)]

    total_intensity = intensity * 24 #24hour
    rainfall = [total_intensity*(cumulate_pattern[i+1]-cumulate_pattern[i]) /1*6
                for i in range(pattern_len-1)] # per 10 min --> per 1 hour

    start_time = datetime.strptime(start_date, '%m/%d/%Y %H:%M')
    end_time = start_time + timedelta(hours = 24)      #24hr     
    resolution = timedelta(minutes=10) #10min based on cumulate_pattern
    time_list = []
    current_time = start_time
    while current_time < end_time:
        time_list.append(current_time.strftime('%m/%d/%Y %H:%M'))
        current_time += resolution

    
    with open(outputfile_name, 'w') as file:
        for item1, item2 in zip(time_list, rainfall):
            # Write each item to the file
            file.write(f"{item1} {item2}\n")
        file.write(f";\n")

    if return_rainfall:
        return rainfall


def write_dat_file(rainfall_data, folder_name = 'intensity_dat', start_time = '01/01/2024 00:00'):
    """
    Generate a folder with .dat files, where are 24-h temporal distributed rainfall data given hourly average values in rainfall_data 
    """
    # Create the new directory
    os.makedirs(folder_name, exist_ok=True)
    for index, row in rainfall_data.iterrows():
        nrcs_method_24hr(folder_name + '/' + str(row.climate_id) +'.dat', row.intensity_in_hr, start_time, cumulate)


def write_dat_file_given_name(rainfall_data, name_column, folder_name = 'intensity_dat', start_time = '01/01/2025 00:00'):
    """
    Generate a folder with .dat files, where are 24-h temporal distributed rainfall data given hourly average values in rainfall_data
    """
    # Create the new directory
    os.makedirs(folder_name, exist_ok=True)
    for index, row in rainfall_data.iterrows():
        nrcs_method_24hr(folder_name + '/' + str(row[name_column]) +'.dat', row.intensity_in_hr, start_time, cumulate)