# Interconnect estimation of SIAM tool
import os, re, glob, sys, math
import numpy as np
import pandas as pd
from subprocess import call
from pathlib import Path
import math


def interconnect_estimation():
    homepath = os.getcwd()
    # print(homepath)
    homepath += '/MNSIM/NoC'

    network_type, arch_type = obtain_interconnect_spec(homepath)
    # arch_type = 'serial'
    # print(arch_type)
    num_layers, num_tiles_per_layer, ip_activation_per_tile, volume_per_tile = create_injection_rate(network_type,
                                                                                                     arch_type, homepath)
    latency_array, NoC_latency = interconnect_latency_estimation(homepath)
    interconnect_latency = postprocess_latency_array(num_layers, num_tiles_per_layer, ip_activation_per_tile,
                                                     volume_per_tile, latency_array, network_type, arch_type)
    interconnect_area, interconnect_power = interconnect_area_power_estimation(num_tiles_per_layer, homepath)

    area_file = open(homepath + '/Final_Results/area.csv', 'a')
    area_file.write('Total interconnect area is ' + interconnect_area + ' um^2')
    area_file.close()

    power_file = open(homepath + '/Final_Results/Energy.csv', 'a')
    power_file.write(
        'Total interconnect energy is ' + str(float(interconnect_power) * float(interconnect_latency)) + ' pJ')
    power_file.close()

    latency_file = open(homepath + '/Final_Results/Latency.csv', 'a')
    latency_file.write('Total interconnect latency is ' + str(interconnect_latency) + ' ns')
    latency_file.close()

    return NoC_latency


def obtain_interconnect_spec(homepath):
    spec_file = '/interconnect_spec_file.txt'
    arch_type = os.popen('grep "arch_type" ' + homepath + spec_file + ' | tail -1 | awk \'{print $3}\'').read().strip()
    network_type = os.popen('grep "network_type" ' + homepath + spec_file + ' | tail -1 | awk \'{print $3}\'').read().strip()

    return network_type, arch_type


# Area and power estimation for interconnect
def interconnect_area_power_estimation(num_tiles_per_layer, homepath):
    num_tile_total = np.sum(num_tiles_per_layer)
    # mesh_size = 10
    mesh_size = int(math.sqrt(num_tile_total))

    # Open read file handle of config file
    fp = open(homepath + '/mesh_config_dummy', 'r')

    # Set path to config file
    config_file = homepath + '/mesh_config'

    # Open write file handle for config file
    outfile = open(config_file, 'w')

    for line in fp:

        line = line.strip()

        # Search for pattern
        matchobj = re.match(r'^k=', line)

        # Set size of mesh if line in file corresponds to mesh size
        if matchobj:
            line = 'k=' + str(mesh_size) + ';'

        # Write config to file
        outfile.write(line + '\n')

    # Close file handles
    fp.close()
    outfile.close()

    # Run Booksim with config file and save log
    log_file = homepath + '/dummy_output.log'
    booksim_command = homepath + '/booksim ' + config_file + ' > ' + log_file
    os.system(booksim_command)

    # Grep for area
    # latency = os.popen('grep "Packet latency average" ' + log_file + ' | tail -1 | awk \'{print $5}\'').read().strip()
    area = os.popen('grep "Total Area" ' + log_file + ' | tail -1 | awk \'{print $4}\'').read().strip()

    print('[ INFO] Area: ' + area + '\n')

    power = os.popen('grep "Total Power" ' + log_file + ' | tail -1 | awk \'{print $4}\'').read().strip()

    print('[ INFO] Power: ' + power + '\n')

    return area, power


# Latency estimation for interconnect
def interconnect_latency_estimation(homepath):
    # trace_file_dir = "trace_dir"

    inj_rate_dir = homepath + "/inj_dir"
    NoC_latency = []

    # Initialize dictionary to hold latency values
    latency_dict = dict()

    # Get a list of all files in directory
    files = glob.glob(inj_rate_dir + '/*txt')
    # print(len(files))

    # Initialize file counter
    file_counter = 0

    # Create directory to store config files
    os.system('mkdir -p ' + homepath + '/logs/configs')

    # Iterate over all files
    for file in files:

        # Increment file counter
        file_counter += 1

        print('[ INFO] Processing file ' + file + ' ...')

        # Extract file name without extension and absolute path from filename
        run_name = os.path.splitext(os.path.basename(file))[0]

        # Extract first line of file
        line1 = os.popen('head -1 ' + file).read()

        # Extract size of mesh
        line1 = line1.strip()
        values = line1.split()
        mesh_size = int(math.sqrt(len(values)))

        if (mesh_size < 30):
            # Open read file handle of config file
            fp = open(homepath + '/mesh_config_inj_rate', 'r')

            # Set path to config file
            config_file = homepath + '/logs/configs/' + run_name + '_mesh_config'

            # Open write file handle for config file
            outfile = open(config_file, 'w')

            # Iterate over file and set size of mesh in config file
            for line in fp:

                line = line.strip()

                # Search for pattern
                matchobj = re.match(r'^k=', line)

                # Set size of mesh if line in file corresponds to mesh size
                if matchobj:
                    line = 'k=' + str(mesh_size) + ';'

                # Write config to file
                outfile.write(line + '\n')

            # Close file handles
            fp.close()
            outfile.close()

            # Set path to log file
            log_file = homepath + '/logs/' + run_name + '.log'

            # Copy injection rate matrix file
            os.system('cp ' + file + ' ' + homepath + '/inj_rate.txt')

            # Run Booksim with config file and save log
            booksim_command = homepath + '/booksim ' + config_file + ' > ' + log_file
            os.system(booksim_command)

            # Grep for packet latency average from log file
            latency = os.popen(
                'grep "Packet latency average" ' + log_file + ' | tail -1 | awk \'{print $5}\'').read().strip()
        # latency = os.popen('grep "Trace is finished in" ' + log_file + ' | tail -1 | awk \'{print $5}\'').read().strip()
        else:
            latency = str(mesh_size)

        print('[ INFO] Latency: ' + latency + '\n')

        # Add key, value to dictionary
        latency_dict[run_name] = latency

    # Open output file handle
    # print(homepath + '/logs/latency_mesh.csv')
    outfile = open(homepath + '/logs/latency_mesh.csv', 'w')

    latency_array = np.zeros(file_counter)

    # Write latencies to CSV
    for index in range(file_counter):
        # print(index)
        run_name = 'inj_rate_' + str(index)
        outfile.write(latency_dict[run_name] + '\n')
        NoC_latency.append(float(latency_dict[run_name]))
        # total_latency = total_latency + int(latency_dict[run_name])
        latency_array[index] = latency_dict[run_name]

    outfile.close()

    return latency_array, NoC_latency


def create_injection_rate(network_type, arch_type, homepath):
    injection_directory_name = homepath + '/inj_dir'
    dir_exist = os.path.isdir(injection_directory_name)
    if dir_exist == True:
        os.system('rm -rf ' + injection_directory_name)
    os.mkdir(injection_directory_name)

    quantization_bit = 1
    bus_width = 32
    freq = 1000000000

    num_tiles_per_layer = pd.read_csv(homepath + '/to_interconnect/num_tiles_per_layer.csv', header=None)
    ip_activation = pd.read_csv(homepath + '/to_interconnect/ip_activation.csv', header=None)
    fps = pd.read_csv(homepath + '/to_interconnect/fps.csv', header=None)
    num_layers = num_tiles_per_layer.size
    total_tiles = num_tiles_per_layer.sum()
    volume_per_tile = np.zeros(num_layers - 1)

    for layer_idx in range(num_layers - 1):
        volume_per_tile[layer_idx] = ((ip_activation.loc[layer_idx + 1][0] * quantization_bit + bus_width) * fps.loc[0][
            0]) / \
                                     (num_tiles_per_layer.loc[layer_idx][0] * num_tiles_per_layer.loc[layer_idx + 1][
                                         0])

    ip_activation_per_tile = np.zeros(num_layers - 1)
    for layer_idx in range(num_layers - 1):
        ip_activation_per_tile[layer_idx] = ip_activation.loc[layer_idx + 1][0] / (
                    num_tiles_per_layer.loc[layer_idx][0] * num_tiles_per_layer.loc[layer_idx + 1][0]);

    # assert(num_tiles_per_layer.size()) == num_layers);
    if (arch_type == 'parallel'):
        num_tiles_total = np.sum(num_tiles_per_layer)
        if (network_type == 'mesh'):
            NO_OF_ROWS = math.ceil(math.sqrt(num_tiles_total))
            NO_OF_COLS = NO_OF_ROWS
            num_node = NO_OF_ROWS * NO_OF_COLS
        elif (network_type == 'htree'):
            NO_OF_ROWS = math.ceil(math.log2(num_tiles_total))
            NO_OF_COLS = NO_OF_ROWS
            num_node = 2 ^ NO_OF_ROWS
        else:
            print('Network type not supported')
            assert (0)
        lambda_array = np.zeros((num_node, num_node))

    num_src_tiles = 0
    num_dest_tiles = 0

    for layer_idx in range(0, num_layers - 1):

        if (arch_type == 'serial'):
            num_tiles_till_prev_layer = 0
            num_src_tiles = num_tiles_per_layer.loc[layer_idx][0];
            num_dest_tiles = num_tiles_per_layer.loc[layer_idx + 1][0];
        elif (arch_type == 'parallel'):
            num_tiles_till_prev_layer = num_src_tiles
            num_src_tiles += num_tiles_per_layer.loc[layer_idx][0]
            num_dest_tiles = num_tiles_per_layer.loc[layer_idx + 1][0]
        else:
            print('Architecture type is not supported')
            assert (0)

        if (arch_type == 'serial'):
            if (network_type == 'mesh'):
                NO_OF_ROWS = math.ceil(math.sqrt(num_src_tiles + num_dest_tiles));
                NO_OF_COLS = NO_OF_ROWS;
                num_node = NO_OF_ROWS * NO_OF_COLS;
            elif (network_type == 'htree'):
                NO_OF_ROWS = math.ceil(math.log2(num_src_tiles + num_dest_tiles));
                NO_OF_COLS = NO_OF_ROWS;
                num_node = 2 ^ NO_OF_ROWS;
            else:
                print('Network type not supported')
                assert (0);

            lambda_array = np.zeros((num_node, num_node));

        for src_node in range(num_tiles_till_prev_layer, num_src_tiles):
            for dest_node in range((num_src_tiles), (num_src_tiles + num_dest_tiles)):
                lambda_array[src_node, dest_node] = (volume_per_tile[layer_idx] / bus_width) / freq;
                if (lambda_array[src_node, dest_node] < 0.00001):
                    lambda_array[src_node, dest_node] = 0.0001

        if (arch_type == 'serial'):
            os.chdir(injection_directory_name)
            filename = 'inj_rate_' + str(layer_idx) + '.txt'
            np.savetxt(filename, lambda_array, fmt='%.12f')
            os.chdir("..")

    if (arch_type == 'parallel'):
        os.chdir(injection_directory_name);
        filename = 'inj_rate_0.txt'
        np.savetxt(filename, lambda_array, fmt='%.12f')
        os.chdir("..")

    return num_layers, num_tiles_per_layer, ip_activation_per_tile, volume_per_tile


def postprocess_latency_array(num_layers, num_tiles_per_layer, ip_activation_per_tile, volume_per_tile, latency_array,
                              network_type, arch_type):
    quantization_bit = 1
    bus_width = 32
    freq = 1000000000

    avg_const_delay = np.zeros(num_layers - 1)
    per_layer_latency = np.zeros(num_layers - 1)
    effective_delay = np.zeros(num_layers - 1)

    if (arch_type == 'parallel'):
        num_tiles_total = np.sum(num_tiles_per_layer)
        if (network_type == 'mesh'):
            NO_OF_ROWS = math.ceil(math.sqrt(num_tiles_total))
            NO_OF_COLS = NO_OF_ROWS
            num_node = NO_OF_ROWS * NO_OF_COLS
        elif (network_type == 'htree'):
            NO_OF_ROWS = math.ceil(math.log2(num_tiles_total))
            NO_OF_COLS = NO_OF_ROWS
            num_node = 2 ^ NO_OF_ROWS
        else:
            print('Architecture type is not supported')
            assert (0)

        lambda_array = np.zeros((num_node, num_node))

    num_src_tiles = 0
    num_dest_tiles = 0

    for layer_idx in range(0, num_layers - 1):

        if (arch_type == 'serial'):
            num_tiles_till_prev_layer = 0
            num_src_tiles = num_tiles_per_layer.loc[layer_idx][0];
            num_dest_tiles = num_tiles_per_layer.loc[layer_idx + 1][0];
            avg_latency_layer = latency_array[layer_idx]
        elif (arch_type == 'parallel'):
            num_tiles_till_prev_layer = num_src_tiles
            num_src_tiles += num_tiles_per_layer.loc[layer_idx][0]
            num_dest_tiles = num_tiles_per_layer.loc[layer_idx + 1][0]
            avg_latency_layer = latency_array[0]
        else:
            print('Architecture type is not supported')
            assert (0)

        if (arch_type == 'serial'):
            if (network_type == 'mesh'):
                NO_OF_ROWS = math.ceil(math.sqrt(num_src_tiles + num_dest_tiles));
                NO_OF_COLS = NO_OF_ROWS;
                num_node = NO_OF_ROWS * NO_OF_COLS;
            elif (network_type == 'htree'):
                NO_OF_ROWS = math.ceil(math.log2(num_src_tiles + num_dest_tiles));
                NO_OF_COLS = NO_OF_ROWS;
                num_node = 2 ^ NO_OF_ROWS;
            else:
                print('Network type not supported')
                assert (0)

            lambda_array = np.zeros((num_node, num_node))

        for src_node in range(num_tiles_till_prev_layer, num_src_tiles):
            for dest_node in range((num_src_tiles), (num_src_tiles + num_dest_tiles)):
                lambda_array[src_node, dest_node] = (volume_per_tile[layer_idx] / bus_width) / freq;

        weighted_const_delay = 0

        for src_node in range(num_tiles_till_prev_layer, num_src_tiles):
            for dest_node in range((num_src_tiles), (num_src_tiles + num_dest_tiles)):
                src_row, src_col = extract_row_and_column_from_id(src_node, NO_OF_ROWS, NO_OF_COLS)
                dest_row, dest_col = extract_row_and_column_from_id(dest_node, NO_OF_ROWS, NO_OF_COLS)

                const_dist = abs(src_row - dest_row) + abs(src_col - dest_col)  # number of links
                const_pipeline_delay = 4 * (
                            const_dist + 1)  # number of routers visited is one more than number of links
                source_sink_delay = 3;

                total_const_delay = const_dist + const_pipeline_delay + source_sink_delay
                weighted_const_delay += lambda_array[src_node, dest_node] * total_const_delay

        avg_const_delay[layer_idx] = weighted_const_delay / np.sum(lambda_array)

        if (math.isnan(avg_latency_layer)):
            effective_delay[layer_idx] = 0
        else:
            effective_delay[layer_idx] = avg_latency_layer - avg_const_delay[layer_idx]

        if (effective_delay[layer_idx] < 0):
            effective_delay[layer_idx] = 0

        per_layer_latency[layer_idx] = (effective_delay[layer_idx] + 1) * (
                    ip_activation_per_tile[layer_idx] * quantization_bit + 32) / bus_width + avg_const_delay[layer_idx]

    if (arch_type == 'serial'):
        total_latency = np.sum(per_layer_latency)
    elif (arch_type == 'parallel'):
        total_latency = np.sum(per_layer_latency) / (num_layers - 1)
    else:
        total_latency = -1

    return total_latency


def extract_row_and_column_from_id(ID, NO_OF_ROWS, NO_OF_COLS):
    remainder = ID % NO_OF_COLS
    quotient = ID // NO_OF_COLS

    if remainder == 0:
        column = NO_OF_COLS
        row = quotient
    else:
        column = remainder
        row = quotient + 1

    return row, column


if __name__ == '__main__':
    interconnect_estimation()
