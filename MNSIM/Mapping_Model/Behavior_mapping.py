#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import configparser as cp
work_path = os.path.dirname(os.getcwd())
# print("ok", work_path)
sys.path.append(work_path)
from MNSIM.Hardware_Model import *
from MNSIM.Hardware_Model.Tile import tile
from MNSIM.Interface.interface import *


class behavior_mapping(tile):
    def __init__(self, NetStruct, SimConfig_path):
        self.SimConfig_path = SimConfig_path
        tile.__init__(self, SimConfig_path)
        bm_config = cp.ConfigParser()
        bm_config.read(SimConfig_path, encoding='UTF-8')
        self.xbar_polarity = int(bm_config.get('Process element level', 'Xbar_Polarity'))
        self.net_structure = NetStruct
        self.arch_config = SimConfig_path
        self.total_layer_num = len(self.net_structure)
        self.tile_list = []
        self.kernel_length = self.total_layer_num*[0]
        self.sliding_times = self.total_layer_num*[0]
        self.output_channel = self.total_layer_num*[0]
        self.weight_precision = self.total_layer_num*[8]
        self.activation_precision = self.total_layer_num*[8]
        self.operations = self.total_layer_num*[0]
        self.PE_num = self.total_layer_num*[0]
        self.tile_num = self.total_layer_num * [0]
        for i in range(self.total_layer_num):
            self.tile_list.append([])

        self.arch_area = self.total_layer_num*[0]
        self.arch_xbar_area = self.total_layer_num*[0]
        self.arch_ADC_area = self.total_layer_num*[0]
        self.arch_DAC_area = self.total_layer_num*[0]
        self.arch_digital_area = self.total_layer_num*[0]
        self.arch_adder_area = self.total_layer_num*[0]
        self.arch_shiftreg_area = self.total_layer_num*[0]
        self.arch_input_demux_area = self.total_layer_num*[0]
        self.arch_output_mux_area = self.total_layer_num*[0]
        self.arch_total_area = 0
        self.arch_total_xbar_area = 0
        self.arch_total_ADC_area = 0
        self.arch_total_DAC_area = 0
        self.arch_total_digital_area = 0
        self.arch_total_adder_area = 0
        self.arch_total_shiftreg_area = 0
        self.arch_total_input_demux_area = 0
        self.arch_total_output_mux_area = 0

        self.arch_utilization = self.total_layer_num*[0]
        self.arch_total_utilization = 0

        self.arch_latency = self.total_layer_num * [0]
        self.arch_xbar_latency = self.total_layer_num * [0]
        self.arch_ADC_latency = self.total_layer_num * [0]
        self.arch_DAC_latency = self.total_layer_num * [0]
        self.arch_digital_latency = self.total_layer_num * [0]
        self.arch_adder_latency = self.total_layer_num * [0]
        self.arch_shiftreg_latency = self.total_layer_num * [0]
        self.arch_input_demux_latency = self.total_layer_num * [0]
        self.arch_output_mux_latency = self.total_layer_num * [0]
        self.arch_total_latency = 0
        self.arch_total_xbar_latency = 0
        self.arch_total_ADC_latency = 0
        self.arch_total_DAC_latency = 0
        self.arch_total_digital_latency = 0
        self.arch_total_adder_latency = 0
        self.arch_total_shiftreg_latency = 0
        self.arch_total_input_demux_latency = 0
        self.arch_total_output_mux_latency = 0

        self.arch_power = self.total_layer_num * [0]
        self.arch_xbar_power = self.total_layer_num * [0]
        self.arch_ADC_power = self.total_layer_num * [0]
        self.arch_DAC_power = self.total_layer_num * [0]
        self.arch_digital_power = self.total_layer_num * [0]
        self.arch_adder_power = self.total_layer_num * [0]
        self.arch_shiftreg_power = self.total_layer_num * [0]
        self.arch_input_demux_power = self.total_layer_num * [0]
        self.arch_output_mux_power = self.total_layer_num * [0]
        self.arch_total_power = 0
        self.arch_total_xbar_power = 0
        self.arch_total_ADC_power = 0
        self.arch_total_DAC_power = 0
        self.arch_total_digital_power = 0
        self.arch_total_adder_power = 0
        self.arch_total_shiftreg_power = 0
        self.arch_total_input_demux_power = 0
        self.arch_total_output_mux_power = 0

        self.arch_energy = self.total_layer_num * [0]
        self.arch_xbar_energy = self.total_layer_num * [0]
        self.arch_ADC_energy = self.total_layer_num * [0]
        self.arch_DAC_energy = self.total_layer_num * [0]
        self.arch_digital_energy = self.total_layer_num * [0]
        self.arch_adder_energy = self.total_layer_num * [0]
        self.arch_shiftreg_energy = self.total_layer_num * [0]
        self.arch_input_demux_energy = self.total_layer_num * [0]
        self.arch_output_mux_energy = self.total_layer_num * [0]
        self.arch_total_energy = 0
        self.arch_total_xbar_energy = 0
        self.arch_total_ADC_energy = 0
        self.arch_total_DAC_energy = 0
        self.arch_total_digital_energy = 0
        self.arch_total_adder_energy = 0
        self.arch_total_shiftreg_energy = 0
        self.arch_total_input_demux_energy = 0
        self.arch_total_output_mux_energy = 0

        self.arch_energy_efficiency = 0


    def config_behavior_mapping(self):
        for layer_id in range(len(self.net_structure)):
            layer_dict = self.net_structure[layer_id][0][0]
            layer_type = layer_dict['type']
            if layer_type =='conv' or layer_type=='pooling':
                inputsize = list(map(int, layer_dict['Inputsize']))
                inputsize = inputsize[0]*inputsize[1]
                outputsize = list(map(int, layer_dict['Outputsize']))
                outputsize = outputsize[0]*outputsize[1]
                kernelsize = int(layer_dict['Kernelsize'])
                stride = int(layer_dict['Stride'])
                inputchannel = int(layer_dict['Inputchannel'])
                self.output_channel[layer_id] = int(layer_dict['Outputchannel'])
                self.sliding_times[layer_id] = outputsize
            else:
                assert layer_type == 'fc', "Layer type must be conv or fc"
                inputsize = 1
                outputsize = int(layer_dict['Outfeature'])
                kernelsize = 1
                inputchannel = int(layer_dict['Infeature'])
                self.output_channel[layer_id] = int(layer_dict['Outfeature'])
                self.sliding_times[layer_id] = 1
            self.kernel_length[layer_id] = kernelsize ** 2 * inputchannel
                # The length of the kernel data
            if self.xbar_polarity == 1:
                self.weight_precision[layer_id] = int(layer_dict['Weightbit'])
            else:
                assert self.xbar_polarity == 2, "Crossbar polarity must be 1 or 2"
                self.weight_precision[layer_id] = int(layer_dict['Weightbit']) - 1
            self.activation_precision[layer_id] = int(layer_dict['Inputbit'])

            inputchannel_xbar = min(self.xbar_row // (kernelsize**2), inputchannel)
                # The input channel number stored in one crossbar
            if self.output_channel[layer_id]>self.xbar_column:
                outputchannel_xbar = self.xbar_column
            else:
                outputchannel_xbar = self.output_channel[layer_id]
                    # The output channel number stored in one crossbar
            PE_bitwidth = math.floor(math.log2(self.device_level))*self.group_num
                # The bitwidth of one PE can represent

            self.operations[layer_id] = self.sliding_times[layer_id] * \
                                        self.kernel_length[layer_id] * \
                                        self.output_channel[layer_id] * 2
                # The number of operations

            self.PE_num[layer_id] = math.ceil(inputchannel/inputchannel_xbar) * \
                                    math.ceil(self.output_channel[layer_id]/outputchannel_xbar) * \
                                    math.ceil(self.weight_precision[layer_id]/PE_bitwidth)
                # The PE number used for this layer
            self.tile_num[layer_id] = math.ceil(self.PE_num[layer_id] / (self.tile_PE_total_num))
                # The tile number used for this layer
            xbar_used_length = inputchannel_xbar * (kernelsize**2)
                # The number of used row (length of the used cells in one column)
            # Mapping procedure start:
            tile_index = 0
            current_PE_num = 0
            read_column = []
            read_row = []
            kernel_length_2bsplit = self.kernel_length[layer_id]
            while kernel_length_2bsplit > 0:
                if kernel_length_2bsplit < xbar_used_length:
                    temp_length = kernel_length_2bsplit
                else:
                    temp_length = xbar_used_length
                kernel_length_2bsplit -= temp_length
                channel_width_2bsplit = self.output_channel[layer_id]

                while channel_width_2bsplit > 0:
                    if channel_width_2bsplit < outputchannel_xbar:
                        temp_width = channel_width_2bsplit
                    else:
                        temp_width = outputchannel_xbar
                    channel_width_2bsplit -= temp_width
                    weight_precision_2bsplit = self.weight_precision[layer_id]

                    while weight_precision_2bsplit > 0:
                        if weight_precision_2bsplit < PE_bitwidth:
                            temp_bitwidth = weight_precision_2bsplit
                        else:
                            temp_bitwidth = PE_bitwidth
                        weight_precision_2bsplit -= temp_bitwidth
                        current_PE_num += 1
                        temp_occupied_group = math.ceil(temp_bitwidth/math.floor(math.log2(self.device_level)))
                        read_row.append(temp_occupied_group*[temp_length])
                        read_column.append(temp_occupied_group*[temp_width])
                        if current_PE_num == self.tile_PE_total_num or \
                                ((kernel_length_2bsplit==0)&(channel_width_2bsplit==0)&(weight_precision_2bsplit==0)):
                            # print("yes")
                            __temp_tile = tile(self.SimConfig_path)
                            self.tile_list[layer_id].append(__temp_tile)
                            self.tile_list[layer_id][tile_index].tile_read_config(
                                layer_num=layer_id,
                                activation_precision=self.activation_precision[layer_id],
                                sliding_times=self.sliding_times[layer_id],
                                read_row=read_row,
                                read_column=read_column
                            )
                            tile_index += 1
                            current_PE_num = 0
                            read_column = []
                            read_row = []
            # print(layer_id,':',self.PE_num[layer_id],self.tile_num[layer_id],tile_index)

    def behavior_mapping_area(self):
        # Notice: before calculating area, config_behavior_mapping must be executed
        # unit: um^2
        self.calculate_tile_area()
        for i in range(self.total_layer_num):
            self.arch_area[i] = self.tile_area * self.tile_num[i]
            self.arch_xbar_area[i] = self.tile_xbar_area * self.tile_num[i]
            self.arch_ADC_area[i] = self.tile_ADC_area * self.tile_num[i]
            self.arch_DAC_area[i] = self.tile_DAC_area * self.tile_num[i]
            self.arch_digital_area[i] = self.tile_digital_area * self.tile_num[i]
            self.arch_adder_area[i] = self.tile_adder_area * self.tile_num[i]
            self.arch_shiftreg_area[i] = self.tile_shiftreg_area * self.tile_num[i]
            self.arch_input_demux_area[i] = self.tile_input_demux_area * self.tile_num[i]
            self.arch_output_mux_area[i] = self.tile_output_mux_area * self.tile_num[i]
        self.arch_total_area = sum(self.arch_area)
        self.arch_total_xbar_area = sum(self.arch_xbar_area)
        self.arch_total_ADC_area = sum(self.arch_ADC_area)
        self.arch_total_DAC_area = sum(self.arch_DAC_area)
        self.arch_total_digital_area = sum(self.arch_digital_area)
        self.arch_total_adder_area = sum(self.arch_adder_area)
        self.arch_total_shiftreg_area = sum(self.arch_shiftreg_area)
        self.arch_total_input_demux_area = sum(self.arch_input_demux_area)
        self.arch_total_output_mux_area = sum(self.arch_output_mux_area)

    def behavior_mapping_utilization(self):
        # Notice: before calculating utilization, config_behavior_mapping must be executed
        for i in range(self.total_layer_num):
            for j in range(self.tile_num[i]):
                self.arch_utilization[i] += self.tile_list[i][j].tile_utilization
                self.arch_total_utilization += self.tile_list[i][j].tile_utilization
            self.arch_utilization[i] /= self.tile_num[i]
        self.arch_total_utilization /= sum(self.tile_num)

    def behavior_mapping_latency(self):
        # Notice: before calculating latency, config_behavior_mapping must be executed
        # unit: ns
        # TODO: add arch level adder tree estimation
        for i in range(self.total_layer_num):
            temp_latency = 0
            latency_index = 0
            for j in range(self.tile_num[i]):
                self.tile_list[i][j].calculate_tile_read_latency()
                if self.tile_list[i][j].tile_read_latency > temp_latency:
                    latency_index = j
            self.arch_latency[i] = self.tile_list[i][latency_index].tile_read_latency
            self.arch_xbar_latency[i] = self.tile_list[i][latency_index].tile_xbar_read_latency
            self.arch_ADC_latency[i] = self.tile_list[i][latency_index].tile_ADC_read_latency
            self.arch_DAC_latency[i] = self.tile_list[i][latency_index].tile_DAC_read_latency
            self.arch_digital_latency[i] = self.tile_list[i][latency_index].tile_digital_read_latency
            self.arch_adder_latency[i] = self.tile_list[i][latency_index].tile_adder_read_latency
            self.arch_shiftreg_latency[i] = self.tile_list[i][latency_index].tile_shiftreg_read_latency
            self.arch_input_demux_latency[i] = self.tile_list[i][latency_index].tile_input_demux_read_latency
            self.arch_output_mux_latency[i] = self.tile_list[i][latency_index].tile_output_mux_read_latency
        self.arch_total_latency = sum(self.arch_latency)
        self.arch_total_xbar_latency = sum(self.arch_xbar_latency)
        self.arch_total_ADC_latency = sum(self.arch_ADC_latency)
        self.arch_total_DAC_latency = sum(self.arch_DAC_latency)
        self.arch_total_digital_latency = sum(self.arch_digital_latency)
        self.arch_total_adder_latency = sum(self.arch_adder_latency)
        self.arch_total_shiftreg_latency = sum(self.arch_shiftreg_latency)
        self.arch_total_input_demux_latency = sum(self.arch_input_demux_latency)
        self.arch_total_output_mux_latency = sum(self.arch_output_mux_latency)

    def behavior_mapping_power(self):
        # Notice: before calculating power, config_behavior_mapping must be executed
        # unit: W
        # TODO: add arch level adder tree estimation
        for i in range(self.total_layer_num):
            for j in range(self.tile_num[i]):
                self.tile_list[i][j].calculate_tile_read_power()
                self.arch_power[i] += self.tile_list[i][j].tile_read_power
                self.arch_xbar_power[i] += self.tile_list[i][j].tile_xbar_read_power
                self.arch_ADC_power[i] += self.tile_list[i][j].tile_ADC_read_power
                self.arch_DAC_power[i] += self.tile_list[i][j].tile_DAC_read_power
                self.arch_digital_power[i] += self.tile_list[i][j].tile_digital_read_power
                self.arch_adder_power[i] += self.tile_list[i][j].tile_adder_read_power
                self.arch_shiftreg_power[i] += self.tile_list[i][j].tile_shiftreg_read_power
                self.arch_input_demux_power[i] += self.tile_list[i][j].tile_input_demux_read_power
                self.arch_output_mux_power[i] += self.tile_list[i][j].tile_output_mux_read_power
        self.arch_total_power = sum(self.arch_power)
        self.arch_total_xbar_power = sum(self.arch_xbar_power)
        self.arch_total_ADC_power = sum(self.arch_ADC_power)
        self.arch_total_DAC_power = sum(self.arch_DAC_power)
        self.arch_total_digital_power = sum(self.arch_digital_power)
        self.arch_total_adder_power = sum(self.arch_adder_power)
        self.arch_total_shiftreg_power = sum(self.arch_shiftreg_power)
        self.arch_total_input_demux_power = sum(self.arch_input_demux_power)
        self.arch_total_output_mux_power = sum(self.arch_output_mux_power)

    def behavior_mapping_energy(self):
        # Notice: before calculating energy, config_behavior_mapping must be executed
        # unit: nJ
        # TODO: add arch level adder tree estimation
        for i in range(self.total_layer_num):
            for j in range(self.tile_num[i]):
                self.tile_list[i][j].calculate_tile_read_energy()
                self.arch_energy[i] += self.tile_list[i][j].tile_read_energy
                self.arch_xbar_energy[i] += self.tile_list[i][j].tile_xbar_read_energy
                self.arch_ADC_energy[i] += self.tile_list[i][j].tile_ADC_read_energy
                self.arch_DAC_energy[i] += self.tile_list[i][j].tile_DAC_read_energy
                self.arch_digital_energy[i] += self.tile_list[i][j].tile_digital_read_energy
                self.arch_adder_energy[i] += self.tile_list[i][j].tile_adder_read_energy
                self.arch_shiftreg_energy[i] += self.tile_list[i][j].tile_shiftreg_read_energy
                self.arch_input_demux_energy[i] += self.tile_list[i][j].tile_input_demux_read_energy
                self.arch_output_mux_energy[i] += self.tile_list[i][j].tile_output_mux_read_energy
        self.arch_total_energy = sum(self.arch_energy)
        self.arch_total_xbar_energy = sum(self.arch_xbar_energy)
        self.arch_total_ADC_energy = sum(self.arch_ADC_energy)
        self.arch_total_DAC_energy = sum(self.arch_DAC_energy)
        self.arch_total_digital_energy = sum(self.arch_digital_energy)
        self.arch_total_adder_energy = sum(self.arch_adder_energy)
        self.arch_total_shiftreg_energy = sum(self.arch_shiftreg_energy)
        self.arch_total_input_demux_energy = sum(self.arch_input_demux_energy)
        self.arch_total_output_mux_energy = sum(self.arch_output_mux_energy)
        self.arch_energy_efficiency = sum(self.operations)/self.arch_total_energy #unit: GOPs/W

    def behavior_mapping_output(self, module_information = 1, layer_information = 1):
        # module_information: 1: print module information
        # layer_information: 1: print hardware performance of each layer
        print("=====================================")
        print("CNN model information:")
        print("Layer number:", self.total_layer_num)
        for i in range(len(self.net_structure)):
            layer = self.net_structure[i][0][0]
            print("     Layer", i, ":", layer['type'])
            if layer['type'] == 'conv':
                print("     |----Input Size:", layer['Inputsize'])
                print("     |----Input Precision:", layer['Inputbit'])
                print("     |----Kernel Size:", layer['Kernelsize'])
                print("     |----Weight Precision:", layer['Weightbit'])
                print("     |----Input Channel:", layer['Inputchannel'])
                print("     |----Stride:", layer['Stride'])
                print("     |----Output Size:", layer['Outputsize'])
                print("     |----Output Channel:", layer['Outputchannel'])
                # print("     |----Operations:", self.operations[i])
            elif layer['type'] == 'fc':
                print("     |----Input Size:", layer['Infeature'])
                print("     |----Input Precision:", layer['Inputbit'])
                print("     |----Weight Precision:", layer['Weightbit'])
                print("     |----Output Size:", layer['Outfeature'])
                # print("     |----Operations:", self.operations[i])
            elif layer['type'] == 'pooling':
                print("     |----Input Size:", layer['Inputsize'])
                print("     |----Input Precision:", layer['Inputbit'])
                print("     |----Kernel Size:", layer['Kernelsize'])
                print("     |----Weight Precision:", layer['Weightbit'])
                print("     |----Input Channel:", layer['Inputchannel'])
                print("     |----Stride:", layer['Stride'])
                print("     |----Output Size:", layer['Outputsize'])
                print("     |----Output Channel:", layer['Outputchannel'])
                # print("     |----Operations:", self.operations[i])
        print("======================================")
        print("Hardware performance finished!")
        print("Tile number:", sum(self.tile_num))
        print("Resource utilization:", self.arch_total_utilization)
        print("Hardware area:", self.arch_total_area, "um^2")
        if module_information:
            print("		crossbar area:", self.arch_total_xbar_area, "um^2")
            print("		DAC area:", self.arch_total_DAC_area, "um^2")
            print("		ADC area:", self.arch_total_ADC_area, "um^2")
            print("		digital part area:", self.arch_total_digital_area, "um^2")
            print("			|---adder area:", self.arch_total_adder_area, "um^2")
            print("			|---shift-reg area:", self.arch_total_shiftreg_area, "um^2")
            print("			|---input_demux area:", self.arch_total_input_demux_area, "um^2")
            print("			|---output_mux area:", self.arch_total_output_mux_area, "um^2")
        print("Hardware power:", self.arch_total_power, "W")
        if module_information:
            print("		crossbar power:", self.arch_total_xbar_power, "W")
            print("		DAC power:", self.arch_total_DAC_power, "W")
            print("		ADC power:", self.arch_total_ADC_power, "W")
            print("		digital part power:", self.arch_total_digital_power, "W")
            print("			|---adder power:", self.arch_total_adder_power, "W")
            print("			|---shift-reg power:", self.arch_total_shiftreg_power, "W")
            print("			|---input_demux power:", self.arch_total_input_demux_power, "W")
            print("			|---output_mux power:", self.arch_total_output_mux_power, "W")
        print("Hardware energy:", self.arch_total_energy, "nJ")
        if module_information:
            print("		crossbar energy:", self.arch_total_xbar_energy, "nJ")
            print("		DAC energy:", self.arch_total_DAC_energy, "nJ")
            print("		ADC energy:", self.arch_total_ADC_energy, "nJ")
            print("		digital part energy:", self.arch_total_digital_energy, "nJ")
            print("			|---adder energy:", self.arch_total_adder_energy, "nJ")
            print("			|---shift-reg energy:", self.arch_total_shiftreg_energy, "nJ")
            print("			|---input_demux energy:", self.arch_total_input_demux_energy, "nJ")
            print("			|---output_mux energy:", self.arch_total_output_mux_energy, "nJ")
        if layer_information:
            for i in range(self.total_layer_num):
                print("Layer", i, ":")
                print("     Operations:", self.operations[i])
                print("     Tile number:", self.tile_num[i])
                print("     Tile utilization:", self.arch_utilization[i])
                print("     Hardware area:", self.arch_area[i])
                print("     Hardware power:", self.arch_power[i])
                print("     Hardware latency:", self.arch_latency[i])
                print("     Hardware energy:", self.arch_energy[i])
                print("         Crossbar energy:", self.arch_xbar_energy[i])
                print("         ADC energy:", self.arch_ADC_energy[i])
        print("total operations:",sum(self.operations))
        print("Hardware energy efficiency:", self.arch_energy_efficiency, " GOPs/W")


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "alexnet_channels_bit/alexnet_16_5_params.pth")

    __TestInterface = TrainTestInterface('alexnet_16_5', 'MNSIM.Interface.cifar10', test_SimConfig_path, test_weights_file_path, 'cpu')
    structure_file = __TestInterface.get_structure()

    _bm = behavior_mapping(structure_file, test_SimConfig_path)
    _bm.config_behavior_mapping()
    _bm.behavior_mapping_area()
    _bm.behavior_mapping_utilization()
    _bm.behavior_mapping_latency()
    _bm.behavior_mapping_power()
    _bm.behavior_mapping_energy()
    _bm.behavior_mapping_output(1,1)
