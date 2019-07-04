#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
work_path = os.path.dirname(os.getcwd())
print("ok", work_path)
sys.path.append(work_path)
from MNSIM.Hardware_Model import *
from MNSIM.Hardware_Model.Bank import bank


class behavior_mapping(bank):
    def __init__(self, NetStruct_path, SimConfig_path):
        self.SimConfig_path = SimConfig_path
        bank.__init__(self, SimConfig_path)
        print("CNN structure file is loaded:", NetStruct_path)
        print("Hardware config file is loaded:", SimConfig_path)
        self.net_structure = torch.load(NetStruct_path)
        self.arch_config = SimConfig_path
        self.total_layer_num = len(self.net_structure)
        self.bank_list = []
        self.kernel_length = self.total_layer_num*[0]
        self.sliding_times = self.total_layer_num*[0]
        self.output_channel = self.total_layer_num*[0]
        self.weight_precision = self.total_layer_num*[8]
        self.activation_precision = self.total_layer_num*[8]
        self.PE_num = self.total_layer_num*[0]
        self.bank_num = self.total_layer_num*[0]
        for i in range(self.total_layer_num):
            self.bank_list.append([])

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


    def config_behavior_mapping(self):
        i = 0
        for layer in self.net_structure.items():
            # print("----------layer", i, "-----------")
            inputsize = int(layer[1]['Inputsize'])
            outputsize = int(layer[1]['Outputsize'])
            kernelsize = int(layer[1]['Kernelsize'])
            stride = int(layer[1]['Stride'])
            inputchannel = int(layer[1]['Inputchannel'])
            self.output_channel[i] = int(layer[1]['Outputchannel'])
            self.weight_precision[i] = int(layer[1]['Weightbit'])
            self.activation_precision[i] = int(layer[1]['Inputbit'])
            self.kernel_length[i] = kernelsize**2 * inputchannel
            self.sliding_times[i] = outputsize**2
            self.PE_num[i] = math.ceil(self.output_channel[i]/self.xbar_column) * \
                             math.ceil(self.weight_precision[i]/self.group_num) * \
                             math.ceil(self.kernel_length[i]/self.xbar_row)

            # print(self.PE_num[i])
            self.bank_num[i] = math.ceil(self.PE_num[i]/self.bank_PE_total_num)
            # print("bank_num", self.bank_num[i])
            temp_weightprecision = self.weight_precision[i]
            temp_outputchannel = self.output_channel[i]
            temp_kernellength = self.kernel_length[i]
            bank_index = 0
            read_column = []
            read_row = []
            while temp_weightprecision > 0:
                if temp_weightprecision <= self.group_num:
                    num_occupied_group = temp_weightprecision
                else:
                    num_occupied_group = self.group_num
                temp_weightprecision -= num_occupied_group
                temp_outputchannel = self.output_channel[i]
                while temp_outputchannel > 0:
                    if temp_outputchannel <= self.xbar_column * self.bank_PE_num[1]:
                        temp_read_column = temp_outputchannel
                    else:
                        temp_read_column = self.xbar_column * self.bank_PE_num[1]
                    temp_outputchannel -= temp_read_column
                    temp_kernellength = self.kernel_length[i]
                    while temp_kernellength > 0:
                        if temp_kernellength <= self.xbar_row * self.bank_PE_num[0]:
                            temp_read_row = temp_kernellength
                        else:
                            temp_read_row = self.xbar_row * self.bank_PE_num[0]
                        temp_kernellength -= temp_read_row
                        temp_temp_read_column = temp_read_column
                        while temp_temp_read_column > 0:
                            temp_temp_read_row = temp_read_row
                            if temp_temp_read_column <= self.xbar_column:
                                while temp_temp_read_row > 0:
                                    if temp_temp_read_row <= self.xbar_row:
                                        if len(read_column) == self.bank_PE_total_num:
                                            __temp_bank = bank(self.SimConfig_path)
                                            self.bank_list[i].append(__temp_bank)
                                            self.bank_list[i][bank_index].bank_read_config(layer_num=i,
                                                                                       activation_precision=
                                                                                       self.activation_precision[i],
                                                                                       sliding_times=self.sliding_times[i],
                                                                                       read_row=read_row,
                                                                                       read_column=read_column)
                                            # print("read_row", read_row)
                                            # print("read_column", read_column)
                                            bank_index += 1
                                            read_column = []
                                            read_row = []
                                        read_row.append(num_occupied_group * [temp_temp_read_row])
                                    else:
                                        if len(read_column) == self.bank_PE_total_num:
                                            __temp_bank = bank(self.SimConfig_path)
                                            self.bank_list[i].append(__temp_bank)
                                            self.bank_list[i][bank_index].bank_read_config(layer_num=i,
                                                                                       activation_precision=
                                                                                       self.activation_precision[i],
                                                                                       sliding_times=self.sliding_times[i],
                                                                                       read_row=read_row,
                                                                                       read_column=read_column)
                                            # print("read_row", read_row)
                                            # print("read_column", read_column)
                                            bank_index += 1
                                            read_column = []
                                            read_row = []
                                        read_row.append(num_occupied_group * [self.xbar_row])
                                    read_column.append(num_occupied_group * [temp_temp_read_column])
                                    temp_temp_read_row -= self.xbar_row
                            else:
                                while temp_temp_read_row > 0:
                                    if temp_temp_read_row <= self.xbar_row:
                                        if len(read_column) == self.bank_PE_total_num:
                                            __temp_bank = bank(self.SimConfig_path)
                                            self.bank_list[i].append(__temp_bank)
                                            self.bank_list[i][bank_index].bank_read_config(layer_num=i,
                                                                                           activation_precision=
                                                                                           self.activation_precision[i],
                                                                                           sliding_times=
                                                                                           self.sliding_times[i],
                                                                                           read_row=read_row,
                                                                                           read_column=read_column)
                                            # print("read_row", read_row)
                                            # print("read_column", read_column)
                                            bank_index += 1
                                            read_column = []
                                            read_row = []
                                        read_row.append(num_occupied_group * [temp_temp_read_row])
                                    else:
                                        if len(read_column) == self.bank_PE_total_num:
                                            __temp_bank = bank(self.SimConfig_path)
                                            self.bank_list[i].append(__temp_bank)
                                            self.bank_list[i][bank_index].bank_read_config(layer_num=i,
                                                                                           activation_precision=
                                                                                           self.activation_precision[i],
                                                                                           sliding_times=self.sliding_times[i],
                                                                                           read_row=read_row,
                                                                                           read_column=read_column)
                                            # print("read_row", read_row)
                                            # print("read_column", read_column)
                                            bank_index += 1
                                            read_column = []
                                            read_row = []
                                        read_row.append(num_occupied_group * [self.xbar_row])
                                    read_column.append(num_occupied_group * [self.xbar_column])
                                    temp_temp_read_row -= self.xbar_row
                            temp_temp_read_column -= self.xbar_column

                        if (temp_weightprecision <= 0) & (temp_outputchannel <= 0) & (temp_kernellength <= 0):
                            __temp_bank = bank(self.SimConfig_path)
                            self.bank_list[i].append(__temp_bank)
                            self.bank_list[i][bank_index].bank_read_config(layer_num=i,
                                                                           activation_precision=
                                                                           self.activation_precision[i],
                                                                           sliding_times=self.sliding_times[i],
                                                                           read_row=read_row,
                                                                           read_column=read_column)
                            # print("read_row", read_row)
                            # print("read_column", read_column)
                            bank_index += 1
            # print("bank_index", bank_index)
            i += 1

    def behavior_mapping_area(self):
        # Notice: before calculating area, config_behavior_mapping must be executed
        # unit: um^2
        self.calculate_bank_area()
        for i in range(self.total_layer_num):
            self.arch_area[i] = self.bank_area * self.bank_num[i]
            self.arch_xbar_area[i] = self.bank_xbar_area * self.bank_num[i]
            self.arch_ADC_area[i] = self.bank_ADC_area * self.bank_num[i]
            self.arch_DAC_area[i] = self.bank_DAC_area * self.bank_num[i]
            self.arch_digital_area[i] = self.bank_digital_area * self.bank_num[i]
            self.arch_adder_area[i] = self.bank_adder_area * self.bank_num[i]
            self.arch_shiftreg_area[i] = self.bank_shiftreg_area * self.bank_num[i]
            self.arch_input_demux_area[i] = self.bank_input_demux_area * self.bank_num[i]
            self.arch_output_mux_area[i] = self.bank_output_mux_area * self.bank_num[i]
            # print(self.bank_num[i], self.arch_area[i])
        self.arch_total_area = sum(self.arch_area)
        self.arch_total_xbar_area = sum(self.arch_xbar_area)
        self.arch_total_ADC_area = sum(self.arch_ADC_area)
        self.arch_total_DAC_area = sum(self.arch_DAC_area)
        self.arch_total_digital_area = sum(self.arch_digital_area)
        self.arch_total_adder_area = sum(self.arch_adder_area)
        self.arch_total_shiftreg_area = sum(self.arch_shiftreg_area)
        self.arch_total_input_demux_area = sum(self.arch_input_demux_area)
        self.arch_total_output_mux_area = sum(self.arch_output_mux_area)
        # print(self.arch_total_area)

    def behavior_mapping_utilization(self):
        # Notice: before calculating utilization, config_behavior_mapping must be executed
        for i in range(self.total_layer_num):
            for j in range(self.bank_num[i]):
                self.arch_utilization[i] += self.bank_list[i][j].bank_utilization
                self.arch_total_utilization += self.bank_list[i][j].bank_utilization
            self.arch_utilization[i] /= self.bank_num[i]
            # print(self.arch_utilization[i])
        self.arch_total_utilization /= sum(self.bank_num)
        # print(self.arch_total_utilization)

    def behavior_mapping_latency(self):
        # Notice: before calculating latency, config_behavior_mapping must be executed
        # unit: ns
        # TODO: add arch level adder tree estimation
        for i in range(self.total_layer_num):
            temp_latency = 0
            latency_index = 0
            for j in range(self.bank_num[i]):
                self.bank_list[i][j].calculate_bank_read_latency()
                if self.bank_list[i][j].bank_read_latency > temp_latency:
                    latency_index = j
            self.arch_latency[i] = self.bank_list[i][latency_index].bank_read_latency
            self.arch_xbar_latency[i] = self.bank_list[i][latency_index].bank_xbar_read_latency
            self.arch_ADC_latency[i] = self.bank_list[i][latency_index].bank_ADC_read_latency
            self.arch_DAC_latency[i] = self.bank_list[i][latency_index].bank_DAC_read_latency
            self.arch_digital_latency[i] = self.bank_list[i][latency_index].bank_digital_read_latency
            self.arch_adder_latency[i] = self.bank_list[i][latency_index].bank_adder_read_latency
            self.arch_shiftreg_latency[i] = self.bank_list[i][latency_index].bank_shiftreg_read_latency
            self.arch_input_demux_latency[i] = self.bank_list[i][latency_index].bank_input_demux_read_latency
            self.arch_output_mux_latency[i] = self.bank_list[i][latency_index].bank_output_mux_read_latency
            # print(self.bank_num[i], self.arch_latency[i])
        self.arch_total_latency = sum(self.arch_latency)
        self.arch_total_xbar_latency = sum(self.arch_xbar_latency)
        self.arch_total_ADC_latency = sum(self.arch_ADC_latency)
        self.arch_total_DAC_latency = sum(self.arch_DAC_latency)
        self.arch_total_digital_latency = sum(self.arch_digital_latency)
        self.arch_total_adder_latency = sum(self.arch_adder_latency)
        self.arch_total_shiftreg_latency = sum(self.arch_shiftreg_latency)
        self.arch_total_input_demux_latency = sum(self.arch_input_demux_latency)
        self.arch_total_output_mux_latency = sum(self.arch_output_mux_latency)
        # print(self.arch_total_latency)

    def behavior_mapping_power(self):
        # Notice: before calculating power, config_behavior_mapping must be executed
        # unit: W
        # TODO: add arch level adder tree estimation
        for i in range(self.total_layer_num):
            for j in range(self.bank_num[i]):
                self.bank_list[i][j].calculate_bank_read_power()
                self.arch_power[i] += self.bank_list[i][j].bank_read_power
                self.arch_xbar_power[i] += self.bank_list[i][j].bank_xbar_read_power
                self.arch_ADC_power[i] += self.bank_list[i][j].bank_ADC_read_power
                self.arch_DAC_power[i] += self.bank_list[i][j].bank_DAC_read_power
                self.arch_digital_power[i] += self.bank_list[i][j].bank_digital_read_power
                self.arch_adder_power[i] += self.bank_list[i][j].bank_adder_read_power
                self.arch_shiftreg_power[i] += self.bank_list[i][j].bank_shiftreg_read_power
                self.arch_input_demux_power[i] += self.bank_list[i][j].bank_input_demux_read_power
                self.arch_output_mux_power[i] += self.bank_list[i][j].bank_output_mux_read_power
            # print(self.bank_num[i], self.arch_power[i])
        self.arch_total_power = sum(self.arch_power)
        self.arch_total_xbar_power = sum(self.arch_xbar_power)
        self.arch_total_ADC_power = sum(self.arch_ADC_power)
        self.arch_total_DAC_power = sum(self.arch_DAC_power)
        self.arch_total_digital_power = sum(self.arch_digital_power)
        self.arch_total_adder_power = sum(self.arch_adder_power)
        self.arch_total_shiftreg_power = sum(self.arch_shiftreg_power)
        self.arch_total_input_demux_power = sum(self.arch_input_demux_power)
        self.arch_total_output_mux_power = sum(self.arch_output_mux_power)
        # print(self.arch_total_power)
        # print("xbar", self.arch_total_xbar_power)
        # print("ADC", self.arch_total_ADC_power)

    def behavior_mapping_energy(self):
        # Notice: before calculating energy, config_behavior_mapping must be executed
        # unit: nJ
        # TODO: add arch level adder tree estimation
        for i in range(self.total_layer_num):
            for j in range(self.bank_num[i]):
                self.bank_list[i][j].calculate_bank_read_energy()
                self.arch_energy[i] += self.bank_list[i][j].bank_read_energy
                self.arch_xbar_energy[i] += self.bank_list[i][j].bank_xbar_read_energy
                self.arch_ADC_energy[i] += self.bank_list[i][j].bank_ADC_read_energy
                self.arch_DAC_energy[i] += self.bank_list[i][j].bank_DAC_read_energy
                self.arch_digital_energy[i] += self.bank_list[i][j].bank_digital_read_energy
                self.arch_adder_energy[i] += self.bank_list[i][j].bank_adder_read_energy
                self.arch_shiftreg_energy[i] += self.bank_list[i][j].bank_shiftreg_read_energy
                self.arch_input_demux_energy[i] += self.bank_list[i][j].bank_input_demux_read_energy
                self.arch_output_mux_energy[i] += self.bank_list[i][j].bank_output_mux_read_energy
            # print(self.bank_num[i], self.arch_energy[i])
        self.arch_total_energy = sum(self.arch_energy)
        self.arch_total_xbar_energy = sum(self.arch_xbar_energy)
        self.arch_total_ADC_energy = sum(self.arch_ADC_energy)
        self.arch_total_DAC_energy = sum(self.arch_DAC_energy)
        self.arch_total_digital_energy = sum(self.arch_digital_energy)
        self.arch_total_adder_energy = sum(self.arch_adder_energy)
        self.arch_total_shiftreg_energy = sum(self.arch_shiftreg_energy)
        self.arch_total_input_demux_energy = sum(self.arch_input_demux_energy)
        self.arch_total_output_mux_energy = sum(self.arch_output_mux_energy)
        # print(self.arch_total_energy)
        # print("xbar", self.arch_total_xbar_energy)
        # print("ADC", self.arch_total_ADC_energy)

    def behavior_mapping_output(self, module_information = 0, layer_information = 0):
        # module_information: 1: print module information
        # layer_information: 1: print hardware performance of each layer
        print("--------------CNN model--------------")
        print("Layer number:", self.total_layer_num)
        i = 1
        for layer in self.net_structure.items():
            print("     Layer", i, ":")
            print("     |----Input Size:", int(layer[1]['Inputsize']))
            print("     |----Input Precision:", int(layer[1]['Inputbit']))
            print("     |----Kernel Size:", int(layer[1]['Kernelsize']))
            print("     |----Weight Precision:", int(layer[1]['Weightbit']))
            print("     |----Input Channel:", int(layer[1]['Inputchannel']))
            print("     |----Stride:", int(layer[1]['Stride']))
            print("     |----Output Size:", int(layer[1]['Outputsize']))
            print("     |----Output Channel:", int(layer[1]['Outputchannel']))
            i += 1
        print("---------Hardware Performance---------")
        print("Bank number:", sum(self.bank_num))
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
        print("Hardware latency:", self.arch_total_latency, "ns")
        if module_information:
            print("		crossbar latency:", self.arch_total_xbar_latency, "ns")
            print("		DAC latency:", self.arch_total_DAC_latency, "ns")
            print("		ADC latency:", self.arch_total_ADC_latency, "ns")
            print("		digital part latency:", self.arch_total_digital_latency, "ns")
            print("			|---adder latency:", self.arch_total_adder_latency, "ns")
            print("			|---shift-reg latency:", self.arch_total_shiftreg_latency, "ns")
            print("			|---input_demux latency:", self.arch_total_input_demux_latency, "ns")
            print("			|---output_mux latency:", self.arch_total_output_mux_latency, "ns")
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
                print("     Bank number:", self.bank_num[i])
                print("     Bank utilization:", self.arch_utilization[i])
                print("     Hardware area:", self.arch_area[i])
                print("     Hardware power:", self.arch_power[i])
                print("     Hardware latency:", self.arch_latency[i])
                print("     Hardware energy:", self.arch_energy[i])


if __name__ == '__main__':
    # print("ok")
    net_structure_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "mnist_net.pt")
    # net_structure_path = "/Users/zzh/Desktop/lab/MNSIM_python_v1.2/mnist_net.pt"
    SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    # SimConfig_path = "/Users/zzh/Desktop/lab/MNSIM_python_v1.2/SimConfig.ini"
    # _bank = bank(SimConfig_path)
    # print(net_structure_path)
    # print(SimConfig_path)
    _bm = behavior_mapping(net_structure_path, SimConfig_path)
    _bm.config_behavior_mapping()
    _bm.behavior_mapping_area()
    _bm.behavior_mapping_utilization()
    _bm.behavior_mapping_latency()
    _bm.behavior_mapping_power()
    _bm.behavior_mapping_energy()
    _bm.behavior_mapping_output(1,1)
