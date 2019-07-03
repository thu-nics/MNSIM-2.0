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

    def set_behavior_mapping(self):
        i = 0
        for layer in self.net_structure.items():
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
            print("bank_num",self.bank_num[i])
            temp_weightprecision = self.weight_precision[i]
            temp_outputchannel = self.output_channel[i]
            temp_kernellength = self.kernel_length[i]
            index = 0
            while temp_weightprecision > 0:
                # print("temp_weightprecision *", temp_weightprecision)
                # print("group_num",self.group_num)
                if temp_weightprecision <= self.group_num:
                    num_occupied_group = temp_weightprecision
                else:
                    num_occupied_group = self.group_num
                temp_weightprecision -= num_occupied_group
                # print("temp_weightprecision", temp_weightprecision)
                # print("temp_outputchannel *", temp_outputchannel)
                temp_outputchannel = self.output_channel[i]
                while temp_outputchannel > 0:
                    if temp_outputchannel <= self.xbar_column * self.bank_PE_num[1]:
                        temp_read_column = temp_outputchannel
                    else:
                        temp_read_column = self.xbar_column * self.bank_PE_num[1]
                    temp_outputchannel -= temp_read_column
                    # print("temp_outputchannel", temp_outputchannel)
                    # print("temp_kernellength*", temp_kernellength)
                    temp_kernellength = self.kernel_length[i]
                    # read_row = []
                    # read_column = []
                    # __temp_bank = bank(self.SimConfig_path)
                    while temp_kernellength > 0:
                        if temp_kernellength <= self.xbar_row * self.bank_PE_total_num:
                        # if temp_kernellength <= self.xbar_row * self.bank_PE_num[0]:
                            temp_read_row = temp_kernellength
                        else:
                            # temp_read_row = self.xbar_row * self.bank_PE_num[0]
                            temp_read_row = self.xbar_row * self.bank_PE_total_num
                        temp_kernellength -= temp_read_row
                        # print("temp_kernellength", temp_kernellength)
                        read_row = []
                        read_column = []
                        __temp_bank = bank(self.SimConfig_path)
                        self.bank_list[i].append(__temp_bank)
                        temp_temp_read_column = temp_read_column
                        # print("temp_read_colmn*", temp_read_column)
                        while temp_temp_read_column > 0:
                            if temp_temp_read_column <= self.xbar_column:
                                while temp_read_row > 0:
                                    if temp_read_row <= self.xbar_row:
                                        read_row.append(num_occupied_group * [temp_read_row])
                                    else:
                                        read_row.append(num_occupied_group * [self.xbar_row])
                                    read_column.append(num_occupied_group * [temp_temp_read_column])
                                    temp_read_row -= self.xbar_row
                            else:
                                while temp_read_row > 0:
                                    if temp_read_row <= self.xbar_row:
                                        read_row.append(num_occupied_group * [temp_read_row])
                                    else:
                                        read_row.append(num_occupied_group * [self.xbar_row])
                                    read_column.append(num_occupied_group * [self.xbar_column])
                                    temp_read_row -= self.xbar_row
                            temp_temp_read_column -= self.xbar_column
                            # print("temp_read_colmn", temp_read_column)
                            # print("temp_read_row*", temp_read_row)
                        print("read_row", read_row)
                        print("read_column", read_column)
                        print("PE usage is ", len(read_row))
                        self.bank_list[i][index].bank_read_config(layer_num=i,
                                                                  activation_precision=self.activation_precision[i],
                                                                  sliding_times=self.sliding_times[i],
                                                                  read_row=read_row, read_column=read_column)
                    print("ojbk")
                    index += 1
            print("index",index)
            i += 1


if __name__ == '__main__':
    # print("ok")
    net_structure_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "mnist_net.pt")
    # net_structure_path = "/Users/zzh/Desktop/lab/MNSIM_python_v1.2/mnist_net.pt"
    SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    # SimConfig_path = "/Users/zzh/Desktop/lab/MNSIM_python_v1.2/SimConfig.ini"
    # _bank = bank(SimConfig_path)
    # print(net_structure_path)
    # print(SimConfig_path)
    _bm = behavior_mapping(net_structure_path,SimConfig_path)
    _bm.set_behavior_mapping()