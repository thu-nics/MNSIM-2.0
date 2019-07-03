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
            # print(self.bank_num[i])
            for j in range(self.bank_num[i]):
                __temp_bank = bank(self.SimConfig_path)
                self.bank_list[i].append(__temp_bank)
                self.bank_list[i][j].bank_read_config()
            i += 1


if __name__ == '__main__':
    # print("ok")
    # net_structure_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "mnist_net.pt")
    net_structure_path = "/Users/zzh/Desktop/lab/MNSIM_python_v1.1/mnist_net.pt"
    # SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    SimConfig_path = "/Users/zzh/Desktop/lab/MNSIM_python_v1.1/SimConfig.ini"
    # _bank = bank(SimConfig_path)
    # print(net_structure_path)
    # print(SimConfig_path)
    _bm = behavior_mapping(net_structure_path,SimConfig_path)
    _bm.set_behavior_mapping()