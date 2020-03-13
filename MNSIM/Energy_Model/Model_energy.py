#!/usr/bin/python
# -*-coding:utf-8-*-
import sys
import os
import configparser as cp
work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
import numpy as np
from MNSIM.Interface.interface import *
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Hardware_Model.Tile import tile
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from MNSIM.Latency_Model.Model_latency import Model_latency

class Model_energy():
    def __init__(self,NetStruct,SimConfig_path,model_power=None,
                 model_latency=None,multiple=None,TCG_mapping=None):
        self.NetStruct = NetStruct
        self.SimConfig_path = SimConfig_path
        if multiple is None:
            multiple = [1] * len(self.NetStruct)
        if TCG_mapping is None:
            TCG_mapping = TCG(NetStruct, SimConfig_path, multiple)
        self.graph = TCG_mapping
        self.total_layer_num = self.graph.layer_num
        if model_latency is None:
            self.model_latency = Model_latency(NetStruct,SimConfig_path,multiple,TCG_mapping)
            self.model_latency.calculate_model_latency(mode=2)
        else:
            self.model_latency = model_latency
        if model_power is None:
            self.model_power = Model_inference_power(NetStruct,SimConfig_path,multiple,TCG_mapping)
        else:
            self.model_power = model_power
        self.arch_energy = self.total_layer_num * [0]
        self.arch_xbar_energy = self.total_layer_num * [0]
        self.arch_ADC_energy = self.total_layer_num * [0]
        self.arch_DAC_energy = self.total_layer_num * [0]
        self.arch_digital_energy = self.total_layer_num * [0]
        self.arch_adder_energy = self.total_layer_num * [0]
        self.arch_shiftreg_energy = self.total_layer_num * [0]
        self.arch_iReg_energy = self.total_layer_num * [0]
        self.arch_input_demux_energy = self.total_layer_num * [0]
        self.arch_output_mux_energy = self.total_layer_num * [0]
        self.arch_jointmodule_energy = self.total_layer_num * [0]
        self.arch_buf_energy = self.total_layer_num * [0]
        self.arch_buf_r_energy = self.total_layer_num * [0]
        self.arch_buf_w_energy = self.total_layer_num * [0]
        self.arch_pooling_energy = self.total_layer_num * [0]
        self.arch_total_energy = 0
        self.arch_total_xbar_energy = 0
        self.arch_total_ADC_energy = 0
        self.arch_total_DAC_energy = 0
        self.arch_total_digital_energy = 0
        self.arch_total_adder_energy = 0
        self.arch_total_shiftreg_energy = 0
        self.arch_total_iReg_energy = 0
        self.arch_total_input_demux_energy = 0
        self.arch_total_jointmodule_energy = 0
        self.arch_total_buf_energy = 0
        self.arch_total_buf_r_energy = 0
        self.arch_total_buf_w_energy = 0
        self.arch_total_output_mux_energy = 0
        self.arch_total_pooling_energy = 0
        self.calculate_model_energy()
    def calculate_model_energy(self):
        for i in range(self.total_layer_num):
            tile_num = self.graph.layer_tileinfo[i]['tilenum']
            self.arch_xbar_energy[i] = self.model_power.arch_xbar_power[i]*self.model_latency.total_xbar_latency[i]
            self.arch_ADC_energy[i] = self.model_power.arch_ADC_power[i]*self.model_latency.total_ADC_latency[i]
            self.arch_DAC_energy[i] = self.model_power.arch_DAC_power[i]*self.model_latency.total_DAC_latency[i]
            self.arch_adder_energy[i] = self.model_power.arch_adder_power[i]*self.model_latency.total_adder_latency[i]
            self.arch_shiftreg_energy[i] = self.model_power.arch_shiftreg_power[i]*self.model_latency.total_shiftreg_latency[i]
            self.arch_iReg_energy[i] = self.model_power.arch_iReg_power[i]*self.model_latency.total_iReg_latency[i]
            self.arch_input_demux_energy[i] = self.model_power.arch_input_demux_power[i]*self.model_latency.total_input_demux_latency[i]
            self.arch_output_mux_energy[i] = self.model_power.arch_output_mux_power[i]*self.model_latency.total_output_mux_latency[i]
            self.arch_jointmodule_energy[i] = self.model_power.arch_jointmodule_power[i]*self.model_latency.total_jointmodule_latency[i]
            self.arch_buf_r_energy[i] = self.model_power.arch_buf_r_power[i]*self.model_latency.total_buffer_r_latency[i]
            self.arch_buf_w_energy[i] = self.model_power.arch_buf_w_power[i]*self.model_latency.total_buffer_w_latency[i]
            self.arch_buf_energy[i] = self.arch_buf_r_energy[i] + self.arch_buf_w_energy[i]
            self.arch_pooling_energy[i] = self.model_power.arch_pooling_power[i]*self.model_latency.total_pooling_latency[i]
            self.arch_digital_energy[i] = self.arch_shiftreg_energy[i]+self.arch_iReg_energy[i]+\
                                          self.arch_input_demux_energy[i]+self.arch_output_mux_energy[i]+self.arch_jointmodule_energy[i]
            self.arch_energy[i] = self.arch_xbar_energy[i]+self.arch_ADC_energy[i]+self.arch_DAC_energy[i]+\
                                  self.arch_digital_energy[i]+self.arch_buf_energy[i]+self.arch_pooling_energy[i]
        self.arch_total_energy = sum(self.arch_energy)
        self.arch_total_xbar_energy = sum(self.arch_xbar_energy)
        self.arch_total_ADC_energy = sum(self.arch_ADC_energy)
        self.arch_total_DAC_energy = sum(self.arch_DAC_energy)
        self.arch_total_digital_energy = sum(self.arch_digital_energy)
        self.arch_total_adder_energy = sum(self.arch_adder_energy)
        self.arch_total_shiftreg_energy = sum(self.arch_shiftreg_energy)
        self.arch_total_iReg_energy = sum(self.arch_iReg_energy)
        self.arch_total_input_demux_energy = sum(self.arch_input_demux_energy)
        self.arch_total_output_mux_energy = sum(self.arch_output_mux_energy)
        self.arch_total_jointmodule_energy = sum(self.arch_jointmodule_energy)
        self.arch_total_buf_energy = sum(self.arch_buf_energy)
        self.arch_total_pooling_energy = sum(self.arch_pooling_energy)

    def model_energy_output(self, module_information = 1, layer_information = 1):
        print("Hardware energy:", self.arch_total_energy, "nJ")
        if module_information:
            print("		crossbar energy:", self.arch_total_xbar_energy, "nJ")
            print("		DAC energy:", self.arch_total_DAC_energy, "nJ")
            print("		ADC energy:", self.arch_total_ADC_energy, "nJ")
            print("		Buffer energy:", self.arch_total_buf_energy, "nJ")
            print("			|---read buffer energy:", self.arch_total_buf_r_energy, "nJ")
            print("			|---write buffer energy:", self.arch_total_buf_w_energy, "nJ")
            print("		Pooling energy:", self.arch_total_pooling_energy, "nJ")
            print("		Other digital part energy:", self.arch_total_digital_energy, "nJ")
            print("			|---adder energy:", self.arch_total_adder_energy, "nJ")
            print("			|---output-shift-reg energy:", self.arch_total_shiftreg_energy, "nJ")
            print("			|---input-reg energy:", self.arch_total_iReg_energy, "nJ")
            print("			|---input_demux energy:", self.arch_total_input_demux_energy, "nJ")
            print("			|---output_mux energy:", self.arch_total_output_mux_energy, "nJ")
            print("			|---joint_module energy:", self.arch_total_jointmodule_energy, "nJ")
        if layer_information:
            for i in range(self.total_layer_num):
                print("Layer", i, ":")
                print("     Hardware energy:", self.arch_energy[i], "nJ")
if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "vgg8_params.pth")

    __TestInterface = TrainTestInterface('vgg8_128_9', 'MNSIM.Interface.cifar10', test_SimConfig_path,
                                         test_weights_file_path, 'cpu')
    structure_file = __TestInterface.get_structure()
    __TCG_mapping = TCG(structure_file, test_SimConfig_path)
    __energy = Model_energy(NetStruct=structure_file,SimConfig_path=test_SimConfig_path,TCG_mapping=__TCG_mapping)
    __energy.model_energy_output(1,1)