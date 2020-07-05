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
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.Adder import adder
class Model_inference_power():
    def __init__(self, NetStruct, SimConfig_path, multiple=None, TCG_mapping=None):
        self.NetStruct = NetStruct
        self.SimConfig_path = SimConfig_path
        if multiple is None:
            multiple = [1] * len(self.NetStruct)
        if TCG_mapping is None:
            TCG_mapping = TCG(NetStruct,SimConfig_path,multiple)
        self.graph = TCG_mapping
        self.total_layer_num = self.graph.layer_num
        self.arch_power = self.total_layer_num * [0]
        self.arch_xbar_power = self.total_layer_num * [0]
        self.arch_ADC_power = self.total_layer_num * [0]
        self.arch_DAC_power = self.total_layer_num * [0]
        self.arch_digital_power = self.total_layer_num * [0]
        self.arch_adder_power = self.total_layer_num * [0]
        self.arch_shiftreg_power = self.total_layer_num * [0]
        self.arch_iReg_power = self.total_layer_num * [0]
        self.arch_oReg_power = self.total_layer_num * [0]
        self.arch_input_demux_power = self.total_layer_num * [0]
        self.arch_output_mux_power = self.total_layer_num * [0]
        self.arch_jointmodule_power = self.total_layer_num * [0]
        self.arch_buf_power = self.total_layer_num * [0]
        self.arch_buf_r_power = self.total_layer_num * [0]
        self.arch_buf_w_power = self.total_layer_num * [0]
        self.arch_pooling_power = self.total_layer_num * [0]
        self.arch_total_power = 0
        self.arch_total_xbar_power = 0
        self.arch_total_ADC_power = 0
        self.arch_total_DAC_power = 0
        self.arch_total_digital_power = 0
        self.arch_total_adder_power = 0
        self.arch_total_shiftreg_power = 0
        self.arch_total_iReg_power = 0
        self.arch_total_oReg_power = 0
        self.arch_total_input_demux_power = 0
        self.arch_total_jointmodule_power = 0
        self.arch_total_buf_power = 0
        self.arch_total_buf_r_power = 0
        self.arch_total_buf_w_power = 0
        self.arch_total_output_mux_power = 0
        self.arch_total_pooling_power = 0
        self.calculate_model_power()

    def calculate_model_power(self):
        self.global_buf = buffer(SimConfig_path=self.SimConfig_path, buf_level=1,
                                 default_buf_size=self.graph.global_buf_size)
        self.global_buf.calculate_buf_read_power()
        self.global_buf.calculate_buf_write_power()
        self.global_add = adder(SimConfig_path=self.SimConfig_path, bitwidth=self.graph.global_adder_bitwidth)
        self.global_add.calculate_adder_power()
        for i in range(self.total_layer_num):
            tile_num = self.graph.layer_tileinfo[i]['tilenum']
            max_column = self.graph.layer_tileinfo[i]['max_column']
            max_row = self.graph.layer_tileinfo[i]['max_row']
            max_PE = self.graph.layer_tileinfo[i]['max_PE']
            max_group = self.graph.layer_tileinfo[i]['max_group']
            layer_type = self.graph.net[i][0][0]['type']
            self.graph.tile.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                           SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                           default_outbuf_size=self.graph.max_outbuf_size)
            self.arch_power[i] = self.graph.tile.tile_read_power * tile_num
            self.arch_xbar_power[i] = self.graph.tile.tile_xbar_read_power * tile_num
            self.arch_ADC_power[i] = self.graph.tile.tile_ADC_read_power * tile_num
            self.arch_DAC_power[i] = self.graph.tile.tile_DAC_read_power * tile_num
            self.arch_digital_power[i] = self.graph.tile.tile_digital_read_power * tile_num
            self.arch_adder_power[i] = self.graph.tile.tile_adder_read_power * tile_num
            self.arch_shiftreg_power[i] = self.graph.tile.tile_shiftreg_read_power * tile_num
            self.arch_iReg_power[i] = self.graph.tile.tile_iReg_read_power * tile_num
            self.arch_oReg_power[i] = self.graph.tile.tile_oReg_read_power * tile_num
            self.arch_input_demux_power[i] = self.graph.tile.tile_input_demux_read_power * tile_num
            self.arch_output_mux_power[i] = self.graph.tile.tile_output_mux_read_power * tile_num
            self.arch_jointmodule_power[i] = self.graph.tile.tile_jointmodule_read_power * tile_num
            self.arch_buf_power[i] = self.graph.tile.tile_buffer_read_power * tile_num
            self.arch_buf_r_power[i] = self.graph.tile.tile_buffer_r_read_power * tile_num
            self.arch_buf_w_power[i] = self.graph.tile.tile_buffer_w_read_power * tile_num
            self.arch_pooling_power[i] = self.graph.tile.tile_pooling_read_power * tile_num
        self.arch_total_power = sum(self.arch_power)
        self.arch_total_xbar_power = sum(self.arch_xbar_power)
        self.arch_total_ADC_power = sum(self.arch_ADC_power)
        self.arch_total_DAC_power = sum(self.arch_DAC_power)
        self.arch_total_digital_power = sum(self.arch_digital_power)+self.global_add.adder_power*self.graph.global_adder_num
        self.arch_total_adder_power = sum(self.arch_adder_power)+self.global_add.adder_power*self.graph.global_adder_num
        self.arch_total_shiftreg_power = sum(self.arch_shiftreg_power)
        self.arch_total_iReg_power = sum(self.arch_iReg_power)
        self.arch_total_oReg_power = sum(self.arch_oReg_power)
        self.arch_total_input_demux_power = sum(self.arch_input_demux_power)
        self.arch_total_output_mux_power = sum(self.arch_output_mux_power)
        self.arch_total_jointmodule_power = sum(self.arch_jointmodule_power)
        self.arch_total_buf_power = sum(self.arch_buf_power)+(self.global_buf.buf_wpower+self.global_buf.buf_rpower)*1e-3
        self.arch_total_buf_r_power = sum(self.arch_buf_r_power)+self.global_buf.buf_rpower*1e-3
        self.arch_total_buf_w_power = sum(self.arch_buf_w_power)+self.global_buf.buf_wpower*1e-3
        self.arch_total_pooling_power = sum(self.arch_pooling_power)
    
    def model_power_output(self, module_information = 1, layer_information = 1):
        print("Hardware power:", self.arch_total_power, "W")
        if module_information:
            print("		crossbar power:", self.arch_total_xbar_power, "W")
            print("		DAC power:", self.arch_total_DAC_power, "W")
            print("		ADC power:", self.arch_total_ADC_power, "W")
            print("		Buffer power:", self.arch_total_buf_power, "W")
            print("			|---read buffer power:", self.arch_total_buf_r_power, "W")
            print("			|---write buffer power:", self.arch_total_buf_w_power, "W")
            print("		Pooling power:", self.arch_total_pooling_power, "W")
            print("		Other digital part power:", self.arch_total_digital_power, "W")
            print("			|---adder power:", self.arch_total_adder_power, "W")
            print("			|---output-shift-reg power:", self.arch_total_shiftreg_power, "W")
            print("			|---input-reg power:", self.arch_total_iReg_power, "W")
            print("			|---output-reg power:", self.arch_total_oReg_power, "W")
            print("			|---input_demux power:", self.arch_total_input_demux_power, "W")
            print("			|---output_mux power:", self.arch_total_output_mux_power, "W")
            print("			|---joint_module power:", self.arch_total_jointmodule_power, "W")
        if layer_information:
            for i in range(self.total_layer_num):
                print("Layer", i, ":")
                layer_dict = self.NetStruct[i][0][0]
                if layer_dict['type'] == 'element_sum':
                    print("     Hardware power (global accumulator):", self.global_add.adder_power*self.graph.global_adder_num
                          +self.global_buf.buf_wpower*1e-3+self.global_buf.buf_rpower*1e-3, "W")
                else:
                    print("     Hardware power:", self.arch_power[i], "W")
if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "vgg8_params.pth")

    __TestInterface = TrainTestInterface('vgg8_128_9', 'MNSIM.Interface.cifar10', test_SimConfig_path,
                                         test_weights_file_path, 'cpu')
    structure_file = __TestInterface.get_structure()
    __TCG_mapping = TCG(structure_file, test_SimConfig_path)
    __power = Model_inference_power(NetStruct=structure_file,SimConfig_path=test_SimConfig_path,TCG_mapping=__TCG_mapping)
    __power.model_power_output(1,1)