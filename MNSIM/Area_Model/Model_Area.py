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
import pandas as pd
from MNSIM.Hardware_Model.Tile import tile
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.Adder import adder
import math
class Model_area():
    def __init__(self, NetStruct, SimConfig_path, multiple=None, TCG_mapping=None):
        self.NetStruct = NetStruct
        self.SimConfig_path = SimConfig_path
        # modelL_config = cp.ConfigParser()
        # modelL_config.read(self.SimConfig_path, encoding='UTF-8')
        # NoC_Compute = int(modelL_config.get('Algorithm Configuration', 'NoC_enable'))
        if multiple is None:
            multiple = [1] * len(self.NetStruct)
        if TCG_mapping is None:
            TCG_mapping = TCG(NetStruct,SimConfig_path,multiple)
        self.graph = TCG_mapping
        self.total_layer_num = self.graph.layer_num
        self.arch_area = self.total_layer_num * [0]
        self.arch_xbar_area = self.total_layer_num * [0]
        self.arch_ADC_area = self.total_layer_num * [0]
        self.arch_DAC_area = self.total_layer_num * [0]
        self.arch_digital_area = self.total_layer_num * [0]
        self.arch_adder_area = self.total_layer_num * [0]
        self.arch_shiftreg_area = self.total_layer_num * [0]
        self.arch_iReg_area = self.total_layer_num * [0]
        self.arch_oReg_area = self.total_layer_num * [0]
        self.arch_input_demux_area = self.total_layer_num * [0]
        self.arch_output_mux_area = self.total_layer_num * [0]
        self.arch_jointmodule_area = self.total_layer_num * [0]
        self.arch_buf_area = self.total_layer_num * [0]
        self.arch_pooling_area = self.total_layer_num * [0]
        self.arch_total_area = 0
        self.arch_total_xbar_area = 0
        self.arch_total_ADC_area = 0
        self.arch_total_DAC_area = 0
        self.arch_total_digital_area = 0
        self.arch_total_adder_area = 0
        self.arch_total_shiftreg_area = 0
        self.arch_total_iReg_area = 0
        self.arch_total_oReg_area = 0
        self.arch_total_input_demux_area = 0
        self.arch_total_jointmodule_area = 0
        self.arch_total_buf_area = 0
        self.arch_total_output_mux_area = 0
        self.arch_total_pooling_area = 0
        self.arch_xbar_utilization = self.total_layer_num * [0]
        self.arch_DAC_utilization = self.total_layer_num * [0]
        self.arch_ADC_utilization = self.total_layer_num * [0]
        self.arch_total_xbar_utilization = 0
        self.arch_total_DAC_utilization = 0
        self.arch_total_ADC_utilization = 0
        # print(data.columns)
        # if NoC_Compute == 1:
        #     path = os.getcwd() + '/Final_Results/'
        #     data = pd.read_csv(path + 'area.csv')
        #     self.arch_Noc_area = float(data.columns[0].split(' ')[-2])
        # else:
        #     self.arch_Noc_area = 0
        self.calculate_model_area()

    def calculate_model_area(self): #Todo: Noc area

        self.graph.tile.calculate_tile_area(SimConfig_path=self.SimConfig_path,
                                            default_inbuf_size = self.graph.max_inbuf_size,
                                            default_outbuf_size = self.graph.max_outbuf_size)
        self.global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=1,default_buf_size=self.graph.global_buf_size)
        self.global_buf.calculate_buf_area()
        self.global_add = adder(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_adder_bitwidth)
        self.global_add.calculate_adder_area()
        self.tile = tile(SimConfig_path=self.SimConfig_path)
        self.tile_xbar_num = self.tile.tile_PE_total_num*self.tile.group_num*self.tile.xbar_column*self.tile.xbar_row
        self.tile_DAC_num = self.tile.tile_PE_total_num*self.tile.group_num*self.tile.xbar_row
        self.tile_ADC_num = self.tile.tile_PE_total_num*self.tile.group_num*self.tile.xbar_column
        total_tile_num = 0
        used_total_xbar_num = 0
        used_total_DAC_num = 0
        used_total_ADC_num = 0
        # not the real DAC/ADC num, but it reflects the DAC/ADC num
        for i in range(self.total_layer_num):
            layer_dict = self.NetStruct[i][0][0]
            tile_num = self.graph.layer_tileinfo[i]['tilenum']
            self.arch_area[i] = self.graph.tile.tile_area * tile_num
            self.arch_xbar_area[i] = self.graph.tile.tile_xbar_area * tile_num
            self.arch_ADC_area[i] = self.graph.tile.tile_ADC_area * tile_num
            self.arch_DAC_area[i] = self.graph.tile.tile_DAC_area * tile_num
            self.arch_digital_area[i] = self.graph.tile.tile_digital_area * tile_num
            self.arch_adder_area[i] = self.graph.tile.tile_adder_area * tile_num
            self.arch_shiftreg_area[i] = self.graph.tile.tile_shiftreg_area * tile_num
            self.arch_iReg_area[i] = self.graph.tile.tile_iReg_area * tile_num
            self.arch_oReg_area[i] = self.graph.tile.tile_oReg_area * tile_num
            self.arch_input_demux_area[i] = self.graph.tile.tile_input_demux_area * tile_num
            self.arch_output_mux_area[i] = self.graph.tile.tile_output_mux_area * tile_num
            self.arch_jointmodule_area[i] = self.graph.tile.tile_jointmodule_area * tile_num
            self.arch_buf_area[i] = self.graph.tile.tile_buffer_area * tile_num
            self.arch_pooling_area[i] = self.graph.tile.tile_pooling_area * tile_num
            if self.graph.layer_tileinfo[i]['type'] == 'conv':
            # only consider the utilization rate of conv layer and fc layer
                total_tile_num += tile_num
                used_xbar_num = self.graph.layer_tileinfo[i]['x_width']*self.graph.layer_tileinfo[i]['y_height']
                used_DAC_num = self.graph.layer_tileinfo[i]['y_height']*self.graph.layer_tileinfo[i]['weight_precision']*math.ceil(int(layer_dict['Outputchannel']) / self.tile.xbar_column)
                used_ADC_num = self.graph.layer_tileinfo[i]['x_width']*self.graph.layer_tileinfo[i]['my']
                self.arch_xbar_utilization[i] = used_xbar_num/(tile_num*self.tile_xbar_num)
                self.arch_DAC_utilization[i] = used_DAC_num/(tile_num*self.tile_DAC_num)
                self.arch_ADC_utilization[i] = used_ADC_num/(tile_num*self.tile_ADC_num)
                used_total_xbar_num += used_xbar_num
                used_total_DAC_num += used_DAC_num
                used_total_ADC_num += used_ADC_num
            if self.graph.layer_tileinfo[i]['type'] == 'fc':
            # only consider the utilization rate of conv layer and fc layer
                total_tile_num += tile_num
                used_xbar_num = self.graph.layer_tileinfo[i]['x_width']*self.graph.layer_tileinfo[i]['y_height']
                used_DAC_num = self.graph.layer_tileinfo[i]['y_height']*self.graph.layer_tileinfo[i]['weight_precision']*math.ceil(int(layer_dict['Outfeature']) / self.tile.xbar_column)
                used_ADC_num = self.graph.layer_tileinfo[i]['x_width']*self.graph.layer_tileinfo[i]['my']
                self.arch_xbar_utilization[i] = used_xbar_num/(tile_num*self.tile_xbar_num)
                self.arch_DAC_utilization[i] = used_DAC_num/(tile_num*self.tile_DAC_num)
                self.arch_ADC_utilization[i] = used_ADC_num/(tile_num*self.tile_ADC_num)
                used_total_xbar_num += used_xbar_num
                used_total_DAC_num += used_DAC_num
                used_total_ADC_num += used_ADC_num
        self.arch_total_area = sum(self.arch_area)
        self.arch_total_xbar_area = sum(self.arch_xbar_area)
        self.arch_total_ADC_area = sum(self.arch_ADC_area)
        self.arch_total_DAC_area = sum(self.arch_DAC_area)
        self.arch_total_digital_area = sum(self.arch_digital_area)+self.global_add.adder_area*self.graph.global_adder_num
        self.arch_total_adder_area = sum(self.arch_adder_area)+self.global_add.adder_area*self.graph.global_adder_num
        self.arch_total_shiftreg_area = sum(self.arch_shiftreg_area)
        self.arch_total_iReg_area = sum(self.arch_iReg_area)
        self.arch_total_oReg_area = sum(self.arch_oReg_area)
        self.arch_total_input_demux_area = sum(self.arch_input_demux_area)
        self.arch_total_output_mux_area = sum(self.arch_output_mux_area)
        self.arch_total_jointmodule_area = sum(self.arch_jointmodule_area)
        self.arch_total_buf_area = sum(self.arch_buf_area)+self.global_buf.buf_area
        self.arch_total_pooling_area = sum(self.arch_pooling_area)
        self.arch_total_xbar_utilization = used_total_xbar_num/(total_tile_num*self.tile_xbar_num)
        self.arch_total_DAC_utilization = used_total_DAC_num/(total_tile_num*self.tile_DAC_num)
        self.arch_total_ADC_utilization = used_total_ADC_num/(total_tile_num*self.tile_ADC_num)

    def model_area_output(self, module_information = 1, layer_information = 1):
        print("Hardware area:", self.arch_total_area, "um^2")
        if module_information:
            print("		crossbar area:", self.arch_total_xbar_area, "um^2")
            print("		DAC area:", self.arch_total_DAC_area, "um^2")
            print("		ADC area:", self.arch_total_ADC_area, "um^2")
            print("		Buffer area:", self.arch_total_buf_area, "um^2")
            print("		Pooling area:", self.arch_total_pooling_area, "um^2")
            print("		Other digital part area:", self.arch_total_digital_area, "um^2")
            print("			|---adder area:", self.arch_total_adder_area, "um^2")
            print("			|---output-shift-reg area:", self.arch_total_shiftreg_area, "um^2")
            print("			|---input-reg area:", self.arch_total_iReg_area, "um^2")
            print("			|---output-reg area:", self.arch_total_oReg_area, "um^2")
            print("			|---input_demux area:", self.arch_total_input_demux_area, "um^2")
            print("			|---output_mux area:", self.arch_total_output_mux_area, "um^2")
            print("			|---joint_module area:", self.arch_total_jointmodule_area, "um^2")
            print("		crossbar utilization rate: ", self.arch_total_xbar_utilization*100, "%",sep='')
            print("		DAC utilization rate: ", self.arch_total_DAC_utilization*100, "%",sep='')
            print("		ADC utilization rate: ", self.arch_total_ADC_utilization*100, "%",sep='')
            # print("		NoC part area:", self.arch_Noc_area, "um^2")
        if layer_information:
            for i in range(self.total_layer_num):
                print("Layer", i, ":")
                layer_dict = self.NetStruct[i][0][0]
                if layer_dict['type'] == 'element_sum':
                    print("     Hardware area (global accumulator):", self.global_add.adder_area*self.graph.global_adder_num+self.global_buf.buf_area, "um^2")
                else:
                    print("     Hardware area:", self.arch_area[i], "um^2")
                if layer_dict['type'] == 'conv' or layer_dict['type'] == 'fc':
                    print("		crossbar utilization rate: ", self.arch_xbar_utilization[i]*100, "%",sep='')
                    print("		DAC utilization rate: ", self.arch_DAC_utilization[i]*100, "%",sep='')
                    print("		ADC utilization rate: ", self.arch_ADC_utilization[i]*100, "%",sep='')

if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "vgg8_params.pth")

    __TestInterface = TrainTestInterface('vgg8_128_9', 'MNSIM.Interface.cifar10', test_SimConfig_path,
                                         test_weights_file_path)
    structure_file = __TestInterface.get_structure()
    __TCG_mapping = TCG(structure_file, test_SimConfig_path)
    __area = Model_area(NetStruct=structure_file,SimConfig_path=test_SimConfig_path,TCG_mapping=__TCG_mapping)
    __area.model_area_output(1,1)