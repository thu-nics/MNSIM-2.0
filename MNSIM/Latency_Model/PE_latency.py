#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import configparser as cp
work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
from MNSIM.Hardware_Model import *
from MNSIM.Hardware_Model.Crossbar import crossbar
from MNSIM.Hardware_Model.PE import ProcessElement
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.Bank import bank
from MNSIM.Interface.interface import *
from MNSIM.Mapping_Model.Bank_connection_graph import BCG
import collections

class PE_latency_analysis():
    def __init__(self, SimConfig_path, read_row=0, read_column=0, indata=0, rdata=0, inprecision = 8):
        # read_row: activated WL number in crossbar
        # read_column: activated BL number in crossbar
        # indata: volume of input data (for PE) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # outdata: volume of output data (for PE) (Byte)
        # inprecision: input data precision of each Xbar
        PEl_config = cp.ConfigParser()
        PEl_config.read(SimConfig_path, encoding='UTF-8')
        self.buf = buffer(SimConfig_path)
        self.PE = ProcessElement(SimConfig_path)
        self.buf.calculate_buf_write_latency(indata)
        self.buf_wtime = self.buf.buf_wlatency
          # unit: ns
        self.digital_period = 1/float(PEl_config.get('Digital module', 'Digital_Frequency'))*1e3
        self.buf.calculate_buf_read_latency(rdata)
        self.buf_rtime = self.buf.buf_rlatency
        multiple_time = math.ceil(inprecision/self.PE.DAC_precision) * math.ceil(read_row/self.PE.PE_group_DAC_num) *\
                        math.ceil(read_column/self.PE.PE_group_ADC_num)
        self.PE.calculate_xbar_read_latency()
        self.xbar_time = multiple_time * self.PE.xbar_read_latency
        self.PE.calculate_DAC_latency()
        self.DAC_time = multiple_time * self.PE.DAC_latency
        self.PE.calculate_ADC_latency()
        self.ADC_time = multiple_time * self.PE.ADC_latency
        self.ireg_time = multiple_time*self.digital_period
        self.shiftadd_time = multiple_time*self.digital_period
        self.computing_time = self.ireg_time+self.DAC_time+self.xbar_time+self.ADC_time+self.shiftadd_time
        self.inPE_add_time = math.ceil(math.log2(self.PE.group_num))*self.digital_period
        self.oreg_time = self.digital_period
        self.PE_latency = self.buf_wtime + self.buf_rtime + self.computing_time + self.inPE_add_time + self.oreg_time

if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    _test = PE_latency_analysis(test_SimConfig_path, 100,100,32,96)
    print(_test)