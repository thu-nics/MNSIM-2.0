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

class Pooling_latency_analysis():
    def __init__(self, SimConfig_path, indata=0, rdata=0):
        # read_row: activated WL number in crossbar
        # read_column: activated BL number in crossbar
        # indata: volume of input data (for PE) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # outdata: volume of output data (for PE) (Byte)
        # inprecision: input data precision of each Xbar
        Poolingl_config = cp.ConfigParser()
        Poolingl_config.read(SimConfig_path, encoding='UTF-8')
        self.buf = buffer(SimConfig_path)
        self.buf.calculate_buf_write_latency(indata)
        self.buf_wlatency = self.buf.buf_wlatency
          # unit: ns
        self.buf.calculate_buf_read_latency(rdata)
        self.buf_rlatency = self.buf.buf_rlatency
        self.digital_period = 1/float(Poolingl_config.get('Digital module', 'Digital_Frequency'))*1e3
        self.pooling_latency = self.buf_wlatency + self.digital_period + self.buf_wlatency

if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    _test = Pooling_latency_analysis(test_SimConfig_path, 8, 4)
    print(_test)