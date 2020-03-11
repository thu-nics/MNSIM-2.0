#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import configparser as cp
work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Interface.interface import *

class pooling_latency_analysis():
    def __init__(self, SimConfig_path, indata=0, rdata=0):
        # indata: volume of input data (for pooling) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
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
        # Todo: Add the parameter into config file
        self.pooling_size = 9

        # Todo: update pooling latency estimation
    def update_pooling_latency(self, actual_num=128, layer_size=9, indata=0, rdata=0):
        # update the latency computing when indata and rdata change
        self.buf.calculate_buf_write_latency(indata)
        self.buf_wlatency = self.buf.buf_wlatency
        self.buf.calculate_buf_read_latency(rdata)
        self.buf_rlatency = self.buf.buf_rlatency
        # self.pooling_latency = self.buf_wlatency + self.digital_period + self.buf_wlatency
        pooling_times = math.ceil(actual_num / layer_size) * math.ceil(layer_size / self.pooling_size)
        self.pooling_latency = self.buf_wlatency + self.buf_rlatency + pooling_times*self.digital_period



if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    _test = pooling_latency_analysis(test_SimConfig_path, 8, 4)
    print(_test)
