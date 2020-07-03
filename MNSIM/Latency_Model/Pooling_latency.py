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
from MNSIM.Hardware_Model.Pooling import Pooling
from MNSIM.Interface.interface import *

class pooling_latency_analysis():
    def __init__(self, SimConfig_path, indata=0, rdata=0, outprecision = 8, default_inbuf_size = 16,
                 default_outbuf_size = 4, default_inchannel = 64, default_size = 9):
        # indata: volume of input data (for pooling) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # default_inbuf_size: the default PE-level input buffer size (unit: KB)
        # default_outbuf_size: the default Tile-level output buffer size (unit: KB)
        self.pooling = Pooling(SimConfig_path=SimConfig_path)
        self.inbuf = buffer(SimConfig_path=SimConfig_path, buf_level=1, default_buf_size=default_inbuf_size)
        self.inbuf.calculate_buf_write_latency(indata)
        self.inbuf_wlatency = self.inbuf.buf_wlatency
          # unit: ns
        self.inbuf.calculate_buf_read_latency(rdata)
        self.inbuf_rlatency = self.inbuf.buf_rlatency
        self.pooling.calculate_Pooling_latency(inchannel=default_inchannel, insize=default_size)
        self.digital_latency = self.pooling.Pooling_latency
        self.outbuf = buffer(SimConfig_path=SimConfig_path, buf_level=2, default_buf_size=default_outbuf_size)
        self.outbuf.calculate_buf_write_latency(wdata=(default_inchannel * outprecision / 8))
        self.outbuf_rlatency = 0
        self.outbuf_wlatency = self.outbuf.buf_wlatency
        self.pooling_latency = self.inbuf_wlatency + self.inbuf_rlatency + self.digital_latency + self.outbuf_rlatency + self.outbuf_wlatency
        # Todo: Add the parameter into config file

        # Todo: update pooling latency estimation
    def update_pooling_latency(self,indata=0, rdata=0):
        # update the latency computing when indata and rdata change
        self.inbuf.calculate_buf_write_latency(indata)
        self.inbuf_wlatency = self.inbuf.buf_wlatency
        # unit: ns
        self.inbuf.calculate_buf_read_latency(rdata)
        self.inbuf_rlatency = self.inbuf.buf_rlatency
        self.pooling_latency = self.inbuf_wlatency + self.inbuf_rlatency + self.digital_latency + self.outbuf_rlatency + self.outbuf_wlatency



if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    _test = pooling_latency_analysis(test_SimConfig_path, 8, 4)
    print(_test)
