#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import configparser as cp
work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
from MNSIM.Interface.interface import *
from MNSIM.Latency_Model.PE_latency import PE_latency_analysis
from MNSIM.Hardware_Model.Buffer import buffer


class tile_latency_analysis(PE_latency_analysis):
    def __init__(self, SimConfig_path, read_row=0, read_column=0, indata=0, rdata=0, inprecision = 8,
                 PE_num=0, default_inbuf_size = 16, default_outbuf_size =4):
        # read_row: activated WL number in crossbar
        # read_column: activated BL number in crossbar
        # indata: volume of input data (for PE) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # outdata: volume of output data (for PE) (Byte)
        # inprecision: input data precision of each Xbar
        # PE_num: used PE_number in one tile
        # default_inbuf_size: the default PE-level input buffer size (unit: KB)
        # default_outbuf_size: the default Tile-level output buffer size (unit: KB)
        PE_latency_analysis.__init__(self, SimConfig_path, read_row=read_row, read_column=read_column,
                                     indata=indata, rdata=rdata, inprecision=inprecision, default_buf_size = default_inbuf_size)
        tilel_config = cp.ConfigParser()
        tilel_config.read(SimConfig_path, encoding='UTF-8')
        self.intra_tile_bandwidth = float(tilel_config.get('Tile level', 'Intra_Tile_Bandwidth'))
        merge_time = math.ceil(math.log2(PE_num))
        self.tile_PE_num = list(map(int, tilel_config.get('Tile level', 'PE_Num').split(',')))
        if self.tile_PE_num[0] == 0:
            self.tile_PE_num[0] = 4
            self.tile_PE_num[1] = 4
        assert self.tile_PE_num[0] > 0, "PE number in one PE < 0"
        assert self.tile_PE_num[1] > 0, "PE number in one PE < 0"
        self.tile_PE_total_num = self.tile_PE_num[0] * self.tile_PE_num[1]
        assert PE_num <= self.tile_PE_total_num, "PE number exceeds the range"
        self.outbuf = buffer(SimConfig_path=SimConfig_path, buf_level=2, default_buf_size=default_outbuf_size)
        total_level = math.ceil(math.log2(self.tile_PE_total_num))
        self.jointmodule_latency = merge_time * self.digital_period
        self.transfer_latency = (total_level*(self.PE.ADC_precision+merge_time)-merge_time*(merge_time+1)/2)\
                                *read_column/self.intra_tile_bandwidth
        self.outbuf.calculate_buf_write_latency(wdata=((self.PE.ADC_precision + merge_time)*read_column*PE_num/8))
        self.tile_buf_rlatency = 0
        self.tile_buf_wlatency = self.outbuf.buf_wlatency
         # do not consider
        self.tile_latency = self.PE_latency + self.jointmodule_latency + self.transfer_latency + self.tile_buf_wlatency
    def update_tile_latency(self, indata = 0, rdata = 0):
        self.update_PE_latency(indata=indata,rdata=rdata)
        self.tile_latency = self.PE_latency + self.jointmodule_latency + self.transfer_latency + self.tile_buf_wlatency


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    _test = tile_latency_analysis(test_SimConfig_path, 100, 100, 32, 96, 8, 8)
    print(_test)