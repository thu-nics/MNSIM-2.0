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


class bank_latency_analysis(PE_latency_analysis):
    def __init__(self, SimConfig_path, read_row=0, read_column=0, indata=0, rdata=0, inprecision = 8, PE_num=0):
        # read_row: activated WL number in crossbar
        # read_column: activated BL number in crossbar
        # indata: volume of input data (for PE) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # outdata: volume of output data (for PE) (Byte)
        # inprecision: input data precision of each Xbar
        # PE_num: used PE_number in one bank
        PE_latency_analysis.__init__(self, SimConfig_path, read_row=read_row, read_column=read_column,
                                     indata=indata, rdata=rdata, inprecision=inprecision)
        bankl_config = cp.ConfigParser()
        bankl_config.read(SimConfig_path, encoding='UTF-8')
        self.intra_bank_bandwidth = float(bankl_config.get('Bank level', 'Intra_Bank_Bandwidth'))
        merge_time = math.ceil(math.log2(PE_num))
        self.bank_PE_num = list(map(int, bankl_config.get('Bank level', 'PE_Num').split(',')))
        if self.bank_PE_num[0] == 0:
            self.bank_PE_num[0] = 4
            self.bank_PE_num[1] = 4
        assert self.bank_PE_num[0] > 0, "PE number in one PE < 0"
        assert self.bank_PE_num[1] > 0, "PE number in one PE < 0"
        self.bank_PE_total_num = self.bank_PE_num[0] * self.bank_PE_num[1]
        assert PE_num <= self.bank_PE_total_num, "PE number exceeds the range"
        total_level = math.ceil(math.log2(self.bank_PE_total_num))
        self.merge_latency = merge_time * self.digital_period
        self.transfer_latency = (total_level*(self.PE.ADC_precision+merge_time)-merge_time*(merge_time+1)/2)\
                                *read_column/self.intra_bank_bandwidth
        self.buf.calculate_buf_write_latency(wdata=(self.PE.ADC_precision+merge_time)*read_column/8)
        self.bank_buf_wtime = self.buf.buf_wlatency
         # do not consider
        self.bank_latency = self.PE_latency + self.merge_latency + self.transfer_latency
    def update_bank_latency(self, indata = 0, rdata = 0):
        self.update_PE_latency(indata=indata,rdata=rdata)
        self.bank_latency = self.PE_latency + self.merge_latency + self.transfer_latency


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    _test = bank_latency_analysis(test_SimConfig_path, 100,100,32,96,8,8)
    print(_test)