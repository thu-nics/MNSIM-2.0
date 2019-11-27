#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import configparser as cp
work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
from MNSIM.Hardware_Model.PE import ProcessElement
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Interface.interface import *


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
        self.buf_wlatency = self.buf.buf_wlatency
          # unit: ns
        self.digital_period = 1/float(PEl_config.get('Digital module', 'Digital_Frequency'))*1e3
        self.buf.calculate_buf_read_latency(rdata)
        self.buf_rlatency = self.buf.buf_rlatency
        multiple_time = math.ceil(inprecision/self.PE.DAC_precision) * math.ceil(read_row/self.PE.PE_group_DAC_num) *\
                        math.ceil(read_column/self.PE.PE_group_ADC_num)
        self.PE.calculate_xbar_read_latency()

        Transistor_Tech = int(PEl_config.get('Crossbar level', 'Transistor_Tech'))
        XBar_size = list(map(float, PEl_config.get('Crossbar level', 'Xbar_Size').split(',')))
        DAC_num = int(PEl_config.get('Process element level', 'DAC_Num'))
        ADC_num = int(PEl_config.get('Process element level', 'ADC_Num'))

        Row = XBar_size[0]
        Column = XBar_size[1]
        # ns  (using NVSim)
        decoderLatency_dict = {
            1:0.27933 # 1:8, technology 65um
        }
        if Transistor_Tech is 65:
            decoder1_8 = decoderLatency_dict[1]
            m = 1
            while DAC_num > 0:
                DAC_num = DAC_num // 8
                m += 1
            self.decoderLatency = m * decoder1_8
        else:
            raise RuntimeError("No this kind of technology")

        # ns
        muxLatency_dict = {
            1:32.744/1000
        }
        if Transistor_Tech is 65:
            mux8_1 = muxLatency_dict[1]
            m = 1
            while ADC_num > 0:
                ADC_num = ADC_num // 8
                m += 1
            self.muxLatency = m * mux8_1
        else:
            raise RuntimeError("No this kind of technology")

        # TODO: Consider the bit number of the cell
        # 32KB 0.154ps; 64KB 0.616ps; 128KB 2.463ps; 256KB 9.851ps
        size = Row * Column / 1024 / 8 #KB
        self.RC_delay = 0.001 * (0.0002 * size**2 + 5 * 10**-6 * size + 4 * 10**-14) # ns

        self.xbar_latency = multiple_time * (self.PE.xbar_read_latency + self.RC_delay)
        self.PE.calculate_DAC_latency()
        self.DAC_latency = multiple_time * self.PE.DAC_latency
        self.PE.calculate_ADC_latency()
        self.ADC_latency = multiple_time * self.PE.ADC_latency
        self.ireg_latency = multiple_time*self.digital_period
        self.shiftadd_latency = multiple_time*self.digital_period
        self.computing_latency = self.ireg_latency+self.DAC_latency+self.xbar_latency+self.ADC_latency+self.shiftadd_latency
        # print(self.computing_latency/3*2)
        self.inPE_add_latency = math.ceil(math.log2(self.PE.group_num))*self.digital_period
        self.oreg_latency = self.digital_period
        self.PE_latency = self.buf_wlatency + self.buf_rlatency + self.computing_latency + self.inPE_add_latency +\
                          self.oreg_latency + self.muxLatency + self.decoderLatency
    def update_PE_latency(self, indata=0, rdata=0):
        # update the latency computing when indata and rdata change
        self.buf.calculate_buf_write_latency(indata)
        self.buf_wlatency = self.buf.buf_wlatency
        self.buf.calculate_buf_read_latency(rdata)
        self.buf_rlatency = self.buf.buf_rlatency
        self.PE_latency = self.buf_wlatency + self.buf_rlatency + self.computing_latency + self.inPE_add_latency + self.oreg_latency


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    _test = PE_latency_analysis(test_SimConfig_path, 100,100,32,96)
    print(_test)
