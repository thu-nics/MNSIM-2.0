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
    def __init__(self, SimConfig_path, read_row=0, read_column=0, indata=0, rdata=0, inprecision = 8, default_buf_size = 16):
        # read_row: activated WL number in crossbar
        # read_column: activated BL number in crossbar
        # indata: volume of input data (for PE) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # outdata: volume of output data (for PE) (Byte)
        # inprecision: input data precision of each Xbar
        # default_buf_size: default input buffer size (KB)
        PEl_config = cp.ConfigParser()
        PEl_config.read(SimConfig_path, encoding='UTF-8')
        self.inbuf = buffer(SimConfig_path=SimConfig_path, buf_level=1, default_buf_size=default_buf_size)
        self.PE = ProcessElement(SimConfig_path)
        self.inbuf.calculate_buf_write_latency(indata)
        self.PE_buf_wlatency = self.inbuf.buf_wlatency
          # unit: ns
        self.digital_period = 1/float(PEl_config.get('Digital module', 'Digital_Frequency'))*1e3
        self.inbuf.calculate_buf_read_latency(rdata)
        self.PE_buf_rlatency = self.inbuf.buf_rlatency
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
            1:0.27933 # 1:8, technology 65nm
        }
        decoder1_8 = decoderLatency_dict[1]
        Row_per_DAC = math.ceil(Row/DAC_num)
        m = 1
        while Row_per_DAC > 0:
            Row_per_DAC = Row_per_DAC // 8
            m += 1
        self.decoderLatency = m * decoder1_8

        # ns
        muxLatency_dict = {
            1:32.744/1000
        }
        mux8_1 = muxLatency_dict[1]
        m = 1
        Column_per_ADC = math.ceil(Column / ADC_num)
        while Column_per_ADC > 0:
            Column_per_ADC = Column_per_ADC // 8
            m += 1
        self.muxLatency = m * mux8_1

        self.xbar_latency = multiple_time * self.PE.xbar_read_latency
        self.PE.calculate_DAC_latency()
        self.DAC_latency = multiple_time * self.PE.DAC_latency
        self.PE.calculate_ADC_latency()
        self.ADC_latency = multiple_time * self.PE.ADC_latency
        self.iReg_latency = math.ceil(read_row/self.PE.PE_group_DAC_num)*math.ceil(read_column/self.PE.PE_group_ADC_num)*self.digital_period+\
                            multiple_time*self.digital_period
            # write and read
        self.shiftreg_latency = multiple_time * self.digital_period
        self.input_demux_latency = multiple_time*self.decoderLatency
        self.adder_latency = math.ceil(read_column/self.PE.PE_group_ADC_num)*math.ceil(math.log2(self.PE.group_num))*self.digital_period
        self.output_mux_latency = multiple_time*self.muxLatency
        self.computing_latency = self.DAC_latency+self.xbar_latency+self.ADC_latency
        self.oReg_latency = math.ceil(read_column/self.PE.PE_group_ADC_num)*self.digital_period
        self.PE_digital_latency = self.iReg_latency + self.shiftreg_latency + self.input_demux_latency + \
                                  self.adder_latency + self.output_mux_latency + self.oReg_latency
        self.PE_latency = self.PE_buf_wlatency + self.PE_buf_rlatency + self.computing_latency + self.PE_digital_latency
    def update_PE_latency(self, indata=0, rdata=0):
        # update the latency computing when indata and rdata change
        self.inbuf.calculate_buf_write_latency(indata)
        self.PE_buf_wlatency = self.inbuf.buf_wlatency
        self.inbuf.calculate_buf_read_latency(rdata)
        self.PE_buf_rlatency = self.inbuf.buf_rlatency
        self.PE_latency = self.PE_buf_wlatency + self.PE_buf_rlatency + self.computing_latency + self.PE_digital_latency


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    _test = PE_latency_analysis(test_SimConfig_path, 100,100,32,96)
    print(_test)
