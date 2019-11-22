#!/usr/bin/python
# -*-coding:utf-8-*-
import sys
import os
import configparser as cp
work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
from MNSIM.Interface.interface import *
from MNSIM.Mapping_Model.Bank_connection_graph import BCG
from MNSIM.Latency_Model.Bank_latency import bank_latency_analysis
from MNSIM.Latency_Model.Pooling_latency import pooling_latency_analysis

def merge_interval(interval):
    if len(interval) == 0:
        return []
    result = []
    interval.sort()
    lower_bound = interval[0][0]
    upper_bound = interval[0][1]
    for index in range(1,len(interval)):
        if interval[index][0] > upper_bound:
            result.append([lower_bound,upper_bound])
            lower_bound = interval[index][0]
            upper_bound = interval[index][1]
        else:
            if interval[index][1] > upper_bound:
                upper_bound = interval[index][1]
    result.append([lower_bound, upper_bound])
    return result


class Model_latency():
    def __init__(self, NetStruct, SimConfig_path):
        modelL_config = cp.ConfigParser()
        modelL_config.read(SimConfig_path, encoding='UTF-8')
        self.inter_bank_bandwidth = float(modelL_config.get('Bank level', 'Inter_Bank_Bandwidth'))
        self.graph = BCG(NetStruct, SimConfig_path)
        self.graph.mapping_net()
        self.graph.calculate_transfer_distance()
        self.begin_time = []
        self.finish_time = []
        self.layer_bank_latency = []
        self.NetStruct = NetStruct
        self.SimConfig_path = SimConfig_path
        self.compute_interval = []
        self.occupancy = []

        self.buffer_latency = []
        self.computing_latency = []
        self.DAC_latency = []
        self.xbar_latency = []
        self.ADC_latency = []
        self.digital_latency = []
        self.intra_bank_latency = []
        self.inter_bank_latency = []
        self.bank_merge_latency = []
        self.bank_transfer_latency = []

        self.total_buffer_latency = []
        self.total_computing_latency = []
        self.total_DAC_latency = []
        self.total_xbar_latency = []
        self.total_ADC_latency = []
        self.total_digital_latency = []
        self.total_intra_bank_latency = []
        self.total_inter_bank_latency = []
        self.total_bank_merge_latency = []
        self.total_bank_transfer_latency = []

    def calculate_model_latency_nopipe(self):
        for layer_id in range(len(self.NetStruct)):
            layer_dict = self.NetStruct[layer_id][0][0]
            if layer_id == 0:
                # for the first layer, first layer must be conv layer
                self.begin_time.append([])
                self.finish_time.append([])
                self.compute_interval.append([])

                self.buffer_latency.append([])
                self.computing_latency.append([])
                self.DAC_latency.append([])
                self.xbar_latency.append([])
                self.ADC_latency.append([])
                self.digital_latency.append([])
                self.intra_bank_latency.append([])
                self.inter_bank_latency.append([])
                self.bank_merge_latency.append([])
                self.bank_transfer_latency.append([])
                output_size = list(map(int, layer_dict['Outputsize']))
                input_size = list(map(int, layer_dict['Inputsize']))
                kernelsize = int(layer_dict['Kernelsize'])
                stride = int(layer_dict['Stride'])
                inputchannel = int(layer_dict['Inputchannel'])
                outputchannel = int(layer_dict['Outputchannel'])
                padding = int(layer_dict['Padding'])
                inputbit = int(layer_dict['Inputbit'])
                outputbit = int(layer_dict['outputbit'])
                # print(self.graph.layer_bankinfo[layer_id]['max_row'])
                input_channel_PE = self.graph.layer_bankinfo[layer_id]['max_row']/(kernelsize**2)
                 # the input channel number each PE processes
                temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                          read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                          read_column=self.graph.layer_bankinfo[layer_id]['max_column'],
                                                          indata=0, rdata=0, inprecision=inputbit,
                                                          PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                          )
                merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                         self.graph.layer_bankinfo[layer_id][
                                                                             'max_column'] * outputbit / self.inter_bank_bandwidth)
                    # Todo: update merge time (adder tree) and transfer data volume
                transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            outputchannel * outputbit / self.inter_bank_bandwidth)
                    # Todo: update transfer data volume
                for i in range(output_size[0]):
                    for j in range(output_size[1]):
                        if (i==0) & (j==0):
                            # the first output
                            indata = input_channel_PE*(input_size[1]*max(kernelsize-padding-1,0)+max(kernelsize-padding,0))*inputbit/8
                                # fill the line buffer
                            rdata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                                # from the line buffer to the input reg
                            temp_bank_latency.update_bank_latency(indata=indata,rdata=rdata)
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time
                            self.begin_time[0].append(0)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([0,compute_time])

                            self.buffer_latency[layer_id].append(temp_bank_latency.buf_wlatency+temp_bank_latency.buf_rlatency)
                            self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                            self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                            self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                            self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                            self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency+
                                                                  temp_bank_latency.ireg_latency+temp_bank_latency.shiftadd_latency+
                                                                  temp_bank_latency.oreg_latency+temp_bank_latency.merge_latency)
                            self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                            self.inter_bank_latency[layer_id].append(merge_time+transfer_time)
                            self.bank_merge_latency[layer_id].append(merge_time)
                            self.bank_transfer_latency[layer_id].append(transfer_time)
                            # print(self.finish_time[0])
                        elif j==0:
                            indata = input_channel_PE*stride*max(kernelsize-padding,0)*inputbit/8
                                # line feed in line buffer
                            rdata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                                # from the line buffer to the input reg
                            temp_bank_latency.update_bank_latency(indata=indata,rdata=rdata)
                            begin_time = self.finish_time[0][(i-1)*output_size[1]+output_size[1]-1]
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time +\
                                           begin_time
                            self.begin_time[0].append(begin_time)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([begin_time,compute_time])

                            self.buffer_latency[layer_id].append(
                                temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                            self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                            self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                            self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                            self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                            self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                  temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                  temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                            self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                            self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                            self.bank_merge_latency[layer_id].append(merge_time)
                            self.bank_transfer_latency[layer_id].append(transfer_time)
                            # print(self.finish_time[0])
                        else:
                            indata = input_channel_PE*stride**2*inputbit/8
                                # write new input data to line buffer
                            rdata = stride*kernelsize*input_channel_PE*inputbit/8
                            temp_bank_latency.update_bank_latency(indata=indata,rdata=rdata)
                            begin_time = self.finish_time[0][i * output_size[1] + j - 1]
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                           begin_time
                            self.begin_time[0].append(begin_time)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([begin_time,compute_time])

                            self.buffer_latency[layer_id].append(
                                temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                            self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                            self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                            self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                            self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                            self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                  temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                  temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                            self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                            self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                            self.bank_merge_latency[layer_id].append(merge_time)
                            self.bank_transfer_latency[layer_id].append(transfer_time)
                # print("start time: ", self.begin_time[0])
                # print("finish time:", self.finish_time[0])
                # print('==============================')
            else:
                if layer_dict['type'] == 'conv':
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])

                    self.buffer_latency.append([])
                    self.computing_latency.append([])
                    self.DAC_latency.append([])
                    self.xbar_latency.append([])
                    self.ADC_latency.append([])
                    self.digital_latency.append([])
                    self.intra_bank_latency.append([])
                    self.inter_bank_latency.append([])
                    self.bank_merge_latency.append([])
                    self.bank_transfer_latency.append([])
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    # print(self.graph.layer_bankinfo[layer_id]['max_row'])
                    input_channel_PE = self.graph.layer_bankinfo[layer_id]['max_row'] / (kernelsize ** 2)
                    # the input channel number each PE processes
                    temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                              read_column=self.graph.layer_bankinfo[layer_id]['max_column'],
                                                              indata=0, rdata=0, inprecision=inputbit,
                                                              PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                              )
                    merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                             self.graph.layer_bankinfo[layer_id]['max_column'] *
                                                                             outputbit / self.inter_bank_bandwidth)
                    # Todo: update merge time (adder tree) and transfer data volume
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            outputchannel * outputbit / self.inter_bank_bandwidth)
                    # Todo: update transfer data volume
                    last_layer_finish_time = self.finish_time[layer_id-1][-1]
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            last_layer_pos = min((kernelsize + stride * i - padding - 1) * input_size[1] + \
                                             kernelsize + stride * j - padding - 1, len(self.finish_time[layer_id-1])-1)
                            if last_layer_pos > len(self.finish_time[layer_id-1])-1:
                                print("pos error", i,j)
                            if (i == 0) & (j == 0):
                                # the first output
                                indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0) + max(
                                    kernelsize - padding, 0))*inputbit/8
                                # fill the line buffer
                                rdata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                                # from the line buffer to the input reg
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = last_layer_finish_time
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                               begin_time
                                # consider the input data generation time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                                self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                                self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                                self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                                self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                      temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                      temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                                self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                                # print(self.finish_time[layer_id])
                            elif j == 0:
                                indata = input_channel_PE * stride * max(kernelsize - padding, 0)*inputbit/8
                                # line feed in line buffer
                                rdata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                                # from the line buffer to the input reg
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id][(i - 1) * output_size[1] + output_size[1] - 1]
                                # max (the required input data generation time, previous point computation complete time)
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                                self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                                self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                                self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                                self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                      temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                      temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                                self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                                # print(self.finish_time[layer_id])
                            else:
                                indata = input_channel_PE * stride ** 2*inputbit/8
                                # write new input data to line buffer
                                rdata = stride * kernelsize * input_channel_PE*inputbit/8
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id][i * output_size[1] + j - 1]
                                # max (the required input data generation time, previous point computation complete time)
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                                self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                                self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                                self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                                self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                      temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                      temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                                self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                    # print("start time: ",self.begin_time[layer_id])
                    # print("finish time:",self.finish_time[layer_id])
                    # print('==============================')
                elif layer_dict['type'] == 'fc':
                    output_size = int(layer_dict['Outfeature'])
                    input_size = int(layer_dict['Infeature'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])

                    self.buffer_latency.append([])
                    self.computing_latency.append([])
                    self.DAC_latency.append([])
                    self.xbar_latency.append([])
                    self.ADC_latency.append([])
                    self.digital_latency.append([])
                    self.intra_bank_latency.append([])
                    self.inter_bank_latency.append([])
                    self.bank_merge_latency.append([])
                    self.bank_transfer_latency.append([])
                    indata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                    rdata = indata*inputbit/8
                    temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                              read_column=self.graph.layer_bankinfo[layer_id]['max_column'],
                                                              indata=indata, rdata=rdata, inprecision=inputbit,
                                                              PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                              )
                    merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                             self.graph.layer_bankinfo[layer_id]['max_column'] *
                                                                             outputbit / self.inter_bank_bandwidth)
                    # Todo: update merge time (adder tree) and transfer data volume
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            output_size * outputbit / self.inter_bank_bandwidth)
                    begin_time = self.finish_time[layer_id-1][-1]
                    compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + begin_time
                    self.begin_time[layer_id] = output_size * [begin_time]
                    self.finish_time[layer_id]= output_size*[compute_time]
                    self.compute_interval[layer_id].append([begin_time, compute_time])

                    self.buffer_latency[layer_id].append(
                        temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                    self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                    self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                    self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                    self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                    self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                          temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                          temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                    self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                    self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                    self.bank_merge_latency[layer_id].append(merge_time)
                    self.bank_transfer_latency[layer_id].append(transfer_time)
                    # print("start time: ",self.begin_time[layer_id])
                    # print("finish time:",self.finish_time[layer_id])
                    # print('==============================')
                else:
                    assert layer_dict['type'] == 'pooling', "Layer type can only be conv/fc/pooling"
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])

                    self.buffer_latency.append([])
                    self.computing_latency.append([])
                    self.DAC_latency.append([])
                    self.xbar_latency.append([])
                    self.ADC_latency.append([])
                    self.digital_latency.append([])
                    self.intra_bank_latency.append([])
                    self.inter_bank_latency.append([])
                    self.bank_merge_latency.append([])
                    self.bank_transfer_latency.append([])
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    temp_pooling_latency = pooling_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                    indata=0, rdata=0)
                    merge_time = 0
                    # Todo: update merge time of pooling bank
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            outputchannel * outputbit / self.inter_bank_bandwidth)
                    # Todo: update transfer data volume
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            if (i == 0) & (j == 0):
                                # the first output
                                indata = inputchannel * (input_size[1] * max(kernelsize - padding - 1, 0) + max(
                                    kernelsize - padding, 0))*inputbit/8
                                # fill the line buffer
                                rdata = inputchannel*kernelsize**2*inputbit/8
                                # from the line buffer to the input reg
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id - 1][-1]
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                # consider the input data generation time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_pooling_latency.buf_wlatency + temp_pooling_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(0)
                                self.DAC_latency[layer_id].append(0)
                                self.xbar_latency[layer_id].append(0)
                                self.ADC_latency[layer_id].append(0)
                                self.digital_latency[layer_id].append(temp_pooling_latency.digital_period)
                                    # TODO: update pooling latency analysis
                                self.intra_bank_latency[layer_id].append(0)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                                # print(self.finish_time[layer_id])
                            elif j == 0:
                                indata = inputchannel * stride * max(kernelsize - padding, 0)*inputbit/8
                                # line feed in line buffer
                                rdata = inputchannel*kernelsize**2*inputbit/8
                                # from the line buffer to the input reg
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id][(i - 1) * output_size[1] + output_size[1] - 1]
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_pooling_latency.buf_wlatency + temp_pooling_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(0)
                                self.DAC_latency[layer_id].append(0)
                                self.xbar_latency[layer_id].append(0)
                                self.ADC_latency[layer_id].append(0)
                                self.digital_latency[layer_id].append(temp_pooling_latency.digital_period)
                                # TODO: update pooling latency analysis
                                self.intra_bank_latency[layer_id].append(0)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                                # print(self.finish_time[layer_id])
                            else:
                                indata = inputchannel * stride ** 2*inputbit/8
                                # write new input data to line buffer
                                rdata = stride * kernelsize * inputchannel*inputbit/8
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id][i * output_size[1] + j - 1]
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_pooling_latency.buf_wlatency + temp_pooling_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(0)
                                self.DAC_latency[layer_id].append(0)
                                self.xbar_latency[layer_id].append(0)
                                self.ADC_latency[layer_id].append(0)
                                self.digital_latency[layer_id].append(temp_pooling_latency.digital_period)
                                # TODO: update pooling latency analysis
                                self.intra_bank_latency[layer_id].append(0)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                    # print("start time: ",self.begin_time[layer_id])
                    # print("finish time:",self.finish_time[layer_id])
                    # print('==============================')
            self.compute_interval[layer_id] = merge_interval(self.compute_interval[layer_id])
            temp_runtime = 0
            for l in range(len(self.compute_interval[layer_id])):
                temp_runtime += (self.compute_interval[layer_id][l][1]-self.compute_interval[layer_id][l][0])
            self.occupancy.append(temp_runtime/(max(self.finish_time[layer_id])-min(self.begin_time[layer_id])))
            self.total_buffer_latency.append(sum(self.buffer_latency[layer_id]))
            self.total_computing_latency.append(sum(self.computing_latency[layer_id]))
            self.total_DAC_latency.append(sum(self.DAC_latency[layer_id]))
            self.total_xbar_latency.append(sum(self.xbar_latency[layer_id]))
            self.total_ADC_latency.append(sum(self.ADC_latency[layer_id]))
            self.total_digital_latency.append(sum(self.digital_latency[layer_id]))
            self.total_inter_bank_latency.append(sum(self.inter_bank_latency[layer_id]))
            self.total_intra_bank_latency.append(sum(self.intra_bank_latency[layer_id]))
            self.total_bank_merge_latency.append(sum(self.bank_merge_latency[layer_id]))
            self.total_bank_transfer_latency.append(sum(self.bank_transfer_latency[layer_id]))

    def calculate_model_latency_0(self):
        for layer_id in range(len(self.NetStruct)):
            layer_dict = self.NetStruct[layer_id][0][0]
            if layer_id == 0:
                # for the first layer, first layer must be conv layer
                self.begin_time.append([])
                self.finish_time.append([])
                self.compute_interval.append([])
                output_size = list(map(int, layer_dict['Outputsize']))
                input_size = list(map(int, layer_dict['Inputsize']))
                kernelsize = int(layer_dict['Kernelsize'])
                stride = int(layer_dict['Stride'])
                inputchannel = int(layer_dict['Inputchannel'])
                outputchannel = int(layer_dict['Outputchannel'])
                padding = int(layer_dict['Padding'])
                inputbit = int(layer_dict['Inputbit'])
                outputbit = int(layer_dict['outputbit'])
                # print(self.graph.layer_bankinfo[layer_id]['max_row'])
                input_channel_PE = self.graph.layer_bankinfo[layer_id]['max_row'] / (kernelsize ** 2)
                # the input channel number each PE processes
                temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                          read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                          read_column=self.graph.layer_bankinfo[layer_id]['max_column'],
                                                          indata=0, rdata=0, inprecision=inputbit,
                                                          PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                          )
                merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                         self.graph.layer_bankinfo[layer_id][
                                                                             'max_column'] * outputbit / self.inter_bank_bandwidth)
                # Todo: update merge time (adder tree) and transfer data volume
                transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                        outputchannel * outputbit / self.inter_bank_bandwidth)
                # Todo: update transfer data volume
                for i in range(output_size[0]):
                    for j in range(output_size[1]):
                        if (i == 0) & (j == 0):
                            # the first output
                            # indata: the data needed to written into buffer
                            indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0) +
                                                         max(kernelsize - padding, 0)) * inputbit / 8
                            # rdata: the data read from the buffer
                            rdata = self.graph.layer_bankinfo[layer_id]['max_row'] * inputbit / 8
                            temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time
                            self.begin_time[0].append(0)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([0, compute_time])
                        elif j == 0:
                            # TODO: check the changes
                            indata = inputbit / 8 * input_channel_PE * (input_size[1] * (stride - 1) +
                                                                        max(kernelsize - padding, 0))
                            rdata = self.graph.layer_bankinfo[layer_id]['max_row'] * inputbit / 8
                            temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                            begin_time = self.finish_time[0][(i - 1) * output_size[1] + output_size[1] - 1]
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                           begin_time
                            # TODO: check

                            self.begin_time[0].append(begin_time)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([begin_time, compute_time])
                        else:
                            indata = input_channel_PE * stride * inputbit / 8
                            rdata = stride * kernelsize * input_channel_PE * inputbit / 8
                            temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                            begin_time = self.finish_time[0][i * output_size[1] + j - 1]
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                           begin_time

                            self.begin_time[0].append(begin_time)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([begin_time, compute_time])
            else:
                if layer_dict['type'] == 'conv':
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    # print(self.graph.layer_bankinfo[layer_id]['max_row'])
                    input_channel_PE = self.graph.layer_bankinfo[layer_id]['max_row'] / (kernelsize ** 2)
                    # the input channel number each PE processes
                    temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                              read_column=self.graph.layer_bankinfo[layer_id][
                                                                  'max_column'],
                                                              indata=0, rdata=0, inprecision=inputbit,
                                                              PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                              )
                    merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                             self.graph.layer_bankinfo[layer_id][
                                                                                 'max_column'] *
                                                                             outputbit / self.inter_bank_bandwidth)
                    # Todo: update merge time (adder tree) and transfer data volume
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            outputchannel * outputbit / self.inter_bank_bandwidth)
                    # Todo: update transfer data volume
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            last_layer_pos = min((kernelsize + stride * i - padding - 1) * input_size[1] +
                                                 kernelsize + stride * j - padding - 1,
                                                 len(self.finish_time[layer_id - 1]) - 1)

                            if (i == 0) & (j == 0):
                                # the first output
                                # indata: the data needed to written into buffer
                                indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0) +
                                                             max(kernelsize - padding, 0)) * inputbit / 8
                                # rdata: the data read from the buffer
                                rdata = self.graph.layer_bankinfo[layer_id]['max_row'] * inputbit / 8
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id - 1][last_layer_pos]
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + begin_time

                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                            elif j == 0:
                                indata = inputbit / 8 * input_channel_PE * (input_size[1] * (stride - 1) +
                                                                            max(kernelsize - padding, 0))
                                rdata = self.graph.layer_bankinfo[layer_id]['max_row'] * inputbit / 8
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = max(self.finish_time[layer_id - 1][last_layer_pos],
                                                 self.finish_time[layer_id][
                                                     (i - 1) * output_size[1] + output_size[1] - 1])
                                # max(the required input data generation time, previous point computation complete time)
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + begin_time
                                # consider the input data generation time

                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])
                                # print(self.finish_time[layer_id])
                            else:
                                indata = input_channel_PE * stride * inputbit / 8
                                rdata = stride * kernelsize * input_channel_PE * inputbit / 8
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = max(self.finish_time[layer_id - 1][last_layer_pos],
                                                 self.finish_time[layer_id][i * output_size[1] + j - 1])
                                # max (the required input data generation time, previous point computation complete time)
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                               begin_time

                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                elif layer_dict['type'] == 'fc':
                    output_size = int(layer_dict['Outfeature'])
                    input_size = int(layer_dict['Infeature'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])
                    indata = self.graph.layer_bankinfo[layer_id]['max_row'] * inputbit / 8
                    rdata = indata * inputbit / 8
                    temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                              read_column=self.graph.layer_bankinfo[layer_id][
                                                                  'max_column'],
                                                              indata=indata, rdata=rdata, inprecision=inputbit,
                                                              PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                              )
                    merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                             self.graph.layer_bankinfo[layer_id][
                                                                                 'max_column'] *
                                                                             outputbit / self.inter_bank_bandwidth)
                    # Todo: update merge time (adder tree) and transfer data volume
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            output_size * outputbit / self.inter_bank_bandwidth)
                    begin_time = self.finish_time[layer_id - 1][-1]
                    compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + begin_time

                    self.begin_time[layer_id] = output_size * [begin_time]
                    self.finish_time[layer_id] = output_size * [compute_time]
                    self.compute_interval[layer_id].append([begin_time, compute_time])

                else:
                    assert layer_dict['type'] == 'pooling', "Layer type can only be conv/fc/pooling"
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    temp_pooling_latency = pooling_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                    indata=0, rdata=0)
                    merge_time = 0
                    # Todo: update merge time of pooling bank
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            outputchannel * outputbit / self.inter_bank_bandwidth)
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            last_layer_pos = min((kernelsize + stride * i - padding - 1) * input_size[1] +
                                                 kernelsize + stride * j - padding - 1,
                                                 len(self.finish_time[layer_id - 1]) - 1)
                            if (i == 0) & (j == 0):
                                # the first output
                                indata = inputchannel * (input_size[1] * max(kernelsize - padding - 1, 0) + max(
                                    kernelsize - padding, 0)) * inputbit / 8
                                # fill the line buffer
                                rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                # from the line buffer to the input reg
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id - 1][last_layer_pos]
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                # consider the input data generation time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])
                                # print(self.finish_time[layer_id])
                            elif j == 0:
                                indata = inputbit / 8 * inputchannel * (input_size[1] * (stride - 1) +
                                                                        max(kernelsize - padding, 0))
                                rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                # from the line buffer to the input reg
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = max(self.finish_time[layer_id - 1][last_layer_pos],
                                                 self.finish_time[layer_id][
                                                     (i - 1) * output_size[1] + output_size[1] - 1])
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])
                                # print(self.finish_time[layer_id])
                            else:
                                indata = inputchannel * stride ** 2 * inputbit / 8
                                # write new input data to line buffer
                                rdata = stride * kernelsize * inputchannel * inputbit / 8
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = max(self.finish_time[layer_id - 1][last_layer_pos],
                                                 self.finish_time[layer_id][i * output_size[1] + j - 1])
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])
            self.compute_interval[layer_id] = merge_interval(self.compute_interval[layer_id])
            temp_runtime = 0
            for l in range(len(self.compute_interval[layer_id])):
                temp_runtime += (self.compute_interval[layer_id][l][1] - self.compute_interval[layer_id][l][0])
            self.occupancy.append(temp_runtime / (max(self.finish_time[layer_id]) - min(self.begin_time[layer_id])))

    def calculate_model_latency_1(self):
        for layer_id in range(len(self.NetStruct)):
            layer_dict = self.NetStruct[layer_id][0][0]
            if layer_id == 0:
                # for the first layer, first layer must be conv layer
                self.begin_time.append([])
                self.finish_time.append([])
                self.compute_interval.append([])

                self.buffer_latency.append([])
                self.computing_latency.append([])
                self.DAC_latency.append([])
                self.xbar_latency.append([])
                self.ADC_latency.append([])
                self.digital_latency.append([])
                self.intra_bank_latency.append([])
                self.inter_bank_latency.append([])
                self.bank_merge_latency.append([])
                self.bank_transfer_latency.append([])
                output_size = list(map(int, layer_dict['Outputsize']))
                input_size = list(map(int, layer_dict['Inputsize']))
                kernelsize = int(layer_dict['Kernelsize'])
                stride = int(layer_dict['Stride'])
                inputchannel = int(layer_dict['Inputchannel'])
                outputchannel = int(layer_dict['Outputchannel'])
                padding = int(layer_dict['Padding'])
                inputbit = int(layer_dict['Inputbit'])
                outputbit = int(layer_dict['outputbit'])
                # print(self.graph.layer_bankinfo[layer_id]['max_row'])
                input_channel_PE = self.graph.layer_bankinfo[layer_id]['max_row']/(kernelsize**2)
                 # the input channel number each PE processes
                temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                          read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                          read_column=self.graph.layer_bankinfo[layer_id]['max_column'],
                                                          indata=0, rdata=0, inprecision=inputbit,
                                                          PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                          )
                merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                         self.graph.layer_bankinfo[layer_id][
                                                                             'max_column'] * outputbit / self.inter_bank_bandwidth)
                    # Todo: update merge time (adder tree) and transfer data volume
                transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            outputchannel * outputbit / self.inter_bank_bandwidth)
                    # Todo: update transfer data volume
                for i in range(output_size[0]):
                    for j in range(output_size[1]):
                        if (i==0) & (j==0):
                            # the first output
                            indata = input_channel_PE*(input_size[1]*max(kernelsize-padding-1,0)+max(kernelsize-padding,0))*inputbit/8
                                # fill the line buffer
                            rdata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                                # from the line buffer to the input reg
                            temp_bank_latency.update_bank_latency(indata=indata,rdata=rdata)
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time
                            self.begin_time[0].append(0)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([0,compute_time])

                            self.buffer_latency[layer_id].append(temp_bank_latency.buf_wlatency+temp_bank_latency.buf_rlatency)
                            self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                            self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                            self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                            self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                            self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency+
                                                                  temp_bank_latency.ireg_latency+temp_bank_latency.shiftadd_latency+
                                                                  temp_bank_latency.oreg_latency+temp_bank_latency.merge_latency)
                            self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                            self.inter_bank_latency[layer_id].append(merge_time+transfer_time)
                            self.bank_merge_latency[layer_id].append(merge_time)
                            self.bank_transfer_latency[layer_id].append(transfer_time)
                            # print(self.finish_time[0])
                        elif j==0:
                            indata = input_channel_PE*stride*max(kernelsize-padding,0)*inputbit/8
                                # line feed in line buffer
                            rdata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                                # from the line buffer to the input reg
                            temp_bank_latency.update_bank_latency(indata=indata,rdata=rdata)
                            begin_time = self.finish_time[0][(i-1)*output_size[1]+output_size[1]-1]
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time +\
                                           begin_time
                            self.begin_time[0].append(begin_time)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([begin_time,compute_time])

                            self.buffer_latency[layer_id].append(
                                temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                            self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                            self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                            self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                            self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                            self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                  temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                  temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                            self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                            self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                            self.bank_merge_latency[layer_id].append(merge_time)
                            self.bank_transfer_latency[layer_id].append(transfer_time)
                            # print(self.finish_time[0])
                        else:
                            indata = input_channel_PE*stride**2*inputbit/8
                                # write new input data to line buffer
                            rdata = stride*kernelsize*input_channel_PE*inputbit/8
                            temp_bank_latency.update_bank_latency(indata=indata,rdata=rdata)
                            begin_time = self.finish_time[0][i * output_size[1] + j - 1]
                            compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                           begin_time
                            self.begin_time[0].append(begin_time)
                            self.finish_time[0].append(compute_time)
                            self.compute_interval[0].append([begin_time,compute_time])

                            self.buffer_latency[layer_id].append(
                                temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                            self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                            self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                            self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                            self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                            self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                  temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                  temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                            self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                            self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                            self.bank_merge_latency[layer_id].append(merge_time)
                            self.bank_transfer_latency[layer_id].append(transfer_time)
                # print("start time: ", self.begin_time[0])
                # print("finish time:", self.finish_time[0])
                # print('==============================')
            else:
                if layer_dict['type'] == 'conv':
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])

                    self.buffer_latency.append([])
                    self.computing_latency.append([])
                    self.DAC_latency.append([])
                    self.xbar_latency.append([])
                    self.ADC_latency.append([])
                    self.digital_latency.append([])
                    self.intra_bank_latency.append([])
                    self.inter_bank_latency.append([])
                    self.bank_merge_latency.append([])
                    self.bank_transfer_latency.append([])
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    # print(self.graph.layer_bankinfo[layer_id]['max_row'])
                    input_channel_PE = self.graph.layer_bankinfo[layer_id]['max_row'] / (kernelsize ** 2)
                    # the input channel number each PE processes
                    temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                              read_column=self.graph.layer_bankinfo[layer_id]['max_column'],
                                                              indata=0, rdata=0, inprecision=inputbit,
                                                              PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                              )
                    merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                             self.graph.layer_bankinfo[layer_id]['max_column'] *
                                                                             outputbit / self.inter_bank_bandwidth)
                    # Todo: update merge time (adder tree) and transfer data volume
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            outputchannel * outputbit / self.inter_bank_bandwidth)
                    # Todo: update transfer data volume
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            last_layer_pos = min((kernelsize + stride * i - padding - 1) * input_size[1] + \
                                             kernelsize + stride * j - padding - 1, len(self.finish_time[layer_id-1])-1)
                            if last_layer_pos > len(self.finish_time[layer_id-1])-1:
                                print("pos error", i,j)
                            if (i == 0) & (j == 0):
                                # the first output
                                indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0) + max(
                                    kernelsize - padding, 0))*inputbit/8
                                # fill the line buffer
                                rdata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                                # from the line buffer to the input reg
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id-1][last_layer_pos]
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                               begin_time
                                # consider the input data generation time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                                self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                                self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                                self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                                self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                      temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                      temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                                self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                                # print(self.finish_time[layer_id])
                            elif j == 0:
                                indata = input_channel_PE * stride * max(kernelsize - padding, 0)*inputbit/8
                                # line feed in line buffer
                                rdata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                                # from the line buffer to the input reg
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = max(self.finish_time[layer_id-1][last_layer_pos],
                                                   self.finish_time[layer_id][(i - 1) * output_size[1] + output_size[1] - 1])
                                # max (the required input data generation time, previous point computation complete time)
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                                self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                                self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                                self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                                self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                      temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                      temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                                self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                                # print(self.finish_time[layer_id])
                            else:
                                indata = input_channel_PE * stride ** 2*inputbit/8
                                # write new input data to line buffer
                                rdata = stride * kernelsize * input_channel_PE*inputbit/8
                                temp_bank_latency.update_bank_latency(indata=indata, rdata=rdata)
                                begin_time = max(self.finish_time[layer_id-1][last_layer_pos],
                                                   self.finish_time[layer_id][i * output_size[1] + j - 1])
                                # max (the required input data generation time, previous point computation complete time)
                                compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                                self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                                self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                                self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                                self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                                      temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                                      temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                                self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                    # print("start time: ",self.begin_time[layer_id])
                    # print("finish time:",self.finish_time[layer_id])
                    # print('==============================')
                elif layer_dict['type'] == 'fc':
                    output_size = int(layer_dict['Outfeature'])
                    input_size = int(layer_dict['Infeature'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])

                    self.buffer_latency.append([])
                    self.computing_latency.append([])
                    self.DAC_latency.append([])
                    self.xbar_latency.append([])
                    self.ADC_latency.append([])
                    self.digital_latency.append([])
                    self.intra_bank_latency.append([])
                    self.inter_bank_latency.append([])
                    self.bank_merge_latency.append([])
                    self.bank_transfer_latency.append([])
                    indata = self.graph.layer_bankinfo[layer_id]['max_row']*inputbit/8
                    rdata = indata*inputbit/8
                    temp_bank_latency = bank_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              read_row=self.graph.layer_bankinfo[layer_id]['max_row'],
                                                              read_column=self.graph.layer_bankinfo[layer_id]['max_column'],
                                                              indata=indata, rdata=rdata, inprecision=inputbit,
                                                              PE_num=self.graph.layer_bankinfo[layer_id]['max_PE']
                                                              )
                    merge_time = self.graph.inLayer_distance[0][layer_id] * (temp_bank_latency.digital_period +
                                                                             self.graph.layer_bankinfo[layer_id]['max_column'] *
                                                                             outputbit / self.inter_bank_bandwidth)
                    # Todo: update merge time (adder tree) and transfer data volume
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            output_size * outputbit / self.inter_bank_bandwidth)
                    begin_time = self.finish_time[layer_id-1][-1]
                    compute_time = temp_bank_latency.bank_latency + merge_time + transfer_time + begin_time
                    self.begin_time[layer_id] = output_size * [begin_time]
                    self.finish_time[layer_id]= output_size*[compute_time]
                    self.compute_interval[layer_id].append([begin_time, compute_time])

                    self.buffer_latency[layer_id].append(
                        temp_bank_latency.buf_wlatency + temp_bank_latency.buf_rlatency)
                    self.computing_latency[layer_id].append(temp_bank_latency.computing_latency)
                    self.DAC_latency[layer_id].append(temp_bank_latency.DAC_latency)
                    self.xbar_latency[layer_id].append(temp_bank_latency.xbar_latency)
                    self.ADC_latency[layer_id].append(temp_bank_latency.ADC_latency)
                    self.digital_latency[layer_id].append(temp_bank_latency.inPE_add_latency +
                                                          temp_bank_latency.ireg_latency + temp_bank_latency.shiftadd_latency +
                                                          temp_bank_latency.oreg_latency + temp_bank_latency.merge_latency)
                    self.intra_bank_latency[layer_id].append(temp_bank_latency.transfer_latency)
                    self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                    self.bank_merge_latency[layer_id].append(merge_time)
                    self.bank_transfer_latency[layer_id].append(transfer_time)
                    # print("start time: ",self.begin_time[layer_id])
                    # print("finish time:",self.finish_time[layer_id])
                    # print('==============================')
                else:
                    assert layer_dict['type'] == 'pooling', "Layer type can only be conv/fc/pooling"
                    self.begin_time.append([])
                    self.finish_time.append([])
                    self.compute_interval.append([])

                    self.buffer_latency.append([])
                    self.computing_latency.append([])
                    self.DAC_latency.append([])
                    self.xbar_latency.append([])
                    self.ADC_latency.append([])
                    self.digital_latency.append([])
                    self.intra_bank_latency.append([])
                    self.inter_bank_latency.append([])
                    self.bank_merge_latency.append([])
                    self.bank_transfer_latency.append([])
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    temp_pooling_latency = pooling_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                    indata=0, rdata=0)
                    merge_time = 0
                    # Todo: update merge time of pooling bank
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                            outputchannel * outputbit / self.inter_bank_bandwidth)
                    # Todo: update transfer data volume
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            last_layer_pos = min((kernelsize + stride * i - padding - 1) * input_size[1] + \
                                                 kernelsize + stride * j - padding - 1,
                                                 len(self.finish_time[layer_id - 1]) - 1)
                            if (i == 0) & (j == 0):
                                # the first output
                                indata = inputchannel * (input_size[1] * max(kernelsize - padding - 1, 0) + max(
                                    kernelsize - padding, 0))*inputbit/8
                                # fill the line buffer
                                rdata = inputchannel*kernelsize**2*inputbit/8
                                # from the line buffer to the input reg
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id - 1][last_layer_pos]
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                # consider the input data generation time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_pooling_latency.buf_wlatency + temp_pooling_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(0)
                                self.DAC_latency[layer_id].append(0)
                                self.xbar_latency[layer_id].append(0)
                                self.ADC_latency[layer_id].append(0)
                                self.digital_latency[layer_id].append(temp_pooling_latency.digital_period)
                                    # TODO: update pooling latency analysis
                                self.intra_bank_latency[layer_id].append(0)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                                # print(self.finish_time[layer_id])
                            elif j == 0:
                                indata = inputchannel * stride * max(kernelsize - padding, 0)*inputbit/8
                                # line feed in line buffer
                                rdata = inputchannel*kernelsize**2*inputbit/8
                                # from the line buffer to the input reg
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = max(self.finish_time[layer_id - 1][last_layer_pos],
                                                   self.finish_time[layer_id][(i - 1) * output_size[1] + output_size[1] - 1])
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_pooling_latency.buf_wlatency + temp_pooling_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(0)
                                self.DAC_latency[layer_id].append(0)
                                self.xbar_latency[layer_id].append(0)
                                self.ADC_latency[layer_id].append(0)
                                self.digital_latency[layer_id].append(temp_pooling_latency.digital_period)
                                # TODO: update pooling latency analysis
                                self.intra_bank_latency[layer_id].append(0)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                                # print(self.finish_time[layer_id])
                            else:
                                indata = inputchannel * stride ** 2*inputbit/8
                                # write new input data to line buffer
                                rdata = stride * kernelsize * inputchannel*inputbit/8
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = max(self.finish_time[layer_id - 1][last_layer_pos],
                                                   self.finish_time[layer_id][i * output_size[1] + j - 1])
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + \
                                               begin_time
                                self.begin_time[layer_id].append(begin_time)
                                self.finish_time[layer_id].append(compute_time)
                                self.compute_interval[layer_id].append([begin_time, compute_time])

                                self.buffer_latency[layer_id].append(
                                    temp_pooling_latency.buf_wlatency + temp_pooling_latency.buf_rlatency)
                                self.computing_latency[layer_id].append(0)
                                self.DAC_latency[layer_id].append(0)
                                self.xbar_latency[layer_id].append(0)
                                self.ADC_latency[layer_id].append(0)
                                self.digital_latency[layer_id].append(temp_pooling_latency.digital_period)
                                # TODO: update pooling latency analysis
                                self.intra_bank_latency[layer_id].append(0)
                                self.inter_bank_latency[layer_id].append(merge_time + transfer_time)
                                self.bank_merge_latency[layer_id].append(merge_time)
                                self.bank_transfer_latency[layer_id].append(transfer_time)
                    # print("start time: ",self.begin_time[layer_id])
                    # print("finish time:",self.finish_time[layer_id])
                    # print('==============================')
            self.compute_interval[layer_id] = merge_interval(self.compute_interval[layer_id])
            temp_runtime = 0
            for l in range(len(self.compute_interval[layer_id])):
                temp_runtime += (self.compute_interval[layer_id][l][1]-self.compute_interval[layer_id][l][0])
            self.occupancy.append(temp_runtime/(max(self.finish_time[layer_id])-min(self.begin_time[layer_id])))
            self.total_buffer_latency.append(sum(self.buffer_latency[layer_id]))
            self.total_computing_latency.append(sum(self.computing_latency[layer_id]))
            self.total_DAC_latency.append(sum(self.DAC_latency[layer_id]))
            self.total_xbar_latency.append(sum(self.xbar_latency[layer_id]))
            self.total_ADC_latency.append(sum(self.ADC_latency[layer_id]))
            self.total_digital_latency.append(sum(self.digital_latency[layer_id]))
            self.total_inter_bank_latency.append(sum(self.inter_bank_latency[layer_id]))
            self.total_intra_bank_latency.append(sum(self.intra_bank_latency[layer_id]))
            self.total_bank_merge_latency.append(sum(self.bank_merge_latency[layer_id]))
            self.total_bank_transfer_latency.append(sum(self.bank_transfer_latency[layer_id]))


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "cifar10_vgg16_params.pth")

    __TestInterface = TrainTestInterface('vgg16', 'MNSIM.Interface.cifar10', test_SimConfig_path, test_weights_file_path,
                                         'cpu')
    structure_file = __TestInterface.get_structure()

    test = Model_latency(structure_file, test_SimConfig_path)
    # test.caculate_model_latency_1()
    test.calculate_model_latency_1()
    for i in range(len(test.begin_time)):
        print("Layer",i," type:", test.NetStruct[i][0][0]['type'])
        print("start time: ", test.begin_time[i])
        print("finish time:",test.finish_time[i])
        print("used time:", test.compute_interval[i])
        print("Occupancy:", test.occupancy[i])
        # print(test.xbar_latency[i])
        total_latency = test.total_buffer_latency[i]+test.total_computing_latency[i]+\
                        test.total_digital_latency[i]+test.total_intra_bank_latency[i]+test.total_inter_bank_latency[i]
        print("Buffer latency of layer",i,":", test.total_buffer_latency[i],'(', "%.2f" %(100*test.total_buffer_latency[i]/total_latency),'%)')
        print("Computing latency of layer", i, ":", test.total_computing_latency[i],'(', "%.2f" %(100*test.total_computing_latency[i]/total_latency),'%)')
        print("     DAC latency of layer", i, ":", test.total_DAC_latency[i],'(', "%.2f" %(100*test.total_DAC_latency[i]/total_latency),'%)')
        print("     ADC latency of layer", i, ":", test.total_ADC_latency[i],'(', "%.2f" %(100*test.total_ADC_latency[i]/total_latency),'%)')
        print("     xbar latency of layer", i, ":", test.total_xbar_latency[i],'(', "%.2f" %(100*test.total_xbar_latency[i]/total_latency),'%)')
        print("Digital part latency of layer", i, ":", test.total_digital_latency[i],'(', "%.2f" %(100*test.total_digital_latency[i]/total_latency),'%)')
        print("Intra bank communication latency of layer", i, ":", test.total_intra_bank_latency[i],'(', "%.2f" %(100*test.total_intra_bank_latency[i]/total_latency),'%)')
        print("Inter bank communication latency of layer", i, ":", test.total_inter_bank_latency[i],'(', "%.2f" %(100*test.total_inter_bank_latency[i]/total_latency),'%)')
        print("     One layer merge latency of layer", i, ":", test.total_bank_merge_latency[i],'(', "%.2f" %(100*test.total_bank_merge_latency[i]/total_latency),'%)')
        print("     Inter bank transfer latency of layer", i, ":", test.total_bank_transfer_latency[i],'(', "%.2f" %(100*test.total_bank_transfer_latency[i]/total_latency),'%)')
        print('==============================')
    print("Latency simulation finished!")
    print("Entire latency:", max(max(test.finish_time)))