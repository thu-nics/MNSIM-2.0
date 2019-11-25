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
from MNSIM.Hardware_Model.Bank import bank
from MNSIM.Interface.interface import *
import collections

class PE_node():
    def __init__(self, PE_id = 0, ltype='conv', lnum = 0):
        # PE_id: the id of P                                E node, ltype: layer type of this PE, lnum: layer number of this PE
        self.id = PE_id
        self.type = ltype
        self.lnum = lnum
        self.inMerge_list = []
        self.outMerge = 0
    def set_inMerge(self, Merge_id):
        if Merge_id not in self.inMerge_list:
            self.inMerge_list.append(Merge_id)
            self.inMerge_list.sort()
    def set_outMerge(self, Merge_id):
        self.outMerge = Merge_id

class Merge_node():
    def __init__(self, Merge_id = 0, mtype = 0, lnum = 0):
        # Merge_id: the id of Merge node, mtype: merge type (0: add, 1: concat, 2: pooling)
        self.id = Merge_id
        self.type = mtype
        self.lnum = lnum
        self.inPE_list = []
        self.outPE_list = []
        self.inMerge_list = []
        self.outMerge_list = []
    def set_inPE(self, PE_id):
        if PE_id not in self.inPE_list:
            self.inPE_list.append(PE_id)
            self.inPE_list.sort()
    def set_outPE(self, PE_id):
        if PE_id not in self.outPE_list:
            self.outPE_list.append(PE_id)
            self.outPE_list.sort()
    def set_inMerge(self, Merge_id):
        if Merge_id not in self.inMerge_list:
            self.inMerge_list.append(Merge_id)
            self.inMerge_list.sort()
    def set_outMerge(self, Merge_id):
        if Merge_id not in self.outMerge_list:
            self.outMerge_list.append(Merge_id)
            self.outMerge_list.sort()

def generate_normal_matrix(row, column):
    matrix = np.zeros([row,column])
    start = 0
    for i in range(row):
        for j in range(column):
            matrix[i][j] = start
            start += 1
    return matrix

def generate_snake_matrix (row, column):
    matrix = np.zeros([row,column])
    start = 0
    for i in range(row):
        for j in range(column):
            if i % 2:
                matrix[i][column - j - 1] = start
            else:
                matrix[i][j] = start
            start += 1
    return matrix

def generate_hui_matrix (row, column):
    matrix = np.zeros([row,column])
    state = 0
    stride = 1
    step = 0
    start = 0
    dl = 0
    ru = 0
    i = 0
    j = 0
    for x in range(row*column):
        if x == 0:
            matrix[i][j] = start
        else:
            if state == 0:
                j += 1
                matrix[i][j] = start
                state = 1
            elif state == 1:
                if dl == 0:
                    i += 1
                    matrix[i][j] = start
                    step += 1
                    if step == stride:
                        dl = 1
                        step = 0
                elif dl == 1:
                    j -= 1
                    matrix[i][j] = start
                    step += 1
                    if step == stride:
                        dl = 0
                        step = 0
                        stride += 1
                        state = 2
            elif state == 2:
                i += 1
                matrix[i][j] = start
                state = 3
            elif state == 3:
                if ru == 0:
                    j += 1
                    matrix[i][j] = start
                    step += 1
                    if step == stride:
                        ru = 1
                        step = 0
                elif ru == 1:
                    i -= 1
                    matrix[i][j] = start
                    step += 1
                    if step == stride:
                        ru = 0
                        step = 0
                        stride += 1
                        state = 0
        start += 1
    return matrix

def generate_zigzag_matrix(row, column):
    matrix = np.zeros([row,column])
    state = 0
    stride = 1
    step = 0
    i = 0
    j = 0
    start = 0
    for x in range(row*column):
        if x == 0:
            matrix[i][j] = start
        else:
            if state == 0:
                if j < column-1:
                    j += 1
                    matrix[i][j] = start
                else:
                    i += 1
                    matrix[i][j] = start
                state = 1
            elif state == 1:
                i += 1
                j -= 1
                matrix[i][j] = start
                step += 1
                if i == row-1:
                    state = 2
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 2
                    stride += 1
                    step = 0
            elif state == 2:
                if i < row-1:
                    i += 1
                    matrix[i][j] = start
                else:
                    j += 1
                    matrix[i][j] = start
                state = 3
            elif state ==3:
                j += 1
                i -= 1
                matrix[i][j] = start
                step += 1
                if j == column-1:
                    state = 0
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 0
                    stride += 1
                    step = 0
        start += 1
    return matrix

class BCG():
    def __init__(self, NetStruct, SimConfig_path, multiple=None):
        BCG_config = cp.ConfigParser()
        BCG_config.read(SimConfig_path, encoding='UTF-8')
        if multiple is None:
            multiple = [1] * len(NetStruct)
        self.bank = bank(SimConfig_path)
        self.net = NetStruct
        self.layer_num = len(self.net)
        self.layer_bankinfo = []
        self.xbar_polarity = int(BCG_config.get('Process element level', 'Xbar_Polarity'))
        self.bank_connection = int(BCG_config.get('Architecture level', 'Bank_Connection'))
        self.bank_num = list(map(int, BCG_config.get('Architecture level', 'Bank_Num').split(',')))
        if self.bank_num[0] == 0:
            self.bank_num[0] = 8
            self.bank_num[1] = 8
        assert self.bank_num[0] > 0, "Bank number < 0"
        assert self.bank_num[1] > 0, "Bank number < 0"
        self.bank_total_num = self.bank_num[0] * self.bank_num[1]
        self.mapping_order = -1*np.ones(self.bank_num)
        self.mapping_result = -1*np.ones(self.bank_num)
        start_bankid = 0
            # the start PEid
        self.trans_time = np.ones([1, self.layer_num])
        for layer_id in range(self.layer_num):
            layer_dict = self.net[layer_id][0][0]
            tmp_bankinfo = collections.OrderedDict()
            layer_type = layer_dict['type']
            if self.xbar_polarity == 1:
                weight_precision = int(layer_dict['Weightbit'])
            else:
                assert self.xbar_polarity == 2, "Crossbar polarity must be 1 or 2"
                weight_precision = int(layer_dict['Weightbit']) - 1
            tmp_bankinfo['startid'] = start_bankid
            if layer_type == 'conv':
                tmp_bankinfo['type'] = 'conv'
                tmp_bankinfo['mx'] = math.ceil(weight_precision/self.bank.group_num) * \
                                   math.ceil(int(layer_dict['Outputchannel'])/self.bank.xbar_column)
                    # mx: PE number in x-axis
                tmp_bankinfo['my'] = math.ceil(int(layer_dict['Inputchannel'])/
                                             (self.bank.xbar_row // (int(layer_dict['Kernelsize'])**2)))
                    # my: PE number in y-axis
                tmp_bankinfo['max_row'] = min((self.bank.xbar_row // (int(layer_dict['Kernelsize'])**2)),
                                              int(layer_dict['Inputchannel'])) * (int(layer_dict['Kernelsize'])**2)
                    # max_row: maximum used row in one crossbar of this layer
                tmp_bankinfo['max_column'] = min(int(layer_dict['Outputchannel']), self.bank.xbar_column)
                    # max_row: maximum used column in one crossbar of this layer
            elif layer_type == 'fc':
                tmp_bankinfo['type'] = 'fc'
                tmp_bankinfo['mx'] = math.ceil(weight_precision/self.bank.group_num) * \
                                   math.ceil(int(layer_dict['Outfeature'])/self.bank.xbar_column)
                    # mx: PE number in x-axis
                tmp_bankinfo['my'] = math.ceil(int(layer_dict['Infeature'])/self.bank.xbar_row)
                    # my: PE number in y-axis
                tmp_bankinfo['max_row'] = min(int(layer_dict['Infeature']), self.bank.xbar_row)
                # max_row: maximum used row in one crossbar of this layer
                tmp_bankinfo['max_column'] = min(int(layer_dict['Outfeature']), self.bank.xbar_column)
                # max_row: maximum used column in one crossbar of this layer
            else:
                tmp_bankinfo['type'] = 'pooling'
                tmp_bankinfo['mx'] = 1
                tmp_bankinfo['my'] = 1
                tmp_bankinfo['max_row'] = 0
                tmp_bankinfo['max_column'] = 0
            if layer_id < self.layer_num-1:
                next_layer_dict = self.net[layer_id+1][0][0]
                if next_layer_dict['type'] == 'conv' or next_layer_dict['type'] == 'pooling':
                    self.trans_time[0][layer_id] = int(layer_dict['Outputsize'][1]) * \
                                                   max(int(next_layer_dict['Kernelsize'])-int(next_layer_dict['Padding'])-1, 0) +\
                                                   max(int(next_layer_dict['Kernelsize'])-int(next_layer_dict['Padding'])-1, 0)
                elif next_layer_dict['type'] == 'fc':
                    self.trans_time[0][layer_id] = 1
            tmp_bankinfo['PEnum'] = tmp_bankinfo['mx'] * tmp_bankinfo['my'] * multiple[layer_id]
            tmp_bankinfo['banknum'] = math.ceil(tmp_bankinfo['PEnum'] / self.bank.bank_PE_total_num)
            tmp_bankinfo['max_PE'] = min(tmp_bankinfo['PEnum'], self.bank.bank_PE_total_num)
            start_bankid += tmp_bankinfo['banknum']
            self.layer_bankinfo.append(tmp_bankinfo)
        self.bank_num = start_bankid
        assert self.bank_num <= self.bank_total_num, "Bank number is not enough"
        self.inLayer_distance = np.ones([1, self.layer_num])
        self.transLayer_distance = np.ones([1, self.layer_num])
        self.aggregate_arg = np.zeros([self.layer_num,2])

    def mapping_matrix_gen(self):
        if self.bank_connection == 0:
            self.mapping_order = generate_normal_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.bank_connection == 1:
            self.mapping_order = generate_snake_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.bank_connection == 2:
            self.mapping_order = generate_hui_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.bank_connection == 3:
            self.mapping_order = generate_zigzag_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])

    def mapping_net(self):
        self.mapping_matrix_gen()
        for i in range(self.mapping_order.shape[0]):
            for j in range(self.mapping_order.shape[1]):
                if self.mapping_order[i][j] < self.bank_num:
                    for layer_id in range(self.layer_num-1):
                        if ((self.mapping_order[i][j] >= self.layer_bankinfo[layer_id]['startid']) &
                            (self.mapping_order[i][j] < self.layer_bankinfo[layer_id+1]['startid'])):
                            self.mapping_result[i][j] = layer_id
                            break
                        elif self.mapping_order[i][j] >= self.layer_bankinfo[self.layer_num-1]['startid']:
                            self.mapping_result[i][j] = self.layer_num-1

    def calculate_transfer_distance(self):
        for layer_id in range(self.layer_num-1):
            # Determine the aggregate node for layer 0~N-1
            src_pos = np.argwhere(self.mapping_result == layer_id)
            dst_pos = np.argwhere(self.mapping_result == layer_id+1)
            if len(src_pos) == 1:
                self.inLayer_distance[0][layer_id] = 0
                self.aggregate_arg[layer_id] = src_pos[0]
                maxdis = 0
                for i in range(len(dst_pos)):
                    dis = abs(src_pos[0][0]-dst_pos[i][0]) + abs(src_pos[0][1]-dst_pos[i][1])
                    if dis > maxdis:
                        maxdis = dis
                self.transLayer_distance[0][layer_id] = maxdis
            else:
                mindis_total = 100
                for A in range(len(src_pos)):
                    maxdis_in = 0
                    for i in range(len(src_pos)):
                        if i != A:
                            dis_in = abs(src_pos[A][0]-src_pos[i][0]) + abs(src_pos[A][1]-src_pos[i][1])
                            if dis_in > maxdis_in:
                                maxdis_in = dis_in
                    maxdis_out = 0
                    for j in range(len(dst_pos)):
                        dis_out = abs(src_pos[A][0] - dst_pos[j][0]) + abs(src_pos[A][1] - dst_pos[j][1])
                        if dis_out > maxdis_out:
                            maxdis_out= dis_out
                    tempdis = maxdis_in + maxdis_out
                    if tempdis < mindis_total:
                        self.inLayer_distance[0][layer_id] = maxdis_in
                        self.transLayer_distance[0][layer_id] = maxdis_out
                        self.aggregate_arg[layer_id] = src_pos[A]
                        mindis_total = tempdis
        final_pos = np.argwhere(self.mapping_result == self.layer_num-1)
            # Determine the aggregate node for layer N (output layer)
        mindis = 100
        for i in range(len(final_pos)):
            maxdis = 0
            for j in range(len(final_pos)):
                if j != i:
                    dis = abs(final_pos[i][0]-final_pos[j][0])+abs(final_pos[i][1]-final_pos[j][1])
                    if dis > maxdis:
                        maxdis = dis
            if maxdis < mindis:
                mindis = maxdis
                self.inLayer_distance[0][self.layer_num-1] = mindis
                self.aggregate_arg[self.layer_num-1] = final_pos[i]
                self.transLayer_distance[0][self.layer_num-1] = 0
        self.total_distance = sum(sum(self.trans_time * (self.inLayer_distance+self.transLayer_distance)))

if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "cifar10_vgg8_params.pth")

    __TestInterface = TrainTestInterface('vgg8', 'MNSIM.Interface.cifar10', test_SimConfig_path, test_weights_file_path, 'cpu')
    structure_file = __TestInterface.get_structure()

    test = BCG(structure_file, test_SimConfig_path)
    test.mapping_net()
    test.calculate_transfer_distance()
    print(test.total_distance)

