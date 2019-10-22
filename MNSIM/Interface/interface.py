#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import math
import numpy as np
import torch
import collections
import json
import configparser

class TrainInterface(object):
    def __init__(self, SimConfig_path, net_superparams_path, net_weights_path):
        # config parser
        xbar_config = configparser.ConfigParser()
        xbar_config.read(SimConfig_path, encoding = 'UTF-8')
        self.xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.xbar_row = self.xbar_size[0]
        self.xbar_column = self.xbar_size[1]
        self.xbar_bit = 1
        self.pe_group_num = int(xbar_config.get('Process element level', 'Group_Num'))
        self.bank_size = list(map(int, xbar_config.get('Bank level', 'PE_Num').split(',')))
        self.bank_row = self.bank_size[0]
        self.bank_column = self.bank_size[1]
        # net super params load and weights load
        f = open(net_superparams_path)
        self.net_superparams = json.loads(f.read())
        f.close()
        self.net_weights = torch.load(net_weights_path)
        self.__deploy_params = []
        self.__deploy_flag = False
    def get_structure(self):
        if not self.__deploy_flag:
            self.__generate_structure()
        return self.__deploy_params
    def __generate_structure(self):
        for layer_param in self.net_superparams:
            # read weight and check weight shape
            layer_weight = self.net_weights[layer_param['Weightname']].detach().cpu()
            assert layer_weight.shape[0] == layer_param['Outputchannel']
            assert layer_weight.shape[1] == layer_param['Inputchannel']
            assert layer_weight.shape[2] == layer_param['Kernelsize']
            assert layer_weight.shape[3] == layer_param['Kernelsize']
            # change float weight to fix point weight
            scale = torch.max(torch.abs(layer_weight))
            thres = 2 ** (layer_param['Weightbit'] - 1) - 1
            layer_weight.div_(scale).mul_(thres).round_()
            layer_weight = layer_weight.reshape(layer_weight.size(0), -1).t()
            # fix point weight to split weight
            sign_weight = torch.sign(layer_weight)
            sign_positive = (sign_weight == 1).float()
            sign_negative = (sign_weight == -1).float()
            abs_weight = torch.abs(layer_weight)
            pn_weight_list = []
            assert (layer_param['Weightbit'] - 1) % self.xbar_bit == 0
            bit_base = 2 ** (self.xbar_bit)
            for i in range((layer_param['Weightbit'] - 1) // self.xbar_bit):
                tmp = torch.fmod(abs_weight, bit_base)
                positive = torch.mul(tmp, sign_positive)
                negative = torch.mul(tmp, sign_negative)
                pn_weight_list.append([positive, negative])
                abs_weight.div_(bit_base).floor_()
            # split channel for large input or output channel
            weight_h = layer_weight.shape[0]
            weight_w = layer_weight.shape[1]
            h_range = math.ceil(weight_h / (self.bank_row * self.xbar_row))
            w_range = math.ceil(weight_w / (self.bank_column * self.xbar_column))
            xbar_array = []
            for h in range(h_range):
                for w in range(w_range):
                    base_h = h * self.bank_row * self.xbar_row
                    base_w = w * self.bank_column * self.xbar_column
                    total_h = min(weight_h, (h + 1) * self.bank_row * self.xbar_row) - base_h
                    total_w = min(weight_w, (w + 1) * self.bank_column * self.xbar_column) - base_w
                    # split in bank
                    bank_h_range = math.ceil(total_h / self.xbar_row)
                    bank_w_range = math.ceil(total_w / self.xbar_column)
                    for bank_h in range(bank_h_range):
                        for bank_w in range(bank_w_range):
                            bank_base_h = base_h + bank_h * self.xbar_row
                            bank_base_w = base_w + bank_w * self.xbar_column
                            bank_total_h = min(weight_h, bank_base_h + self.xbar_row) - bank_base_h
                            bank_total_w = min(weight_w, bank_base_w + self.xbar_column) - bank_base_w
                            # copy weight data to pe array
                            pe_array = []
                            for i in range((layer_param['Weightbit'] - 1) // self.xbar_bit):
                                tmp_positive = pn_weight_list[i][0][bank_base_h : (bank_base_h + bank_total_h), bank_base_w : (bank_base_w + bank_total_w)].clone()
                                tmp_negative = pn_weight_list[i][1][bank_base_h : (bank_base_h + bank_total_h), bank_base_w : (bank_base_w + bank_total_w)].clone()
                                pe_array.append([tmp_positive.numpy().astype(np.uint8), tmp_negative.numpy().astype(np.uint8)])
                            # copy data to cross array
                            xbar_array.append(pe_array)
            # rearrange for pe bank
            L = math.ceil(len(xbar_array) / (self.bank_row * self.bank_column))
            for i in range(L):
                bank_array = []
                for h in range(self.bank_row):
                    for w in range(self.bank_column):
                        serial_number = i * self.bank_row * self.bank_column + h * self.bank_column + w
                        if serial_number < len(xbar_array):
                            bank_array.append(xbar_array[serial_number])
                # save for every bank array
                layer_info = collections.OrderedDict()
                for key in ['Layernum', 'Inputsize', 'Outputsize', 'Kernelsize', 'Stride', 'Inputchannel', 'Outputchannel', 'Inputbit', 'Weightbit', 'Outputbit']:
                    layer_info[key] = layer_param[key]
                self.__deploy_params.append((layer_info, bank_array))
        self.__deploy_flag = True
