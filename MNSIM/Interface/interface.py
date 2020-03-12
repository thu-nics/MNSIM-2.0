#-*-coding:utf-8-*-
import collections
import configparser
import copy
import math
import os
from importlib import import_module

import numpy as np
import torch


class TrainTestInterface(object):
    def __init__(self, network_module, dataset_module, SimConfig_path, weights_file, device = None, extra_define = None):
        # network_module = 'lenet'
        # dataset_module = 'cifar10'
        # weights_file = './zoo/cifar10_lenet_train_params.pth'
        # load net, dataset, and weights
        self.network_module = network_module
        self.dataset_module = dataset_module
        self.weights_file = weights_file
        self.test_loader = import_module(dataset_module).get_dataloader()[1]
        # load simconfig
        ## xbar_size, input_bit, weight_bit, quantize_bit
        xbar_config = configparser.ConfigParser()
        xbar_config.read(SimConfig_path, encoding = 'UTF-8')
        self.hardware_config = collections.OrderedDict()
        # xbar_size
        xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.xbar_row = xbar_size[0]
        self.xbar_column = xbar_size[1]
        self.hardware_config['xbar_size'] = xbar_size[0]
        # xbar bit
        self.xbar_bit = int(xbar_config.get('Device level', 'Device_Level'))
        self.hardware_config['weight_bit'] = math.floor(math.log2(self.xbar_bit))
        # input bit and ADC bit
        ADC_choice = int(xbar_config.get('Interface level', 'ADC_Choice'))
        DAC_choice = int(xbar_config.get('Interface level', 'DAC_Choice'))
        temp_DAC_bit = int(xbar_config.get('Interface level', 'DAC_Precision'))
        temp_ADC_bit = int(xbar_config.get('Interface level', 'ADC_Precision'))
        ADC_precision_dict = {
            -1: temp_ADC_bit,
            1: 10,
            # reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
            2: 8,
            # reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
            3: 8,  # reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
            4: 6,  # reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
            5: 8,  # ASPDAC1
            6: 6,  # ASPDAC2
            7: 4  # ASPDAC3
        }
        DAC_precision_dict = {
            -1: temp_DAC_bit,
            1: 1,  # 1-bit
            2: 2,  # 2-bit
            3: 3,  # 3-bit
            4: 4,  # 4-bit
            5: 6,  # 6-bit
            6: 8  # 8-bit
        }
        self.input_bit = DAC_precision_dict[DAC_choice]
        self.quantize_bit = ADC_precision_dict[ADC_choice]
        self.hardware_config['input_bit'] = self.input_bit
        self.hardware_config['quantize_bit'] = self.quantize_bit
        # group num
        self.pe_group_num = int(xbar_config.get('Process element level', 'Group_Num'))
        self.tile_size = list(map(int, xbar_config.get('Tile level', 'PE_Num').split(',')))
        self.tile_row = self.tile_size[0]
        self.tile_column = self.tile_size[1]
        # net and weights
        if device != None:
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        if dataset_module.endswith('cifar10'):
            num_classes = 10
        elif dataset_module.endswith('cifar100'):
            num_classes = 100
        else:
            assert 0, f'unknown dataset'
        if extra_define != None:
            self.hardware_config['input_bit'] = extra_define['dac_res']
            self.hardware_config['quantize_bit'] = extra_define['adc_res']
            self.hardware_config['xbar_size'] = extra_define['xbar_size']
        self.net = import_module('MNSIM.Interface.network').get_net(self.hardware_config, cate = self.network_module, num_classes = num_classes)
        self.net.load_change_weights(torch.load(weights_file, map_location=self.device))
        # if device != None:
        #     self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        # else:
        #     self.device = torch.device('cpu')
    def origin_evaluate(self, method = 'SINGLE_FIX_TEST', adc_action = 'SCALE'):
        self.net.to(self.device)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if i > 10:
                    break
                images = images.to(self.device)
                test_total += labels.size(0)
                outputs = self.net(images, method, adc_action)
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        return test_correct / test_total
    def get_net_bits(self):
        net_bit_weights = self.net.get_weights()
        return net_bit_weights
    def set_net_bits_evaluate(self, net_bit_weights, adc_action = 'SCALE'):
        self.net.to(self.device)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if i > 10:
                    break
                images = images.to(self.device)
                test_total += labels.size(0)
                outputs = self.net.set_weights_forward(images, net_bit_weights, adc_action)
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        return test_correct / test_total
    def get_structure(self):
        net_bit_weights = self.net.get_weights()
        net_structure_info = self.net.get_structure()
        assert len(net_bit_weights) == len(net_structure_info)
        net_array = []
        for layer_num, (layer_bit_weights, layer_structure_info) in enumerate(zip(net_bit_weights, net_structure_info)):
            total_array = []
            if layer_bit_weights == None:
                if layer_structure_info['type'] == 'pooling':
                    total_array.append((layer_structure_info, None))
                    net_array.append(total_array)
                continue
            assert len(layer_bit_weights.keys()) == layer_structure_info['row_split_num'] * layer_structure_info['weight_cycle'] * 2
            layer_structure_info['Layernum'] = layer_num
            # split
            for i in range(layer_structure_info['row_split_num']):
                for j in range(layer_structure_info['weight_cycle']):
                    layer_bit_weights[f'split{i}_weight{j}_positive'] = mysplit(layer_bit_weights[f'split{i}_weight{j}_positive'], self.xbar_column)
                    layer_bit_weights[f'split{i}_weight{j}_negative'] = mysplit(layer_bit_weights[f'split{i}_weight{j}_negative'], self.xbar_column)
            # generate pe array
            xbar_array = []
            for i in range(layer_structure_info['row_split_num']):
                L = len(layer_bit_weights[f'split{i}_weight{0}_positive'])
                for j in range(L):
                    pe_array = []
                    for s in range(layer_structure_info['weight_cycle']):
                        pe_array.append([layer_bit_weights[f'split{i}_weight{s}_positive'][j].astype(np.uint8), layer_bit_weights[f'split{i}_weight{s}_negative'][j].astype(np.uint8)])
                    xbar_array.append(pe_array)
            # store in xbar_array
            L = math.ceil(len(xbar_array) / (self.tile_row * self.tile_column))
            for i in range(L):
                tile_array = []
                for h in range(self.tile_row):
                    for w in range(self.tile_column):
                        serial_number = i * self.tile_row * self.tile_column + h * self.tile_column + w
                        if serial_number < len(xbar_array):
                            tile_array.append(xbar_array[serial_number])
                total_array.append((layer_structure_info, tile_array))
            net_array.append(total_array)
        return net_array

def mysplit(array, length):
    # reshape
    array = np.reshape(array, (array.shape[0], -1))
    # split on output
    assert array.shape[0] > 0
    tmp_index = []
    for i in range(1, array.shape[0]):
        if i % length == 0:
            tmp_index.append(i)
    return np.split(array, tmp_index, axis = 0)

if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SimConfig.ini")
    # test_weights_file_path = os.path.join(os.path.dirname(__file__), "zoo/lenet_6_9_params.pth")
    # __TestInterface = TrainTestInterface('lenet_6_9', 'MNSIM.Interface.cifar10', test_SimConfig_path, test_weights_file_path, '1')
    # print(__TestInterface.origin_evaluate(method='SINGLE_FIX_TEST'))
    # print(__TestInterface.set_net_bits_evaluate(__TestInterface.get_net_bits()))
    # structure_info = __TestInterface.get_structure()
    # traverse all kinds
    net_list = ['lenet', 'alexnet', 'vgg8']
    xbar_size_list = [128, 256, 512, 1024]
    adc_res_list = [4, 6, 8, 10]
    dac_res_list = [1, 2, 4]
    for net in net_list:
        for xbar_size in xbar_size_list:
            for adc_res in adc_res_list:
                for dac_res in dac_res_list:
                    # 正常情况下阻止循环的遍历
                    continue
                    tmp_config = collections.OrderedDict()
                    tmp_config['xbar_size'] = xbar_size
                    tmp_config['adc_res'] = adc_res
                    tmp_config['dac_res'] = dac_res
                    # test_weight
                    test_weights_file_path = os.path.join(os.path.dirname(__file__), f'zoo/cifar10_{net}_params.pth')
                    __TestInterface = TrainTestInterface(net, 'MNSIM.Interface.cifar10', test_SimConfig_path, test_weights_file_path, '1', tmp_config)
                    accuracy = __TestInterface.origin_evaluate(method='SINGLE_FIX_TEST')
                    print(f'{net} {xbar_size} {adc_res} {dac_res} {accuracy}', flush=True)