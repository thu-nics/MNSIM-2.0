#-*-coding:utf-8-*-
import math
import numpy as np
import torch
import collections
import configparser
from importlib import import_module
import copy

class TrainTestInterface(object):
    def __init__(self, netwotk_module, dataset_module, SimConfig_path, weights_file, device = None):
        # netwotk_module = 'lenet'
        # dataset_module = 'cifar10'
        # weights_file = './zoo/cifar10_lenet_train_params.pth'
        # load net, dataset, and weights
        self.test_loader = import_module(dataset_module).get_dataloader()[1]
        # load simconfig
        ## xbar_size, input_bit, weight_bit, quantize_bit
        xbar_config = configparser.ConfigParser()
        xbar_config.read(SimConfig_path, encoding = 'UTF-8')
        self.hardware_config = collections.OrderedDict()
        self.hardware_config['fix_method'] = 'SINGLE_FIX_TEST'
        # xbar_size
        xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.xbar_row = xbar_size[0]
        self.xbar_column = xbar_size[1]
        self.hardware_config['xbar_size'] = xbar_size[0]
        # xbar bit
        self.xbar_bit = int(xbar_config.get('Device level', 'Device_Level'))
        self.hardware_config['weight_bit'] = self.xbar_bit
        # input bit and ADC bit
        self.input_bit = 2
        self.quantize_bit = 10
        self.hardware_config['input_bit'] = self.input_bit
        self.hardware_config['quantize_bit'] = self.quantize_bit
        # group num
        self.pe_group_num = int(xbar_config.get('Process element level', 'Group_Num'))
        self.bank_size = list(map(int, xbar_config.get('Bank level', 'PE_Num').split(',')))
        self.bank_row = self.bank_size[0]
        self.bank_column = self.bank_size[1]
        # net and weights
        self.net = import_module(netwotk_module).get_net(self.hardware_config)
        self.net.load_state_dict(torch.load(weights_file))
        if device != None:
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
    def origin_evaluate(self):
        self.net.to(self.device)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                test_total += labels.size(0)
                outputs = self.net(images)
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        return test_correct / test_total
    def get_net_bits(self):
        net_bit_weights = self.net.get_weights()
        return net_bit_weights
    def set_net_bits_evaluate(self, net_bit_weights):
        self.net.to(self.device)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                test_total += labels.size(0)
                outputs = self.net.set_weights_forward(images, net_bit_weights)
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        return test_correct / test_total
    def get_structure(self):
        net_bit_weights = self.net.get_weights()
        net_structure_info = self.net.get_structure()
        assert len(net_bit_weights) == len(net_structure_info)
        total_array = []
        for layer_num, (layer_bit_weights, layer_structure_info) in enumerate(zip(net_bit_weights, net_structure_info)):
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
            L = math.ceil(len(xbar_array) / (self.bank_row * self.bank_column))
            for i in range(L):
                bank_array = []
                for h in range(self.bank_row):
                    for w in range(self.bank_column):
                        serial_number = i * self.bank_row * self.bank_column + h * self.bank_column + w
                        if serial_number < len(xbar_array):
                            bank_array.append(xbar_array[serial_number])
                total_array.append((layer_structure_info, bank_array))
        return total_array

def mysplit(array, length):
    L = array.shape[1] // length
    if L > 0:
        return np.split(array[:,:(L*length)], length, axis = 1).append(array[:,(L*length):])
    else:
        return array

if __name__ == '__main__':
    pass
