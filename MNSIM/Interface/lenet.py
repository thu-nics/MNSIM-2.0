#-*-coding:utf-8-*-
from MNSIM.Interface import quantize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import copy
import re

class LeNet(nn.Module):
    def __init__(self, hardware_config, num_classes):
        super(LeNet, self).__init__()
        # hardware_config = {'fix_method': 'SINGLE_FIX_TEST', 'xbar_size': 512, 'input_bit': 2, 'weight_bit': 1, 'quantize_bit': 10}
        # hardware_config = {'fix_method': 'FIX_TRAIN', 'xbar_size': 512, 'input_bit': 2, 'quantize_bit': 10}
        self.hardware_config = copy.deepcopy(hardware_config)
        quantize_config = {'weight_bit': 9, 'activation_bit': 9, 'point_shift': -2}
        # conv layer 1
        c1_layer_config = {'type': 'conv', 'in_channels': 3, 'out_channels': 6, 'kernel_size': 5}
        self.c1 = quantize.QuantizeLayer(hardware_config, c1_layer_config, quantize_config)
        self.relu1 = nn.ReLU()
        self.s1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # conv layer 2
        c2_layer_config = {'type': 'conv', 'in_channels': 6, 'out_channels': 16, 'kernel_size': 5}
        self.c2 = quantize.QuantizeLayer(hardware_config, c2_layer_config, quantize_config)
        self.relu2 = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # conv layer 3
        c3_layer_config = {'type': 'conv', 'in_channels': 16, 'out_channels': 120, 'kernel_size': 5}
        self.c3 = quantize.QuantizeLayer(hardware_config, c3_layer_config, quantize_config)
        self.relu3 = nn.ReLU()
        # fc layer 4
        fc4_layer_config = {'type': 'fc', 'in_features': 120, 'out_features': 84}
        self.fc4 = quantize.QuantizeLayer(hardware_config, fc4_layer_config, quantize_config)
        self.relu4 = nn.ReLU()
        # fc layer 4
        fc5_layer_config = {'type': 'fc', 'in_features': 84, 'out_features': num_classes}
        self.fc5 = quantize.QuantizeLayer(hardware_config, fc5_layer_config, quantize_config)

    def forward(self, x, method = 'SINGLE_FIX_TEST', adc_action = 'SCALE'):
        # input fix information
        quantize.last_activation_scale = 1. / 255.
        quantize.last_activation_bit = 9
        # forward
        x = self.s1(self.relu1(self.c1(x, method, adc_action)))
        x = self.s2(self.relu2(self.c2(x, method, adc_action)))
        x = self.relu3(self.c3(x, method, adc_action))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc4(x, method, adc_action))
        x = self.fc5(x, method, adc_action)
        return x
    def get_weights(self):
        net_bit_weights = []
        for module in self.modules():
            if isinstance(module, quantize.QuantizeLayer):
                net_bit_weights.append(module.get_bit_weights())
        return net_bit_weights
    def set_weights_forward(self, x, net_bit_weights, adc_action = 'SCALE'):
        # input fix information
        quantize.last_activation_scale = 1. / 255.
        quantize.last_activation_bit = 9
        # forward
        x = self.s1(self.relu1(self.c1.set_weights_forward(x, net_bit_weights[0], adc_action)))
        x = self.s2(self.relu2(self.c2.set_weights_forward(x, net_bit_weights[1], adc_action)))
        x = self.relu3(self.c3.set_weights_forward(x, net_bit_weights[2], adc_action))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc4.set_weights_forward(x, net_bit_weights[3], adc_action))
        x = self.fc5.set_weights_forward(x, net_bit_weights[4], adc_action)
        return x
    def get_structure(self):
        # forward structure
        x = torch.zeros((1, 3, 32, 32))
        x = self.s1(self.relu1(self.c1.structure_forward(x)))
        x = self.s2(self.relu2(self.c2.structure_forward(x)))
        x = self.relu3(self.c3.structure_forward(x))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc4.structure_forward(x))
        x = self.fc5.structure_forward(x)
        # structure information, stored as list
        net_info = []
        for module in self.modules():
            if isinstance(module, quantize.QuantizeLayer):
                net_info.append(module.layer_info)
        return net_info
    def load_change_weights(self, state_dict):
        # input is a state dict, weights
        # concat all layer_list weights
        keys_map = collections.OrderedDict()
        for key in state_dict.keys():
            tmp_key = re.sub('\.layer_list\.\d+\.weight$', '', key)
            if tmp_key not in keys_map.keys():
                keys_map[tmp_key] = [key]
            else:
                keys_map[tmp_key].append(key)
        # concat and split
        tmp_state_dict = collections.OrderedDict()
        for tmp_key, key_list in keys_map.items():
            if len(key_list) == 1 and tmp_key == key_list[0]:
                print('origin weights')
                tmp_state_dict[tmp_key] = state_dict[key_list[0]]
            else:
                print(f'transfer weights {tmp_key}')
                # get layer info
                layer_config = None
                hardware_config = None
                for name, module in self.named_children():
                    if name == tmp_key:
                        layer_config = module.layer_config
                        hardware_config = module.hardware_config
                        layer_list_len = len(module.layer_list)
                assert layer_config, 'layer must have layer config'
                assert hardware_config, 'layer must have hardware config'
                # concat weights
                total_weights = torch.cat([state_dict[key] for key in key_list])
                # split weights
                if layer_config['type'] == 'conv':
                    split_len = (hardware_config['xbar_size'] // (layer_config['kernel_size'] ** 2))
                elif layer_config['type'] == 'fc':
                    split_len = hardware_config['xbar_size']
                else:
                    raise NotImplementedError
                weights_list = torch.split(total_weights, split_len, dim = 1)
                # load weights
                for i, weights in enumerate(weights_list):
                    tmp_state_dict[tmp_key + f'.layer_list.{i}.weight'] = weights
        # load weights
        self.load_state_dict(tmp_state_dict)

def get_net(hardware_config = None):
    # initial config
    if hardware_config == None:
        hardware_config = {'xbar_size': 512, 'input_bit': 2, 'weight_bit': 1, 'quantize_bit': 10}
    net = LeNet(hardware_config, 10)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('this is lenet input shape 3x32x32，output shape 10')
