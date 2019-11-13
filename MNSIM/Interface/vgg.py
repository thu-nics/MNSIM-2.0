#-*-coding:utf-8-*-
from MNSIM.Interface import quantize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import copy
import re

class VGG(nn.Module):
    def __init__(self, hardware_config, num_classes):
        super(VGG, self).__init__()
        # hardware_config = {'fix_method': 'SINGLE_FIX_TEST', 'xbar_size': 512, 'input_bit': 2, 'weight_bit': 1, 'quantize_bit': 10}
        # hardware_config = {'fix_method': 'FIX_TRAIN', 'xbar_size': 512, 'input_bit': 2, 'quantize_bit': 10}
        self.hardware_config = copy.deepcopy(hardware_config)
        quantize_config = {'weight_bit': 9, 'activation_bit': 9, 'point_shift': -2}
        # conv layer 1_1
        conv1_1_layer_config = {'type': 'conv', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'padding': 1}
        self.conv1_1 = quantize.QuantizeLayer(hardware_config, conv1_1_layer_config, quantize_config)
        self.relu1_1 = nn.ReLU()
        # conv layer 1_2
        conv1_2_layer_config = {'type': 'conv', 'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'padding': 1}
        self.conv1_2 = quantize.QuantizeLayer(hardware_config, conv1_2_layer_config, quantize_config)
        self.relu1_2 = nn.ReLU()
        # conv layer 2_1
        conv2_1_layer_config = {'type': 'conv', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}
        self.conv2_1 = quantize.QuantizeLayer(hardware_config, conv2_1_layer_config, quantize_config)
        self.relu2_1 = nn.ReLU()
        # conv layer 2_2
        conv2_2_layer_config = {'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}
        self.conv2_2 = quantize.QuantizeLayer(hardware_config, conv2_2_layer_config, quantize_config)
        self.relu2_2 = nn.ReLU()
        # conv layer 3_1
        conv3_1_layer_config = {'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1}
        self.conv3_1 = quantize.QuantizeLayer(hardware_config, conv3_1_layer_config, quantize_config)
        self.relu3_1 = nn.ReLU()
        # conv layer 3_2
        conv3_2_layer_config = {'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1}
        self.conv3_2 = quantize.QuantizeLayer(hardware_config, conv3_2_layer_config, quantize_config)
        self.relu3_2 = nn.ReLU()
        # conv layer 3_3
        conv3_3_layer_config = {'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1}
        self.conv3_3 = quantize.QuantizeLayer(hardware_config, conv3_3_layer_config, quantize_config)
        self.relu3_3 = nn.ReLU()
        # conv layer 4_1
        conv4_1_layer_config = {'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1}
        self.conv4_1 = quantize.QuantizeLayer(hardware_config, conv4_1_layer_config, quantize_config)
        self.relu4_1 = nn.ReLU()
        # conv layer 4_2
        conv4_2_layer_config = {'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1}
        self.conv4_2 = quantize.QuantizeLayer(hardware_config, conv4_2_layer_config, quantize_config)
        self.relu4_2 = nn.ReLU()
        # conv layer 4_3
        conv4_3_layer_config = {'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1}
        self.conv4_3 = quantize.QuantizeLayer(hardware_config, conv4_3_layer_config, quantize_config)
        self.relu4_3 = nn.ReLU()
        # conv layer 5_1
        conv5_1_layer_config = {'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1}
        self.conv5_1 = quantize.QuantizeLayer(hardware_config, conv5_1_layer_config, quantize_config)
        self.relu5_1 = nn.ReLU()
        # conv layer 5_2
        conv5_2_layer_config = {'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1}
        self.conv5_2 = quantize.QuantizeLayer(hardware_config, conv5_2_layer_config, quantize_config)
        self.relu5_2 = nn.ReLU()
        # conv layer 5_5
        conv5_3_layer_config = {'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1}
        self.conv5_3 = quantize.QuantizeLayer(hardware_config, conv5_3_layer_config, quantize_config)
        self.relu5_3 = nn.ReLU()
        # fc layer 6
        fc6_layer_config = {'type': 'fc', 'in_features': 512, 'out_features': 128}
        self.fc6 = quantize.QuantizeLayer(hardware_config, fc6_layer_config, quantize_config)
        self.fc6_relu = nn.ReLU()
        self.fc6_dropout = nn.Dropout()
        # fc layer 7
        fc7_layer_config = {'type': 'fc', 'in_features': 128, 'out_features': num_classes}
        self.fc7 = quantize.QuantizeLayer(hardware_config, fc7_layer_config, quantize_config)

    def forward(self, x, method = 'SINGLE_FIX_TEST', adc_action = 'SCALE'):
        # input fix information
        quantize.last_activation_scale = 1. / 255.
        quantize.last_activation_bit = 9
        # forward
        x = self.relu1_1(self.conv1_1(x, method, adc_action))
        x = self.relu1_2(self.conv1_2(x, method, adc_action))
        x = F.max_pool2d(x, 2)

        x = self.relu2_1(self.conv2_1(x, method, adc_action))
        x = self.relu2_2(self.conv2_2(x, method, adc_action))
        x = F.max_pool2d(x, 2)

        x = self.relu3_1(self.conv3_1(x, method, adc_action))
        x = self.relu3_2(self.conv3_2(x, method, adc_action))
        x = self.relu3_3(self.conv3_3(x, method, adc_action))
        x = F.max_pool2d(x, 2)

        x = self.relu4_1(self.conv4_1(x, method, adc_action))
        x = self.relu4_2(self.conv4_2(x, method, adc_action))
        x = self.relu4_3(self.conv4_3(x, method, adc_action))
        x = F.max_pool2d(x, 2)

        x = self.relu5_1(self.conv5_1(x, method, adc_action))
        x = self.relu5_2(self.conv5_2(x, method, adc_action))
        x = self.relu5_3(self.conv5_3(x, method, adc_action))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = self.fc6_dropout(self.fc6_relu(self.fc6(x, method, adc_action)))
        x = self.fc7(x, method, adc_action)
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
        x = self.relu1_1(self.conv1_1(x, net_bit_weights[0], adc_action))
        x = self.relu1_2(self.conv1_2(x, net_bit_weights[1], adc_action))
        x = F.max_pool2d(x, 2)

        x = self.relu2_1(self.conv2_1(x, net_bit_weights[2], adc_action))
        x = self.relu2_2(self.conv2_2(x, net_bit_weights[3], adc_action))
        x = F.max_pool2d(x, 2)

        x = self.relu3_1(self.conv3_1(x, net_bit_weights[4], adc_action))
        x = self.relu3_2(self.conv3_2(x, net_bit_weights[5], adc_action))
        x = self.relu3_3(self.conv3_3(x, net_bit_weights[6], adc_action))
        x = F.max_pool2d(x, 2)

        x = self.relu4_1(self.conv4_1(x, net_bit_weights[7], adc_action))
        x = self.relu4_2(self.conv4_2(x, net_bit_weights[8], adc_action))
        x = self.relu4_3(self.conv4_3(x, net_bit_weights[9], adc_action))
        x = F.max_pool2d(x, 2)

        x = self.relu5_1(self.conv5_1(x, net_bit_weights[10], adc_action))
        x = self.relu5_2(self.conv5_2(x, net_bit_weights[11], adc_action))
        x = self.relu5_3(self.conv5_3(x, net_bit_weights[12], adc_action))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = self.fc6_dropout(self.fc6_relu(self.fc6(x, net_bit_weights[13], adc_action)))
        x = self.fc7(x, net_bit_weights[14], adc_action)
        return x
    def get_structure(self):
        # forward structure
        x = torch.zeros((1, 3, 32, 32))
        x = self.relu1_1(self.conv1_1.structure_forward(x))
        x = self.relu1_2(self.conv1_2.structure_forward(x))
        x = F.max_pool2d(x, 2)

        x = self.relu2_1(self.conv2_1.structure_forward(x))
        x = self.relu2_2(self.conv2_2.structure_forward(x))
        x = F.max_pool2d(x, 2)

        x = self.relu3_1(self.conv3_1.structure_forward(x))
        x = self.relu3_2(self.conv3_2.structure_forward(x))
        x = self.relu3_3(self.conv3_3.structure_forward(x))
        x = F.max_pool2d(x, 2)

        x = self.relu4_1(self.conv4_1.structure_forward(x))
        x = self.relu4_2(self.conv4_2.structure_forward(x))
        x = self.relu4_3(self.conv4_3.structure_forward(x))
        x = F.max_pool2d(x, 2)

        x = self.relu5_1(self.conv5_1.structure_forward(x))
        x = self.relu5_2(self.conv5_2.structure_forward(x))
        x = self.relu5_3(self.conv5_3.structure_forward(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = self.fc6_dropout(self.fc6_relu(self.fc6.structure_forward(x)))
        x = self.fc7.structure_forward(x)
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
    net = VGG(hardware_config, 10)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('this is vgg input shape 3x32x32，output shape 10')
